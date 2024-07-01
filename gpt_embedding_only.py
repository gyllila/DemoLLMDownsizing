import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?

## stop indices for additional subsamples used to train the model, must be within range(1, block_size)
## example: [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224]
x_idx = []
## intervals for additional subsamples used to train the model, must < block_size
## example: [1,2,4,8,16,32,64,128]
## because the interval of 1 will be drawn block_size times, it's better not to use the above example many times, and be careful if combining x_idx with x_int to avoid repeated draws.
x_int = []
max_iters = 1280000 ## comparable to 5000 max_iters in the original model, can be accordingly less when x_idx or x_int is used.
eval_interval = 20 ## originally 500
learning_rate = 3e-4
device = 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1 ## originally 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))


vocab_size = len(chars) # 65
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        ## Linear modeling of k, q and v replaced by their embeddings.
        self.token_k_embedding_table = nn.Embedding(vocab_size, head_size)
        self.position_k_embedding_table = nn.Embedding(block_size, head_size)
        self.token_q_embedding_table = nn.Embedding(vocab_size, head_size)
        self.token_v_embedding_table = nn.Embedding(vocab_size, head_size)
        self.position_v_embedding_table = nn.Embedding(block_size, head_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, idx):
        B,T = idx.shape
        q = self.token_q_embedding_table(idx[:, [-1]])
        k = self.token_k_embedding_table(idx) + self.position_k_embedding_table(torch.arange(block_size-T, block_size))
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.token_v_embedding_table(idx) + self.position_v_embedding_table(torch.arange(block_size-T, block_size))
        out = wei @ v
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, idx):
        out = torch.cat([h(x, idx) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(n_embd)

    def forward(self, inputs):
        x = inputs[0]
        idx = inputs[1]
        x = x + self.sa(self.ln1(x), idx)
        return (x, idx)

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        ## Replacing FeedForward by embedding table
        self.token_ffwd_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        
        x = self.token_embedding_table(idx[:, [-1]]) ## (B,1,C) tok_emb 
        x, _ = self.blocks((x, idx)) ## (B,1,C)
        x += self.token_ffwd_embedding_table(idx[:, [-1]]) 
        x = self.ln_f(x) 
        logits = x @ self.token_embedding_table(torch.arange(vocab_size)).T 

        if targets is None:
            loss = None
        else:
            B, TT, C = logits.shape
            targets = targets[:, [-1]]
            logits = logits.view(B, C)
            targets = targets.view(B)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()

if os.path.exists("gpt_trained.pt"):
    model.load_state_dict(torch.load("gpt_trained.pt"))
# m = model.to(device)
elif os.path.exists("gpt_iter290600.pt"): ## epoch: 12000
    check_point_file = "gpt_iter290600.pt"
    print(f"loading from {check_point_file}")
    check_point = torch.load(check_point_file)
    model.load_state_dict(check_point["model_state_dict"])
    option=input("generate or iterate?\n")
    
    if option.startswith("i"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        start = int(check_point["epoch"])
        for iter in range(start, max_iters):
            iter += 1

            # sample a batch of data
            xb, yb = get_batch('train')

            # evaluate the loss
            for x_i in x_idx:
                logits, loss = model(xb[:, :x_i], yb[:, :x_i])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for x_i in x_int:
                n_int = block_size//x_i
                for j in range(n_int):
                    logits, loss = model(xb[:, x_i*j:x_i*(j+1)], yb[:, x_i*j:x_i*(j+1)])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                torch.save({'epoch': iter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss_train': losses['train'], 'loss_val': losses['val']}, f"iteration_{iter}.pt")

        torch.save(model.state_dict(), "gpt_trained.pt")
else:
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        iter += 1

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        for x_i in x_idx:
            logits, loss = model(xb[:, :x_i], yb[:, :x_i])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for x_i in x_int:
                n_int = block_size//x_i
                for j in range(n_int):
                    logits, loss = model(xb[:, x_i*j:x_i*(j+1)], yb[:, x_i*j:x_i*(j+1)])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save({'epoch': iter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss_train': losses['train'], 'loss_val': losses['val']}, f"iteration_{iter}.pt")


    torch.save(model.state_dict(), "gpt_trained.pt")

input_text = input("Generate until exit\n")
while input_text != "exit":
    context = torch.as_tensor([encode(input_text)], dtype=torch.long)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    input_text = input("Generate until exit\n")
else:
    exit()





