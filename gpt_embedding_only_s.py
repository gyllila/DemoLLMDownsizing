import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
x_start = 8
x_idx = []
x_int = []
max_iters = 5000
eval_interval = 50 # 500
learning_rate = 2e-4 # 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
# ------------

torch.manual_seed(1337)
check_point_file = "check_point.pt"

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# here are all the unique characters that occur in this text


if not os.path.exists("tokens.txt"):
    chars = sorted(list(set(text)))
    with open('tokens.txt', 'w', encoding='utf-8') as f:
        for char in chars:
            f.write(char)
else:
    with open('tokens.txt', 'r', encoding='utf-8') as f:
        tokens = f.read()
    chars = list(tokens)


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
        self.token_k_embedding_table = nn.Embedding(vocab_size, head_size)
        self.position_k_embedding_table = nn.Embedding(block_size, head_size)
        self.token_q_embedding_table = nn.Embedding(vocab_size, head_size)
        self.token_v_embedding_table = nn.Embedding(vocab_size, head_size)
        self.position_v_embedding_table = nn.Embedding(block_size, head_size)

    def forward(self, idx):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T = idx.shape
        q = self.token_q_embedding_table(idx[:, [-1]])
        k = self.token_k_embedding_table(idx) + self.position_k_embedding_table(torch.arange(T-1, -1, -1))
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = F.softmax(wei, dim=-1)
        v = self.token_v_embedding_table(idx) + self.position_v_embedding_table(torch.arange(T-1, -1, -1))
        out = wei @ v
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, idx):
        out = torch.cat([h(idx) for h in self.heads], dim=-1)
        return out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)

    def forward(self, inputs):
        x = inputs[0]
        idx = inputs[1]
        x = x + self.sa(idx)
        # x = x + self.ffwd(self.ln2(x))
        return (x, idx)

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
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
        
        x = self.token_embedding_table(idx[:, [-1]]) # (B,1,C) tok_emb 
        x, _ = self.blocks((x, idx)) # (B,T,C)
        x += self.token_ffwd_embedding_table(idx[:, [-1]]) # tok_ffwd_emb
        x = self.ln_f(x) 
        logits = x @ self.token_embedding_table(torch.arange(vocab_size)).T # transpose(-2, -1)

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

if os.path.exists("gpt_trained_s.pt"):
    model.load_state_dict(torch.load("gpt_trained_s.pt"))
elif os.path.exists(check_point_file):
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
            # if x_idx is not None:
            for x_i in range(x_start, block_size//4):
                logits, loss = model(torch.cat((xb[:, :x_i], xb[:, block_size//2:block_size//2+x_i]), dim=0), torch.cat((yb[:, :x_i], yb[:, block_size//2:block_size//2+x_i]), dim=0))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for x_i in range(block_size//4, block_size//2):
                logits, loss = model(xb[:, :x_i], yb[:, :x_i])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for x_i in range(block_size//2, block_size):
                logits, loss = model(xb[:batch_size//2, :x_i], yb[:batch_size//2, :x_i])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                logits, loss = model(xb[batch_size//2:, :x_i], yb[batch_size//2:, :x_i])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
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
            logits, loss = model(xb[:batch_size//2, :], yb[:batch_size//2, :])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            logits, loss = model(xb[batch_size//2:, :], yb[batch_size//2:, :])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                torch.save({'epoch': iter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss_train': losses['train'], 'loss_val': losses['val']}, f"s_iteration_{iter}.pt")

        torch.save(model.state_dict(), "karpathy_gpt.pt")
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
        # if x_idx is not None:
        for x_i in range(x_start, block_size//4):
                logits, loss = model(torch.cat((xb[:, :x_i], xb[:, block_size//2:block_size//2+x_i]), dim=0), torch.cat((yb[:, :x_i], yb[:, block_size//2:block_size//2+x_i]), dim=0))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        for x_i in range(block_size//4, block_size//2):
            logits, loss = model(xb[:, :x_i], yb[:, :x_i])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for x_i in range(block_size//2, block_size):
            logits, loss = model(xb[:batch_size//2, :x_i], yb[:batch_size//2, :x_i])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            logits, loss = model(xb[batch_size//2:, :x_i], yb[batch_size//2:, :x_i])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
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
        logits, loss = model(xb[:batch_size//2, :], yb[:batch_size//2, :])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logits, loss = model(xb[batch_size//2:, :], yb[batch_size//2:, :])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save({'epoch': iter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss_train': losses['train'], 'loss_val': losses['val']}, f"s_iteration_{iter}.pt")


    # generate from the model
    torch.save(model.state_dict(), "gpt_trained_s.pt")

input_text = input("Generate until exit\n")
while input_text != "exit":
    context = torch.as_tensor([encode(input_text)], dtype=torch.long)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    input_text = input("Generate until exit\n")
else:
    exit()
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))




