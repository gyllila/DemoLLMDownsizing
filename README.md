# DemoLLMDownsizing

* Before buying an LLM product like ChatGPT, I first wanted to get well informed and attended the amazing YouTube Course by @karpathy for general audience (https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7), who also kindly made the model files available for practing. I used this one: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
* On my old laptop, I can only use CPU with 4GB RAM and could not run the full model. So I rewrote the model file and reduced the number of parameters from 10M to 2.5M without reducing the quality vis-a-vis the reference text (https://github.com/karpathy/ng-video-lecture/blob/master/more.txt). I think my method could be useful for others, so I uploaded the modified model file (gpt_embedding_only.py). The first takeaway is: buy a new computer if you need to regularly train LLM models. It took me days to train on 290600 data draws (times 64 batches), which corresponds to roughly 1250 iterations (that's a quarter of the recommended maximal iterations) of the original model, also because my laptop doesn't run the code during sleep mode.
* Check the py file and compare it with the original file, you'll see how I did it. If you want to run the py file in your console, you'll have to put it in the same directory as the pt file (gpt_trained.pt). When running, the program will say "Generate until exit" and you can type some text for it to continue with 500 char. Type "exit" to exit. The generated text is hardly readable, though you can recognize the Shakespear writing style. You will also need the original text for training which can be found here: https://github.com/karpathy/ng-video-lecture/blob/master/input.txt
* If you want to train yourself, simply remove the pt file, and if you like you can also replace the input.txt. The python file supports continuation from an existing checkpoint, but unfortunately I couldn't upload my check point file for you because it's too large for GitHub. When running the model anew to continue from the last checkpoint, you have to replace "gpt_iter290600.pt" by the filename of the checkpoint. (It's recommended to also change the seed.) By default, the python file writes a checkpoint file after every 20 iterations. If it's too frequent for you (because you have a faster computer), feel free to change *eval_interval* to a higher value you prefer.
* By default, the model only trains on the last character with the given context length of each batch. If you want to train on more data at each iteration, use *x_idx* or *x_int*. At *block_size = 256*, *x_idx = [1:9]* also trains on the first 8 characters with a context length from 1 to 8, *x_int = [64,128]* makes for each batch 4 additional subsamples of the length 64 and 2 additional subsamples of the length 128, for example. It is recommended to train more intensive on context length between 8 and 32. 
* This (modified) model is for CPU with limited resources, I have no idea how it performs on GPU.
If you aim for higher quality, you can apply this method to a more full-fledged model like nanoGPT by karpathy: https://github.com/karpathy/nanoGPT
* UPDATE: I have further reduced the number of parameters to 1.68M (gpt_embedding_only_s.py, gpt_trained_s.pt), the quality seems unchanged. Example:
```console
The top in a world. You virt, the ban't,
Wallr most us, he hath as from all nock,
The cheke my son me queet, give me in!
For Juwe'le:
Be Lord Cancion just and kind.
If Petter for of death;
Slay of world he while not a beling the sheep;
The Warwick, the sweares pake shall not.
```
😄
