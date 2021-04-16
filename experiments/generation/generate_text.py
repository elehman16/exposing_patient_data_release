## Edited and reused from  https://github.com/nyu-dl/bert-gen/blob/master/bert-babble.ipynb

import os
import random

import numpy as np
import torch
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer

cuda = torch.cuda.is_available()
print(cuda)

# Load pre-trained model (weights)
model_version = os.environ["MODEL_PATH"]
model = BertForMaskedLM.from_pretrained(model_version)
model.eval()

if cuda:
    model = model.cuda()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(os.environ["TOK_PATH"], do_lower_case=True)


def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]


def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent


CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]


def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]

    return tokenize_batch(batch)


def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


def to_file(sents, file):
    with open(file, "a") as f:
        f.write("\n".join(sents) + "\n")


# Generation modes as functions
import math
import time


def parallel_sequential_generation(
    seed_text,
    batch_size=10,
    max_len=15,
    top_k=0,
    temperature=None,
    max_iter=300,
    burnin=200,
    cuda=False,
    print_every=10,
    verbose=True,
):
    """Generate for one random position at a timestep

    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    #     seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    mask_pos = [i for i, y in enumerate(batch[0]) if y == mask_id]

    for ii in range(max_iter):
        kk = mask_pos[np.random.randint(0, len(mask_pos))]
        for jj in range(batch_size):
            batch[jj][kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(
            out,
            gen_idx=kk,
            top_k=topk if (ii >= burnin) else 0,
            temperature=temperature,
            sample=(ii < burnin),
        )
        for jj in range(batch_size):
            batch[jj][kk] = idxs[jj]

        if verbose and np.mod(ii + 1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[: kk + 1] + ["(*)"] + for_print[kk + 1 :]
            print("iter", ii + 1, " ".join(for_print))

    return untokenize_batch(batch)


def generate(
    n_samples,
    seed_text="[CLS]",
    batch_size=10,
    max_len=25,
    top_k=100,
    temperature=1.0,
    burnin=200,
    max_iter=500,
    cuda=False,
    print_every=1,
):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        batch = parallel_sequential_generation(
            seed_text,
            batch_size=batch_size,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            verbose=False,
        )

        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()

        sentences += batch
    return sentences


n_samples = 50
batch_size = 50
max_len = 100
top_k = 40
temperature = 1.0
leed_out_len = 5  # max_len
burnin = 250
max_iter = 500

# Choose the prefix context

seeds = [
    "[CLS] mr",
    "[CLS] ms",
]

import secrets

dirname = os.environ["OUT_PATH"]

with open(f"{dirname}/test_{secrets.token_hex(16)}.txt", "a") as f:
    for i in range(200):
        if random.random() < 0.3:
            seed_text = np.random.choice(seeds)
            if random.random() < 0.3:
                seed_text += " ." + " [MASK]" * 5 + " is a yo"
        else:
            seed_text = "[CLS]"

        print(seed_text)
        seed_text = seed_text.split()

        torch.cuda.empty_cache()
        bert_sents = generate(
            n_samples,
            seed_text=seed_text,
            batch_size=batch_size,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
        )

        sents = list(map(lambda x: " ".join(detokenize(x)), bert_sents))
        f.write("\n".join(sents) + "\n")
        f.flush()
