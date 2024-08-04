# %%
with open("input.txt", "r",encoding='utf-8') as  f:
    text = f.read()

# %%
len(text)

# %%
print(text[:1000])

# %%
chars=sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# %%
#map characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #take a string and output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hey there"))
print(decode(encode("hey there")))

# %%
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# %%
#90/10 train/val split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(len(train_data))
print(len(val_data))


# %%
block_size = 8
train_data[:block_size+1]

# %%
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

# %%
torch.manual_seed(1337)
batch_size = 4
block_size = 8 

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print("inputs")
print (xb.shape)
print(yb.shape)
print(xb)
print(yb)

for b in range(batch_size): #batch dimension
    for t in range(block_size): #time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target is {target}");

# %% [markdown]
# 

# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) #vocab size by embedding dimension

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) #batch x time x channels (vocab size), 4x8x64
        if targets is None:
            loss = None
        else: 
            B,T,C = logits.shape
            logits = logits.view(B*T, C) #concatenate into 2D for cross entropy function, B+T, C. Basically the batches just get squished.
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx in (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # apply softmax to get probabilities. we use -1 to only look at the last time step element. this represents the latest prediction.
            logits = logits[:, -1, :]
            # use latest logit score into probability distribution to get a class label from the distribution sample in next step.
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx;


# print(vocab_size)
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape) 
print(loss)


# test this out with a tensor of batch size 1 and time size 1. and then we produce 100 tokens based off that size one input.
idx = torch.zeros((1,1),dtype=torch.long)
# check dimension [0] because there is only one batch and the generate function is expecting many batches. 
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
#printed result is not good because we have an untrained model that is printing based off random weights.

# %%
#optimize learning rate. 1e-3 is a good starting point however we can get away with using a higher learning rate for a simple bigram model like the one we are currently training.
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

# %%
batch_size = 32
for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    #use gradients to update the parameters
    optimizer.step();
    print(loss.item())

# %%
#again test the dummy input from above after training. the result should be more comprehensible
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# %% [markdown]
# Self Attention code begins

# %%
# simple example not related to model
torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

#The 8 tokens should talk to each other. However the tokens should only talk with previous tokens (so that we don't look forward)

# %%
# we will take the average of all previous tokens as a first attempt at attention

#x bag of words - used for averaging
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] #(t, C)
        xbow[b,t] = torch.mean(xprev, 0) #takes the average of dim 0 and gives you a vector of size C

# %%
#version 2 averages with matrix mul

wei = torch.tril(torch.ones(T,T))
wei = wei/wei.sum(1,keepdim=True)
wei #weights, which take an average and are only backwards looking

xbow2 = wei @ x #(B,T,T) @ (B, T, C) ---> (B,T,C)

# we can see that this produces the same result as the manual averages that we calculated previously. matrix mul is more efficient here.
torch.allclose(xbow, xbow2)

# %%
# version 3 averages with softmax
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) # for all elements that are zero in tril, we make them negative infinity
wei = F.softmax(wei, dim=-1)
print(wei)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)


# %%
print(x[0])
print(xbow[0])

# %%
# we can speed this up using matrix mul. example below

torch.manual_seed(42)
# triangular ones
a = torch.tril(torch.ones(3,3))
# take average
a = a / torch.sum(a, 1, keepdim=True)
b=torch.randint(0,10,(3,2)).float()
c=a@b
print(a)
print(b)
print(c)

# %%



