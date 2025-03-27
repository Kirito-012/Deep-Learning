import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

with open('input.txt', 'r') as f:
    text = f.read()

# ---------Constants---------
vocab_size = len(set(text))
batch_size = 32
block_size = 128
learning_rate = 0.01
max_iters = 10001
eval_interval = 500
eval_iters = 200
n_embed = 64
num_heads = 6
n_block_layer = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------

# Converting words into tokens
stoi = {ch:i for i, ch in enumerate(sorted(list(set(text))))}
itos = {i:ch for i, ch in enumerate(sorted(list(set(text))))}

# Encoding and Decoding
encode = lambda x : [stoi[i] for i in x]
decode = lambda x : ''.join([itos[i] for i in x])

# Converting the whole data into tokens
data = torch.tensor(encode(text)).to(device)

# train-test data split
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
train_data = train_data.to(device)
val_data = val_data.to(device)

# getting a random set of data to train on
def get_batch(split):
    data = train_data if split=='train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+1+block_size] for i in idx])
    return x, y

x,y = get_batch('train')
torch.manual_seed(42)

# class to caculate attention values -> K, V, Q
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B,T,C = x.shape
        query = self.query(x)
        key = self.key(x)
        
        weight = query @ key.transpose(-2,-1) * (C**-0.5)
        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)

        value = self.value(x)
        output = weight @ value
        return output

# class for multiple heads -> stacking heads over heads for even better optimization 
class MultiHead(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads*head_size, n_embed)
    
    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(output)
        return output

# expanding and shrinking the dimension for better grasping
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed)
        )
    
    def forward(self, x):
        return self.net(x)

# creating a block which would first normalize -> multihead self-attention + normalize -> feed forwar
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.sa = MultiHead(head_size=head_size, num_heads=n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# making the model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed=n_embed, n_head=num_heads) for _ in range(n_block_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)        

    def forward(self, idx, target=None):
        B,T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            losses = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            losses = F.cross_entropy(logits, target)
        return logits, losses

    def generate(self, idx, max_tokens):
        for k in range(max_tokens):
            # to avoid generating a larger output than the block size
            idx_cond = idx[:, -block_size:]
            logits, loss = self.forward(idx_cond)
            logits = logits[:,-1,:]
            soft = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(soft, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# defining a function to calculate loss
@torch.no_grad()
def estimateLoss():
    output = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    return output

# initiating the model
model = BigramLanguageModel(vocab_size).to(device)
logits, loss = model(x,y)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
print("Using device -->", device)

for iter in range(max_iters):
    if (iter % eval_interval) == 0:
        losses = estimateLoss()
        print(f"Iter {iter}: Train Loss --> {losses['train']}, Val Loss --> {losses['val']}")
    x,y = get_batch('train')
    logits, loss = model(x,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

fill = torch.zeros((1,1), dtype=torch.long).to(device)
output = decode(model.generate(fill, 600)[0].tolist())
print(output)

# Make so that the model is saved and generate the output in a file