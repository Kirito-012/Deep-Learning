import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 64
block_size = 256 
max_iters = 5001
eval_interval = 500
lr = 3e-4 
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384  # Consistent value
n_head = 6
n_layer = 6
dropout = 0.2

print(f"Using device: {device}")

# Fetching the data
with open("input.txt", "r") as file:
    text = file.read()

# Preprocessing the data
vocab_size = len(set(text))
string_to_integer = {char: i for i, char in enumerate(set(text))}
integer_to_string = {i: char for i, char in enumerate(set(text))}
encode = lambda x: [string_to_integer[char] for char in x]
decode = lambda x: "".join([integer_to_string[i] for i in x])
data = torch.tensor(encode(text), dtype=torch.long)

# Train-test data split
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Function to get a batch
torch.manual_seed(1337)
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

# Loss Estimator
@torch.no_grad()
def estimateloss():
    out = {}
    model.eval()    
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2,-1) / (C ** 0.5)
        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        v = self.value(x)   
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape    
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(min(T, block_size), device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None       
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1:, :]
            probs = F.softmax(logits, dim=-1).squeeze(1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.Adam(m.parameters(), lr=lr)

# Training Loop with Gradient Clipping
for iter in range(max_iters):
    if iter % 500 == 0:
        losses = estimateloss()
        print(f"iter: {iter}, train loss: {losses['train']}, val loss: {losses['val']}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))