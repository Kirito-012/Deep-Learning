import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # For mixed precision training

with open('input.txt', 'r') as f:
    text = f.read()

# ---------Constants---------
vocab_size = len(set(text))
batch_size = 64  # Increased for better GPU utilization
block_size = 128  # Longer context for better dependency capture
learning_rate = 3e-3
max_iters = 10001
eval_interval = 500
eval_iters = 200
n_embed = 128  # Larger embedding size for better representation
num_heads = 5  # More heads for richer attention
n_block_layer = 5  # Deeper model for increased capacity
dropout = 0.15 # Dropout rate for regularization
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------

# Converting words into tokens
stoi = {ch: i for i, ch in enumerate(sorted(list(set(text))))}
itos = {i: ch for i, ch in enumerate(sorted(list(set(text))))}

# Encoding and Decoding
encode = lambda x: [stoi[i] for i in x]
decode = lambda x: ''.join([itos[i] for i in x])

# Converting the whole data into tokens
data = torch.tensor(encode(text)).to(device)

# Train-test data split
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

# Getting a random set of data to train on
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+1+block_size] for i in idx])
    return x.to(device), y.to(device)

x, y = get_batch('train')
torch.manual_seed(42)

# Class to calculate attention values -> K, V, Q
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        query = self.query(x)
        key = self.key(x)
        
        weight = query @ key.transpose(-2, -1) * (C**-0.5)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)  # Apply dropout to attention weights
        
        value = self.value(x)
        output = weight @ value
        return output

# Class for multiple heads
class MultiHead(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)  # Dropout after projection

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(output)
        output = self.dropout(output)  # Apply dropout
        return output

# Feedforward network with dropout
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),  # GELU instead of ReLU for smoother gradients
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)  # Dropout for regularization
        )

    def forward(self, x):
        return self.net(x)

# Transformer block
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHead(head_size=head_size, num_heads=n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Language model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed=n_embed, n_head=num_heads) for _ in range(n_block_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            losses = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            losses = F.cross_entropy(logits, target)
        return logits, losses

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# Loss estimation function
@torch.no_grad()
def estimate_loss(model):
    output = {}
    model.eval()  # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):  # Specify device_type
                logits, loss = model(x, y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()  # Back to training mode
    return output

# Initialize model and training components
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler()  # For mixed precision training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)  # LR scheduler
print("Using device -->", device)

# Training loop with mixed precision and gradient clipping
for iter in range(max_iters):
    if (iter % eval_interval) == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Iter {iter}: Train Loss --> {losses['train']:.4f}, Val Loss --> {losses['val']:.4f}")
    
    x, y = get_batch('train')
    optimizer.zero_grad()
    
    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):  # Specify device_type
        logits, loss = model(x, y)
    
    scaler.scale(loss).backward()  # Scale gradients for mixed precision
    scaler.unscale_(optimizer)  # Unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    scaler.step(optimizer)  # Optimizer step with scaler
    scaler.update()  # Update scaler
    scheduler.step()  # Update learning rate

# Generation
fill = torch.zeros((1, 1), dtype=torch.long).to(device)
output = decode(model.generate(fill, 600)[0].tolist())
print("Generated output:\n", output)