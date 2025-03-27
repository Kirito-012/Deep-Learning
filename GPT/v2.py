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
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


print(f"Using device: {device}")

# Fetching the data
with (open("input.txt", "r")) as file:
    text = file.read()

# Preprocessing the data
vocab_size = len(set(text))
string_to_integer = {char: i for i, char in enumerate(set(text))}
integer_to_string = {i: char for i, char in enumerate(set(text))}
encode = lambda x: [string_to_integer[char] for char in x]
decode = lambda x: "".join([integer_to_string[i] for i in x])
data = torch.tensor(encode(text))

# train-test data split
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
train_data = train_data.to(device)
val_data = val_data.to(device)

# function to get a batch
torch.manual_seed(1337)
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

# Loss Estimator - Evaluating the loss on the train and validation set
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

        # this is efficent and saves computation because if you do this in the forward method, 
        # you're recreating the same lower-tringular mask from scratch every forward pass
        # that's why you compute it once in __init__ and 
        # store it as buffer (model's memory) and keep using the already-made-tril in the forward pass

        # the reason this works is because the block_size is a hyperparameter, 
        # which basically defines the --> maximum sequence length the model supports.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2,-1) / (C ** 0.5)

        # T here is basically the size, width of the Attention score
#       self.tril[:4, :4] = [
#                               [1, 0, 0, 0],
#                               [1, 1, 0, 0],
#                               [1, 1, 1, 0],
#                               [1, 1, 1, 1]
#                           ]
#       self.tril[:4, :4] == 0 = [
#                                   [False, True,  True,  True ],
#                                   [False, False, True,  True ],
#                                   [False, False, False, True ],
#                                   [False, False, False, False]
#                                ]

#       After Masking: Only past and present are heard
#             [I]   [am]  [good]  [.]
#       [I]    ✓    mute   mute   mute
#       [am]   ✓     ✓     mute   mute
#       [good] ✓     ✓      ✓     mute
#       [.]    ✓     ✓      ✓      ✓

        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        v = self.value(x)   
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # num_heads basically means for parallel processing, that is different number of Q,V,K values for each head
        # one might focuses on context
        # second might focus on semantics
        # third might focus on broader context etc
        # basically num_of_heads you want in parallel processing

        # nn.ModuleList --> a PyTorch container that registers each Head as a submodule, 
        # their parameters are tracked for optimization and moved to the correct device

        # creates a list of num_heads(4) instances of the Head
        # head_size --> dimensionality of the output of each head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # a linear projection layer that combines the outputs of all heads (all 4 heads * output for each head) 
        # back into the original embedding dimension (n_embed)
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # passing the input x through each of the num_heads attention heads in self.heads --> [head(x) for head in self.heads]
        # concatenates the output of all heads along the last dimension --> torch.cat([...], dim=-1)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # OUTPUT --> single tensor out that combines the results of all heads, 
        # where each token’s representation is now num_heads * head_size dimensions wide

        # Passes the concatenated output through the linear projection layer --> self.proj(out)
        # Shape - (B,T, n_embed)
        out = self.dropout(self.proj(out))
        return out

# it expands the dimensionality, applies a nonlinearity, shrinks it back, and adds dropout for regularization
# adds capacity to the model, allowing it to learn more complex patterns beyond what attention alone can capture.
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # takes a input of size n_embed(4) and outupts a size of 4*n_embed(16)
            nn.Linear(n_embed, 4 * n_embed),
            # setting all -ves to 0, introducing non-linearity
            nn.ReLU(),
            # taking the input of 16 and giving the output of 4
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

# combines multi-head self-attention and a feed-forward neural network 
# with residual connections and layer normalization
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        # computes the size of each attention head (384//6 = 64)
        # that is, each head processes 64 no. of embeddings
        head_size = n_embed // n_head
        # initialising multiHeadAttention with n_head = 6 and head_size = 64
        # basically create 6 Head (class) with 64 embeddings (head_size)
        self.sa = MultiHeadAttention(n_head, head_size)
        # expands the dimensionality --> 384 → 1536 → 384
        self.ffwd = FeedForward(n_embed)
        # layer normalization for input to attention mechanism
        self.ln1 = nn.LayerNorm(n_embed)
        # layer normalization for input to feed-forward network
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residual connection --> x + output_multi_head_values(attention scores)
        x = x + self.sa(self.ln1(x))
        # residual connection --> x + output_feed_forward
        x = x + self.ffwd(self.ln2(x))
        return x

# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # first layer which maps each token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # 65, 32

        # that maps positional indices to vectors of size n_embed
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])

        # that normalizes across the n_embed dimension
        self.ln_f = nn.LayerNorm(n_embed)

        # linear layer that maps the n_embed hidden state backt to the vocab_size dimension to predict possibilities
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape    
        
        token_emb = self.token_embedding_table(idx) # Shape = (Batch_size, Seq_Len, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Shape = (T, C)
        x = token_emb + pos_emb
        x = self.blocks(x) # Shape = (B, T, C) 
        x = self.ln_f(x)
        logits = self.lm_head(x) # Shape = ( B, T, vocab_size)

        if targets is None:
            loss = None       
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:,-1:, :]
            probs = F.softmax(logits, dim=-1).squeeze(1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr = lr)


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Training Step
for iter in range(max_iters):
    if  iter % 500 == 0:
        losses = estimateloss()
        print(f"iter: {iter}, train loss: {losses['train']}, val loss: {losses['val']}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))

# Might not be completely GPU-favourable