import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 32
block_size = 8 
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
max_iters = 3000
eval_interval = 300
eval_iters = 200

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


# function to get a batch
torch.manual_seed(1337)
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x,y = x.to(device), y.to(device)
    return x, y

# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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
            logits, _ = self.forward(idx)
            logits = logits[:,-1:, :]
            probs = F.softmax(logits, dim=-1).squeeze(1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

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

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.Adam(m.parameters(), lr = lr)

# Training Step
for iter in range(max_iters):
    if  iter % eval_interval == 0:
        losses = estimateloss()
        print(f"iter: {iter}, train loss: {losses['train']}, val loss: {losses['val']}")
    
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))