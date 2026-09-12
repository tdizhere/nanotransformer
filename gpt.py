import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kagglehub

# --- Configuration and Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)#u can use 67 too

# --- Data Loading and Tokenization ---
path = kagglehub.dataset_download("kingburrito666/shakespeare-plays")
with open(f"{path}/alllines.txt","r") as file:
  data = file.read()

char = sorted(list(set(data)))
vocab_size = len(char)
stoi = {ch:i for i,ch in enumerate(char) }
itos = {i:ch for i,ch in enumerate(char) }
encoder = lambda s:[stoi[c] for c in s]
decoder = lambda s:"".join([itos[c] for c in s])

# Convert data to tensor and split
input_tensor = torch.tensor(encoder(data), dtype=torch.long)
n = int(0.9 * len(input_tensor))
train_data = input_tensor[:n]
val_data = input_tensor[n:]

# --- Hyperparameters ---
batch_size = 128 # how many independent sequences will we process in parallel
block_size = 256 #how many tokens will we process in parallel
learning_rate = 1e-4 # tired 1e-5 , 3e-5 
val_iter = 300 # how many times to validate
max_iter = 5000 # how many times to train
n_embd = 384 # how many hidden units
n_head = 6 #number of heads 
n_layer = 6 #how many layer of blocks 
dropout = 0.4 #how much to drop out 
# --- Data Batching Function ---
def get_batch(split):
  data = train_data if split=="train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x,y
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(val_iter)
    for k in range(val_iter):
      X , Y = get_batch(split)
      logits , loss = model(X,Y)
      losses[k] = loss.item()
    out[split]=losses.mean()
  model.train()
  return out

# ---Head Definition ---
class head(nn.Module):
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embd,head_size,bias=False)#(b,t,16)
    self.Query = nn.Linear(n_embd,head_size,bias=False)#(b,t,16)
    self.Value = nn.Linear(n_embd,head_size,bias=False)#(b,t,16)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout =  nn.Dropout(dropout)
  def forward(self,x):
    B , T , C = x.shape
    k = self.key(x)
    q = self.Query(x)
    v = self.Value(x)
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #(b,t,16)@(b,16,t)==(b,t,t)
    wei = wei.masked_fill(self.tril[:T,:T] == 0 , float('-inf'))#masking future tokens
    wei = F.softmax(wei,dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    return out

# ---Multi Head Attention ---
class MultiHeadAttention(nn.Module):
  def __init__(self,n_head,head_size):
    super().__init__()
    self.heads = nn.ModuleList([head(head_size)for _ in range(n_head)])
    self.proj = nn.Linear(n_embd,n_embd) # n_emed = head_size * n_head
  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads],dim=-1)
    out = self.proj(out)
    return out 
# --- Feedforward ---
class FeedForward(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd,4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd,n_embd),
        nn.Dropout(dropout)
    )
  def forward(self,x):
    return self.net(x)
#--- block ---
class Block(nn.Module):
  def __init__(self,n_embd,n_head) -> None:
    super().__init__()
    head_size = n_embd // n_head
    self.sahead = MultiHeadAttention(n_head,head_size)
    self.ffn = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  def forward(self,x):
    x = x+ self.sahead(self.ln1(x))
    x = x+ self.ffn(self.ln2(x))
    return x    
# --- Model Definition ---
class gpt(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.positional_embeding_table = nn.Embedding(block_size,n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd,n_head)for _ in range(n_layer)])
    self.lm_head = nn.Linear(n_embd,vocab_size)
    self.ln1 = nn.LayerNorm(n_embd)
   
  def forward(self,idx,target=None):
    B , T = idx.shape
    token_embd = self.token_embedding_table(idx) # (B, T, C)
    pos_embd = self.positional_embeding_table(torch.arange(T,device=device))#(T,C)
    x = token_embd + pos_embd
    x = self.blocks(x)
    x = self.ln1(x)
    logits = self.lm_head(x)#(B,T,vocab_size)

    if target is None:
      loss = None
    else:
      B, T, C = logits.shape
      # reshape logits (B*T, C)
      logits_reshaped = logits.view(B*T, C)
      # reshape target (B*T)
      target_reshaped = target.view(B*T)
      loss = F.cross_entropy(logits_reshaped, target_reshaped)

    return logits, loss # (B, T, C)

  def generate(self,idx,max_new_token):
    # idx is (B, T) array of indices in the current context
    for i in range(max_new_token):
      # Crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits , loss = self(idx_cond) # Only need logits for generation
      # focus only on the last time step
      logits = logits[:,-1, :] #(B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits,dim=-1)#(B,C)
      # sample from the distribution
      idx_next = torch.multinomial(probs,num_samples=1)#(B,1)
      # append sampled index to the running sequence
      idx = torch.cat((idx,idx_next),dim=1)#(B,T+1)
    return idx

# -Model Instantiation and Training-
model = gpt(vocab_size)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for steps in range(max_iter):
  if steps%500==0:
    losses = estimate_loss()
    print(f"Step {steps}: train loss = {losses['train']} ,val loss  = {losses['val']}")
  xb, yb = get_batch('train')
  logits , loss = model(xb,yb) 
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

#generation after training
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(model.generate(context, max_new_token=500)[0].tolist())) 