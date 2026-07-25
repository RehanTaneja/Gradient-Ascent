import torch
import torch.nn as nn
from torch.nn import functional as F

# ----- Reading the Data -------

with open('brainrot_corpus.txt','r') as f:
    text = f.read()

chars = list(set(text))
vocab_size = len(chars)

# ------ Tokenization -------

char_to_int = {c:i for i,c in enumerate(chars)}
int_to_char = {i:c for i,c in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda i: ''.join(int_to_char[n] for n in i)

data = torch.tensor(encode(text),dtype=torch.long)
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

# ------- Hyperparameters --------

batch_size = 32
block_size = 8
max_iter = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
head_size = 16
torch.manual_seed(1337)

# -------- Batching ----------

def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:block_size+i] for i in ix])
    y = torch.stack([data[i+1:block_size+i+1] for i in ix])
    return x,y

# ------- Self Attention Head -------

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T][:T]==0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

# -------- Bigram ----------

class Bigram(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed) # this gives us token embeddings
        self.position_embedding_table = nn.Embedding(block_size,n_embed)
        self.sa_head = Head(n_embed) # Attention head
        self.lm_head = nn.Linear(n_embed,vocab_size) # passing token embedding here gives us logits

    def forward(self,idx,target=None):
        B,T = idx.shape
        tokn_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tokn_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
        return logits,loss

    def generate(self,idx,max_possible_token):
        for _ in range(max_possible_token):
            idx_cond = idx[:-block_size:] # crop idx to last block size token
            logits,loss = self(idx_cond)
            logits = logits[:-1:] # focus only on the last time step
            probs = F.softmax(logits,dim=1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

model = Bigram()
m = model.to(device)

# ------ Optimizer -------

optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate)

def train():
    for i in range(max_iter):
        xb,yb = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        logits,loss = m.forward(xb,yb)
        loss.backward()
        optimizer.step()
        print(loss.item())
    return loss.item()

loss = train()



