import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example config
@dataclass
class Config:
    embedding_dims: int = 1024
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 12
    ffn_dims: int = 4096
    vocab_size: int = 16000
    block_size: int = 1024
    eps: float = 1e-6
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    resid_dropout: float = 0.1
    init_std: float = 0.02
    device: str = device

    def __post_init__(self):
        assert self.embedding_dims % self.n_heads == 0, \
            "embedding_dims must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"

config = Config()

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.head_dim = config.embedding_dims // config.n_heads
        # projections
        self.q_proj = nn.Linear(config.embedding_dims, config.embedding_dims, bias=False)
        self.k_proj = nn.Linear(config.embedding_dims, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.embedding_dims, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.embedding_dims, config.embedding_dims, bias=False)
        # dropout
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        # rotary frequencies buffer
        dim_half = self.head_dim // 2
        inv_freq = 1.0 / (10000 ** (2 * torch.arange(dim_half, device=config.device, dtype=torch.float32) / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # LLaMA uses normal(0, init_std)
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def apply_rope(self, tensor, n_heads):
        bsz, seq_len, _ = tensor.size()
        tensor = tensor.view(bsz, seq_len, n_heads, self.head_dim).transpose(1, 2)
        dim_half = self.head_dim // 2
        pos = torch.arange(seq_len, device=tensor.device).unsqueeze(-1)
        theta = pos * self.inv_freq[None, None, :]
        sin, cos = theta.sin(), theta.cos()
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)
        x1, x2 = tensor[..., :dim_half], tensor[..., dim_half:]
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        rotated = torch.cat([x_rot1, x_rot2], dim=-1)
        return rotated.transpose(1, 2).reshape(bsz, seq_len, n_heads * self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # RoPE
        q = self.apply_rope(q, self.config.n_heads)
        k = self.apply_rope(k, self.config.n_kv_heads)
        # reshape
        q = q.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        # expand KV for grouped Q/K
        k = k.repeat_interleave(self.config.n_heads // self.config.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.config.n_heads // self.config.n_kv_heads, dim=1)
        # attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.config.attn_dropout
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.o_proj(attn_out))

class LlamaMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.embedding_dims, config.ffn_dims)
        self.up_proj   = nn.Linear(config.embedding_dims, config.ffn_dims)
        self.down_proj = nn.Linear(config.ffn_dims, config.embedding_dims)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(config.ffn_dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        u = self.silu(self.gate_proj(x))
        v = self.up_proj(x)
        x = u * v
        x = self.dropout(x)
        return self.down_proj(x)

class LlamaRMSNorm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.embedding_dims))
        self.eps = config.eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

class LlamaBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn_norm = LlamaRMSNorm(config)
        self.attn = MultiHeadAttention(config)
        self.mlp_norm = LlamaRMSNorm(config)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class LlamaModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dims)
        self.blocks = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.fnorm = LlamaRMSNorm(config)
        self.lm_head = nn.Linear(config.embedding_dims, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.fnorm(x)
        return self.lm_head(x)

class LlamaLMHeadModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.model = LlamaModel(config)

    def forward(self, input_ids, labels=None):
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        return (loss, logits) if loss is not None else logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device in usage: {device}')
    
    input_ids = torch.randint(0, 16000, (32, 128), device=device)
    input_ids = input_ids.to(device)
    llama = LlamaLMHeadModel(config)
    llama = llama.to(device)
    loss, logits = llama(input_ids, input_ids.clone())
    print(f"Loss: {loss.item():.4f}")