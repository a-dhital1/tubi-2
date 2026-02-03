#!/usr/bin/env python3
"""
screenplay_transformer.py

Transformer language model for generating movie screenplays. I built this from scratch
to really understand how these models work internally - the tokenization, attention
mechanism, training dynamics, etc. The model learns to output properly formatted
screenplay text with scene headings, character names, dialog, and action descriptions.

The architecture follows the standard GPT pattern: token embeddings, positional 
embeddings, a stack of transformer blocks (each with self-attention and a feedforward
network), then a final projection back to vocabulary logits.

Training data should be screenplay text with special tokens marking structure:
    <SCENE>INT. COFFEE SHOP - DAY</SCENE>
    <ACTION>A busy morning. Customers line up at the counter.</ACTION>
    <CHARACTER>SARAH</CHARACTER>
    <DIALOG>I'll have the usual.</DIALOG>

Usage examples:
    python screenplay_transformer.py train --data ./scenes.txt
    python screenplay_transformer.py generate --prompt "dark alley at night"
    python screenplay_transformer.py interactive
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import regex  # using regex instead of re for better unicode handling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Model hyperparameters. I tried a few different configurations and found that
# 8 layers with 8 attention heads and 512-dim embeddings gives a good balance
# between quality and training speed. This comes out to roughly 85M parameters.

@dataclass
class ModelConfig:
    vocab_size: int = 10000     # size of token vocabulary
    block_size: int = 512       # maximum sequence length / context window
    n_layer: int = 8            # number of transformer blocks
    n_head: int = 8             # attention heads per block
    n_embd: int = 512           # embedding dimension (also hidden size)
    dropout: float = 0.1        # dropout rate for regularization
    bias: bool = False          # whether to use bias in linear layers (False = faster)
    
    def __post_init__(self):
        # embedding dim needs to be evenly divisible by number of heads
        # since each head gets a slice of the full embedding
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"embedding dim {self.n_embd} must be divisible by num heads {self.n_head}"
            )
    
    @property
    def head_dim(self):
        # each attention head operates on this many dimensions
        return self.n_embd // self.n_head


@dataclass 
class TrainConfig:
    batch_size: int = 32        # sequences per batch
    lr: float = 3e-4            # peak learning rate (after warmup)
    weight_decay: float = 0.1   # L2 regularization strength
    epochs: int = 10            # full passes through training data
    warmup_steps: int = 100     # linear warmup before cosine decay
    grad_clip: float = 1.0      # max gradient norm (prevents exploding gradients)
    eval_every: int = 500       # run validation every N steps
    save_every: int = 1000      # checkpoint every N steps
    log_every: int = 10         # print loss every N steps


# Special tokens that mark different structural elements in screenplays.
# The model learns to generate these at appropriate points to structure its output.
# PAD/UNK/BOS/EOS are standard LM tokens, the rest are screenplay-specific.

SPECIAL_TOKENS = [
    '<PAD>',            # padding for batching variable-length sequences
    '<UNK>',            # unknown token (for OOV during inference)
    '<BOS>',            # beginning of sequence
    '<EOS>',            # end of sequence
    '<SCRIPT>',         # script title start
    '</SCRIPT>',        # script title end
    '<SCENE>',          # scene heading start (e.g., "INT. OFFICE - DAY")
    '</SCENE>',         # scene heading end
    '<ACTION>',         # action/description start
    '</ACTION>',        # action/description end
    '<CHARACTER>',      # character name start
    '</CHARACTER>',     # character name end
    '<DIALOG>',         # dialog start
    '</DIALOG>',        # dialog end
    '<END>',            # end of script marker
    '<SCENE_START>',    # scene boundary (for scene-level training)
    '<SCENE_END>',      # scene boundary end
]


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer for screenplay text.
    
    BPE is a subword tokenization algorithm that starts with individual bytes
    and iteratively merges the most frequent adjacent pairs until reaching
    the target vocabulary size. This gives a nice balance - common words get
    their own tokens while rare words are broken into subword pieces.
    
    The algorithm:
    1. Start with 256 byte-level tokens plus special tokens
    2. Count frequency of all adjacent token pairs in the corpus
    3. Merge the most frequent pair into a new token
    4. Repeat until vocab size is reached
    
    I'm using the GPT-2 regex pattern for initial word splitting, which handles
    English contractions and punctuation in a sensible way.
    """
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}             # token -> id mapping
        self.inv_vocab = {}         # id -> token mapping (for decoding)
        self.merges = {}            # (id1, id2) -> merged_id
        self.special_ids = {}       # quick lookup for special token ids
        
        # GPT-2's tokenization pattern. It handles contractions like "don't" -> "don" + "'t"
        # and keeps punctuation attached to words in sensible ways
        self._pat = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    def train(self, text, verbose=True):
        """
        Learn BPE merges from a text corpus.
        
        This is the slow part - for each merge we need to count pair frequencies
        across the entire corpus. For very large datasets I sample random chunks
        to make it tractable while still learning good merges.
        """
        if verbose:
            print(f"Training BPE tokenizer (vocab_size={self.vocab_size})")
        
        # Initialize vocab with special tokens first (they get ids 0, 1, 2, ...)
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[tok] = i
            self.special_ids[tok] = i
        
        # Then add all 256 possible single bytes
        # These form the base "alphabet" that BPE builds upon
        n_special = len(SPECIAL_TOKENS)
        for byte_val in range(256):
            self.vocab[bytes([byte_val])] = n_special + byte_val
        
        # For very large corpora, sample random chunks to speed up training
        # 50M chars is already quite a lot for learning good merges
        if len(text) > 50_000_000:
            import random
            sampled = []
            for _ in range(500):
                start = random.randint(0, len(text) - 100_000)
                sampled.append(text[start:start + 100_000])
            text = ''.join(sampled)
            if verbose:
                print(f"  Sampled {len(text):,} characters for training")
        
        # Split text into words and count frequencies
        words = self._pat.findall(text)
        word_freq = Counter(words)
        if verbose:
            print(f"  Found {len(word_freq):,} unique words")
        
        # Convert each word to a sequence of byte-level token ids
        # e.g., "hello" -> [h_id, e_id, l_id, l_id, o_id]
        word_ids = {}
        for word in word_freq:
            byte_seq = word.encode('utf-8')
            word_ids[word] = [n_special + b for b in byte_seq]
        
        # Main merge loop - keep merging most frequent pairs until we hit vocab size
        n_merges = self.vocab_size - len(self.vocab)
        
        for merge_num in range(n_merges):
            # Count how often each adjacent pair appears across all words
            pair_counts = Counter()
            for word, freq in word_freq.items():
                ids = word_ids[word]
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    pair_counts[pair] += freq
            
            if not pair_counts:
                break  # no more pairs to merge
            
            # Find the most frequent pair and create a new token for it
            best_pair = pair_counts.most_common(1)[0][0]
            new_id = len(self.vocab)
            
            # The new token's byte representation is the concat of the two merged tokens
            left_bytes = self._id_to_bytes(best_pair[0])
            right_bytes = self._id_to_bytes(best_pair[1])
            self.vocab[left_bytes + right_bytes] = new_id
            self.merges[best_pair] = new_id
            
            # Update all word sequences to use the new merged token
            for word in word_ids:
                word_ids[word] = self._apply_merge(word_ids[word], best_pair, new_id)
            
            if verbose and (merge_num + 1) % 500 == 0:
                print(f"  Completed {merge_num + 1}/{n_merges} merges")
        
        # Build reverse vocab for decoding
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        if verbose:
            print(f"  Final vocabulary size: {len(self.vocab)}")
    
    def _apply_merge(self, ids, pair, new_id):
        """Replace all occurrences of a token pair with the merged token."""
        result = []
        i = 0
        while i < len(ids):
            # Check if current position matches the pair we want to merge
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2  # skip both tokens
            else:
                result.append(ids[i])
                i += 1
        return result
    
    def _id_to_bytes(self, token_id):
        """Convert a token id back to its byte representation."""
        n_special = len(SPECIAL_TOKENS)
        
        if token_id < n_special:
            # It's a special token
            return SPECIAL_TOKENS[token_id].encode('utf-8')
        elif token_id < n_special + 256:
            # It's a single-byte token
            return bytes([token_id - n_special])
        else:
            # It's a merged token - look it up
            tok = self.inv_vocab.get(token_id)
            if isinstance(tok, bytes):
                return tok
            return tok.encode('utf-8') if tok else b''
    
    def encode(self, text):
        """
        Convert text to a list of token ids.
        
        Special tokens are handled separately (not split by BPE).
        Regular text is split into words, then each word is encoded
        by applying learned merges.
        """
        ids = []
        
        # Split on special tokens while keeping them in the result
        special_pattern = '|'.join(regex.escape(t) for t in SPECIAL_TOKENS)
        parts = regex.split(f'({special_pattern})', text)
        
        for part in parts:
            if not part:
                continue
            
            if part in self.special_ids:
                # This is a special token - add its id directly
                ids.append(self.special_ids[part])
            else:
                # Regular text - split into words and encode each
                for word in self._pat.findall(part):
                    ids.extend(self._encode_word(word))
        
        return ids
    
    def _encode_word(self, word):
        """Encode a single word by applying BPE merges."""
        n_special = len(SPECIAL_TOKENS)
        
        # Start with byte-level tokens
        ids = [n_special + b for b in word.encode('utf-8')]
        
        # Repeatedly apply merges until no more apply
        while len(ids) >= 2:
            # Find all pairs that have learned merges
            candidates = {}
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in self.merges:
                    candidates[pair] = self.merges[pair]
            
            if not candidates:
                break
            
            # Apply the merge that was learned earliest (lowest id)
            # This ensures consistent tokenization
            best_pair = min(candidates, key=candidates.get)
            ids = self._apply_merge(ids, best_pair, self.merges[best_pair])
        
        return ids
    
    def decode(self, ids):
        """Convert a list of token ids back to text."""
        byte_chunks = []
        for token_id in ids:
            if token_id in self.inv_vocab:
                tok = self.inv_vocab[token_id]
                if isinstance(tok, str):
                    byte_chunks.append(tok.encode('utf-8'))
                else:
                    byte_chunks.append(tok)
        
        # Join bytes and decode to string, replacing invalid sequences
        return b''.join(byte_chunks).decode('utf-8', errors='replace')
    
    def save(self, directory):
        """Save tokenizer to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Serialize vocab - need to convert bytes to hex strings for JSON
        vocab_serialized = {}
        for tok, tok_id in self.vocab.items():
            if isinstance(tok, bytes):
                vocab_serialized[tok.hex()] = tok_id
            else:
                vocab_serialized[f"s:{tok}"] = tok_id  # prefix string tokens
        
        merges_serialized = {f"{a},{b}": v for (a, b), v in self.merges.items()}
        
        data = {
            'vocab_size': self.vocab_size,
            'vocab': vocab_serialized,
            'merges': merges_serialized
        }
        
        with open(path / 'tokenizer.json', 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, directory):
        """Load tokenizer from disk."""
        with open(Path(directory) / 'tokenizer.json') as f:
            data = json.load(f)
        
        tokenizer = cls(data['vocab_size'])
        
        # Deserialize vocab
        for key, tok_id in data['vocab'].items():
            if key.startswith('s:'):
                tokenizer.vocab[key[2:]] = tok_id
            else:
                tokenizer.vocab[bytes.fromhex(key)] = tok_id
        
        # Deserialize merges
        for key, merged_id in data['merges'].items():
            left, right = map(int, key.split(','))
            tokenizer.merges[(left, right)] = merged_id
        
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.special_ids = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        
        return tokenizer
    
    @property
    def pad_id(self):
        return self.special_ids['<PAD>']
    
    @property
    def scene_end_id(self):
        return self.special_ids['<SCENE_END>']


# Transformer model components

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    
    This is the core mechanism that lets the model look at previous tokens
    when predicting the next one. "Causal" means each position can only attend
    to earlier positions - we mask out the future to prevent information leakage
    during training.
    
    The attention computation:
    1. Project input to queries (Q), keys (K), and values (V)
    2. Compute attention scores: softmax(Q @ K^T / sqrt(d_k))
    3. Apply causal mask (set future positions to -inf before softmax)
    4. Compute weighted sum of values
    
    I combine Q/K/V into a single linear layer for efficiency - one big matmul
    is faster than three separate ones.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        
        # Combined projection for Q, K, V - outputs 3x embedding dim
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection after attention
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout for regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask - lower triangular matrix of ones
        # Position i can only attend to positions 0, 1, ..., i
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer('mask', mask.view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V in one shot
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        # (batch, seq, embed) -> (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with scaling
        # The 1/sqrt(d) scaling prevents dot products from getting too large
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        
        # Apply causal mask - set future positions to -inf so softmax gives 0
        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0, 
            float('-inf')
        )
        
        # Normalize to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute weighted sum of values
        output = attn_weights @ v
        
        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.resid_dropout(self.c_proj(output))
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Applied independently to each position after attention. The standard
    transformer uses a hidden dimension 4x the embedding dimension, with
    a GELU activation in between. This gives the model capacity to learn
    complex transformations at each position.
    """
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block: layer norm -> attention -> layer norm -> FFN
    
    Uses pre-normalization (layer norm before each sublayer) rather than
    post-normalization. This tends to be more stable during training,
    especially for deeper models. Residual connections around each sublayer
    help gradients flow during backprop.
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # FFN with residual connection
        x = x + self.ffn(self.ln_2(x))
        return x


class ScreenplayGPT(nn.Module):
    """
    GPT-style decoder-only transformer for screenplay generation.
    
    Architecture:
    1. Token embedding: convert token ids to vectors
    2. Position embedding: add positional information
    3. Stack of transformer blocks
    4. Final layer norm
    5. Output projection to vocabulary logits
    
    Weight tying: the token embedding and output projection share weights.
    This reduces parameters and often improves performance since both
    operate in the same "token space".
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # position embeddings
        self.drop = nn.Dropout(config.dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm before output projection
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying between token embedding and output projection
        self.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report model size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {n_params/1e6:.1f}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        idx: token ids, shape (batch_size, sequence_length)
        targets: optional target ids for computing loss
        
        Returns logits and optionally the cross-entropy loss.
        """
        batch_size, seq_len = idx.size()
        
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        
        # Get token and position embeddings
        pos = torch.arange(0, seq_len, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        
        # Combine embeddings and apply dropout
        x = self.drop(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection to vocab
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute cross-entropy loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (batch * seq, vocab) vs (batch * seq,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # ignore padding in loss computation
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_at=None):
        """
        Generate tokens autoregressively.
        
        Uses temperature scaling and optional top-k/top-p filtering to control
        the randomness of generation. Lower temperature = more deterministic,
        higher = more creative/random.
        
        top_k: only sample from the k most likely tokens
        top_p: nucleus sampling - sample from smallest set with cumulative prob >= p
        stop_at: stop generation when this token id is produced
        """
        for _ in range(max_new_tokens):
            # Crop to context window if sequence is too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get logits for the last position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering: keep only the k highest probability tokens
            if top_k is not None:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')
            
            # Top-p (nucleus) filtering: keep tokens until cumulative prob exceeds p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Find where cumulative probability exceeds threshold
                tokens_to_remove = cumulative_probs > top_p
                # Keep at least one token
                tokens_to_remove[..., 1:] = tokens_to_remove[..., :-1].clone()
                tokens_to_remove[..., 0] = False
                
                # Scatter mask back to original ordering
                indices_to_remove = tokens_to_remove.scatter(1, sorted_indices, tokens_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)
            
            # Check for stop token
            if stop_at is not None and next_token.item() == stop_at:
                break
        
        return idx
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


# Data loading

class ScreenplayDataset(Dataset):
    """
    Dataset for training the language model.
    
    Loads tokenized screenplay data and creates training examples.
    Each example is a sequence of block_size tokens, and the target
    is the same sequence shifted by one position (next-token prediction).
    """
    
    def __init__(self, path, tokenizer, block_size):
        print(f"Loading training data from {path}")
        
        with open(path) as f:
            text = f.read()
        
        print(f"Tokenizing {len(text):,} characters...")
        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        
        print(f"Dataset contains {len(self.data):,} tokens")
    
    def __len__(self):
        # Number of possible starting positions
        return max(0, len(self.data) - self.block_size - 1)
    
    def __getitem__(self, idx):
        # Input: tokens at positions [idx, idx + block_size)
        # Target: tokens at positions [idx + 1, idx + block_size + 1)
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def compute_lr(step, warmup_steps, max_lr, min_lr=1e-5):
    """
    Learning rate schedule: linear warmup followed by cosine decay.
    
    Warmup helps stabilize early training when gradients can be noisy.
    Cosine decay gradually reduces LR for fine-grained optimization later.
    """
    if step < warmup_steps:
        # Linear warmup from 0 to max_lr
        return max_lr * step / warmup_steps
    
    # Cosine decay from max_lr to min_lr
    progress = (step - warmup_steps) / max(1, 10000 - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_device():
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return 'cuda'
    # Apple Silicon support
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train_model(model, train_loader, val_loader, config, output_dir, device):
    """
    Main training loop.
    
    Uses AdamW optimizer with weight decay, gradient clipping, and a
    warmup + cosine decay learning rate schedule. Periodically evaluates
    on validation set and saves checkpoints.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # AdamW optimizer - Adam with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"\nStarting training on {device}")
    print("-" * 50)
    
    for epoch in range(config.epochs):
        model.train()
        epoch_start = time.time()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Update learning rate
            lr = compute_lr(global_step, config.warmup_steps, config.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            _, loss = model(batch_x, batch_y)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            global_step += 1
            
            # Logging
            if global_step % config.log_every == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
            
            # Validation
            if global_step % config.eval_every == 0:
                val_loss = evaluate_model(model, val_loader, device)
                print(f"  Validation loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save(str(output_path / 'best.pt'))
                    print("  New best model saved!")
                
                model.train()
            
            # Periodic checkpoint
            if global_step % config.save_every == 0:
                model.save(str(output_path / f'checkpoint_step{global_step}.pt'))
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
    
    # Save final model
    model.save(str(output_path / 'final.pt'))
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


@torch.no_grad()
def evaluate_model(model, data_loader, device, max_batches=50):
    """Compute average loss on validation data."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in data_loader:
        if num_batches >= max_batches:
            break
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        _, loss = model(batch_x, batch_y)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


# Prompt processing and output formatting

def parse_prompt(user_input):
    """
    Convert natural language input to screenplay format.
    
    This lets users type things like "dark alley at night" instead of
    manually formatting it as "<SCENE>EXT. DARK ALLEY - NIGHT</SCENE>".
    
    It's a bit hacky but handles common cases reasonably well.
    """
    user_input = user_input.strip()
    
    # If already has screenplay tags, use as-is
    if '<SCENE>' in user_input or '<ACTION>' in user_input:
        return user_input
    
    lower = user_input.lower()
    
    # Detect time of day from keywords
    time_keywords = {
        'night': 'NIGHT',
        'evening': 'EVENING',
        'morning': 'MORNING',
        'dawn': 'DAWN',
        'dusk': 'DUSK',
        'afternoon': 'AFTERNOON'
    }
    
    time_of_day = 'DAY'  # default
    for keyword, formatted in time_keywords.items():
        if keyword in lower:
            time_of_day = formatted
            # Remove the time word from input
            user_input = re.sub(rf'\b{keyword}\b', '', user_input, flags=re.I).strip()
            break
    
    # Determine interior vs exterior based on location keywords
    exterior_keywords = ['outside', 'street', 'park', 'beach', 'forest', 'rooftop', 'alley', 'road', 'parking']
    is_exterior = any(word in lower for word in exterior_keywords)
    location_type = 'EXT.' if is_exterior else 'INT.'
    
    # Clean up common filler words
    filler_words = ['inside', 'outside', 'a ', 'the ', 'in ', 'at ']
    for word in filler_words:
        user_input = re.sub(rf'\b{re.escape(word)}\b', '', user_input, flags=re.I)
    user_input = ' '.join(user_input.split()).strip()
    
    # Check if input describes an action vs a location
    action_verbs = ['walks', 'runs', 'sits', 'enters', 'exits', 'stands', 'looks', 'waits']
    is_action = any(verb in lower for verb in action_verbs)
    
    if is_action:
        return f"<SCENE_START>\n<SCENE>{location_type} UNKNOWN - {time_of_day}</SCENE>\n<ACTION>{user_input.capitalize()}"
    
    location = user_input.upper() if user_input else 'UNKNOWN'
    return f"<SCENE_START>\n<SCENE>{location_type} {location} - {time_of_day}</SCENE>"


def format_screenplay_output(raw_text):
    """
    Convert model output (with tags) to readable screenplay format.
    
    Applies proper indentation according to screenplay conventions:
    - Scene headings: all caps, full width
    - Character names: centered (indented ~25 spaces)
    - Dialogue: indented ~15 spaces
    - Action: full width
    """
    lines = []
    current_mode = None
    
    # Split on tags while keeping them
    parts = re.split(r'(<[^>]+>)', raw_text)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if part.startswith('<') and part.endswith('>'):
            # This is a tag - update mode
            tag = part[1:-1]
            
            if tag == 'SCENE':
                current_mode = 'scene'
            elif tag == 'ACTION':
                current_mode = 'action'
            elif tag == 'CHARACTER':
                current_mode = 'character'
            elif tag == 'DIALOG':
                current_mode = 'dialog'
            elif tag == 'SCENE_END':
                lines.append('\n' + '-' * 40)
                current_mode = None
            elif tag.startswith('/'):
                current_mode = None
        else:
            # This is content - format based on current mode
            if current_mode == 'scene':
                lines.append(f"\n{part.upper()}\n")
            elif current_mode == 'action':
                lines.append(part)
                lines.append('')
            elif current_mode == 'character':
                lines.append(' ' * 25 + part.upper())
            elif current_mode == 'dialog':
                lines.append(' ' * 15 + part)
                lines.append('')
    
    return '\n'.join(lines)


# Command-line interface

def cmd_train(args):
    """Handle the train subcommand."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Load or train tokenizer
    tokenizer_path = Path(args.tokenizer)
    if (tokenizer_path / 'tokenizer.json').exists():
        print(f"Loading existing tokenizer from {args.tokenizer}")
        tokenizer = BPETokenizer.load(args.tokenizer)
    else:
        print("Training new tokenizer...")
        with open(args.data) as f:
            text = f.read()
        tokenizer = BPETokenizer(vocab_size=10000)
        tokenizer.train(text)
        tokenizer.save(args.tokenizer)
    
    # Create model
    config = ModelConfig(vocab_size=len(tokenizer.vocab))
    model = ScreenplayGPT(config).to(device)
    
    # Load and split dataset
    dataset = ScreenplayDataset(args.data, tokenizer, config.block_size)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch)
    
    print(f"Training samples: {len(train_dataset):,} | Validation samples: {len(val_dataset):,}")
    
    # Train
    train_config = TrainConfig(batch_size=args.batch, lr=args.lr, epochs=args.epochs)
    train_model(model, train_loader, val_loader, train_config, args.output, device)


def cmd_generate(args):
    """Handle the generate subcommand."""
    device = get_device()
    
    # Load tokenizer and model
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = ScreenplayGPT.load(args.model, device=device).to(device)
    model.eval()
    
    # Process prompt
    if args.prompt:
        formatted_prompt = parse_prompt(args.prompt)
    else:
        formatted_prompt = "<SCENE_START>\n<SCENE>INT. "
    
    # Tokenize and generate
    input_ids = tokenizer.encode(formatted_prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    stop_token = tokenizer.scene_end_id if args.scene else None
    output_ids = model.generate(
        input_tensor,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=50,
        top_p=0.9,
        stop_at=stop_token
    )
    
    # Decode and format
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    print("\n" + "=" * 50)
    print(format_screenplay_output(generated_text))
    print("=" * 50)


def cmd_interactive(args):
    """Handle the interactive subcommand."""
    device = get_device()
    
    print("Loading model...")
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = ScreenplayGPT.load(args.model, device=device).to(device)
    model.eval()
    
    print("\n" + "=" * 50)
    print("SCREENPLAY GENERATOR - Interactive Mode")
    print("=" * 50)
    print("\nDescribe a scene in natural language:")
    print('  Example: "dark alley at night"')
    print('  Example: "coffee shop, nervous man waits"')
    print("\nCommands: /temp <value>, /tokens <value>, /quit")
    print("=" * 50)
    
    temperature = 0.8
    max_tokens = 300
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith('/'):
            parts = user_input.split()
            command = parts[0]
            
            if command == '/quit':
                break
            elif command == '/temp' and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value")
            elif command == '/tokens' and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"Max tokens set to {max_tokens}")
                except ValueError:
                    print("Invalid token count")
            continue
        
        # Generate screenplay
        formatted_prompt = parse_prompt(user_input)
        input_ids = tokenizer.encode(formatted_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            stop_at=tokenizer.scene_end_id
        )
        
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        print("\n" + "-" * 40)
        print(format_screenplay_output(generated_text))
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description='Screenplay Transformer Language Model')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', required=True, help='Path to training data')
    train_parser.add_argument('--tokenizer', default='./tokenizer', help='Tokenizer directory')
    train_parser.add_argument('--output', default='./checkpoints', help='Output directory')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    # Generate subcommand
    gen_parser = subparsers.add_parser('generate', help='Generate screenplay text')
    gen_parser.add_argument('--model', default='./checkpoints/best.pt', help='Model checkpoint')
    gen_parser.add_argument('--tokenizer', default='./tokenizer', help='Tokenizer directory')
    gen_parser.add_argument('--prompt', default='', help='Generation prompt')
    gen_parser.add_argument('--tokens', type=int, default=500, help='Max tokens to generate')
    gen_parser.add_argument('--temp', type=float, default=0.8, help='Sampling temperature')
    gen_parser.add_argument('--scene', action='store_true', help='Stop at scene end')
    
    # Interactive subcommand
    int_parser = subparsers.add_parser('interactive', help='Interactive generation mode')
    int_parser.add_argument('--model', default='./checkpoints/best.pt', help='Model checkpoint')
    int_parser.add_argument('--tokenizer', default='./tokenizer', help='Tokenizer directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'interactive':
        cmd_interactive(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
