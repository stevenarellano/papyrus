import math
from typing import Optional
from torch import nn

from typing import Optional
import torch
from torch import nn, Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Initialize the Multi-Head Attention module.
        Args:
            embed_dim (int): Embedding dimension of the input.
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()

        # Prepare the attention head dimension
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads

        # Prepare the query, key, value projections
        self.query_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.key_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.value_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)

        # Prepare vars for scaled dot-product attention
        self.scale: float = self.head_dim ** 0.5

        # Prepare the output projection
        self.output_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the scaled dot-product attention.
        Args:
            query (Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
            value (Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).
            mask (Optional[Tensor]): Attention mask of shape (batch_size, num_heads, seq_len, seq_len).
        Returns:
            Tensor: Output of the attention module of shape (batch_size, seq_len, embed_dim).
        """
        # Get the batch size
        batch_size: int = query.size(0)  # same as key.size(0) or value.size(0)

        # project the query, key, and value
        # (batch_size, seq_len, embed_dim)
        #   -> (batch_size, seq_len, embed_dim)
        Q: Tensor = self.query_proj(query)
        K: Tensor = self.key_proj(key)
        V: Tensor = self.value_proj(value)

        # reshape for multi-head attention
        # (batch_size, seq_len, embed_dim)
        #   -> (batch_size, seq_len, num_heads, head_dim)
        #   -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads,
                   self.head_dim)
        Q = Q.transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads,
                   self.head_dim)
        K = K.transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads,
                   self.head_dim)
        V = V.transpose(1, 2)
        """
        view(*shape) is used to reshape a tensor.
            *  -1 tells the function to calculate the size of the dimension in that place.
            
        Tensor.transpose(dim0, dim1)
            - dim0 (int): The first dimension to be swapped.
            - dim1 (int): The second dimension to be swapped.
        """

        # compute the attention scores
        # (batch_size, num_heads, seq_len, head_dim)
        #   x ((batch_size, num_heads, seq_len, head_dim).transpose(-2, -1)
        #       -> (batch_size, num_heads, head_dim, seq_len))
        #   -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores: Tensor = torch.matmul(
            Q, K.transpose(-2, -1)) / self.scale
        # (batch_size, num_heads, seq_len, seq_len)
        #   -> (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # apply the attention
        # (batch_size, num_heads, seq_len, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_weights: Tensor = torch.nn.functional.softmax(
            attention_scores, dim=-1)
        # (batch_size, num_heads, seq_len, seq_len)
        # -> (batch_size, num_heads, seq_len, head_dim)
        attention_output: Tensor = torch.matmul(attention_weights, V)
        # (batch_size, num_heads, seq_len, head_dim)
        # .transpose(1, 2) -> (batch_size, seq_len, num_heads, head_dim)
        # .view(batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim)

        # project the output
        # (batch_size, seq_len, embed_dim)
        # -> (batch_size, seq_len, embed_dim)
        return self.output_proj(attention_output)


"""
the number of tokens is the sequence length,
and the number of dimensions is the embedding dimension.

many different types of attention mechanisms:
1. Scaled Dot-Product Attention
2. Mutliplicative Attention
3. Additive Attention
"""


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Initialize the Feed-Forward module.
        Args:
            embed_dim (int): Embedding dimension of the input.
            ff_dim (int): Dimension of the hidden layer in feed-forward.
        """
        super(FeedForward, self).__init__()

        self.linear1: nn.Linear = nn.Linear(embed_dim, ff_dim)
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.linear2: nn.Linear = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for the feed-forward network.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output of the feed-forward network.
        """
        # (batch_size, seq_len, embed_dim)
        # -> (batch_size, seq_len, ff_dim)
        x = self.linear1(x)

        # (batch_size, seq_len, ff_dim)
        # -> (batch_size, seq_len, ff_dim)
        x = self.relu(x)

        # (batch_size, seq_len, ff_dim)
        # -> (batch_size, seq_len, ff_dim)
        x = self.dropout(x)

        # (batch_size, seq_len, ff_dim)
        # -> (batch_size, seq_len, embed_dim)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        A single encoder layer.
        Args:
            embed_dim (int): Embedding dimension of the input.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward hidden layer.
            dropout (float): Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder layer.
        Args:
            x (Tensor): Input tensor.
            mask (Tensor, optional): Attention mask.
        Returns:
            Tensor: Output tensor.
        """
        attn_output = self.attention(
            x, x, x, mask)

        x = self.layer_norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))


class Encoder:
    def __init__(self, embed_dim, num_heads, ff_dim, layer_count=6, dropout=0.1):
        """
        # 2 sublayers: (1) multi-head self-attention, (2) simple, position wise fully connected feed-forward
        # followed by layer normalization
        """
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(layer_count)
        ])

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder stack.
        Args:
            x (Tensor): Input tensor.
            mask (Tensor, optional): Attention mask.
        Returns:
            Tensor: Output tensor.
        """

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder stack.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (Tensor, optional): Attention mask of shape (batch_size, num_heads, seq_len, seq_len).
        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        A single decoder layer.
        Args:
            embed_dim (int): Embedding dimension of the input.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward hidden layer.
            dropout (float): Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.enc_dec_attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, enc_dec_mask=None):
        """
        Forward pass through the decoder layer.
        Args:
            x (Tensor): Input tensor.
            enc_output (Tensor): Encoder output tensor.
            self_mask (Tensor, optional): Mask for preventing positions from attending to subsequent positions.
            enc_dec_mask (Tensor, optional): Mask for attention over encoder's output.
        Returns:
            Tensor: Output tensor.
        """
        attn_output = self.self_attention(x, x, x, self_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        enc_dec_output = self.enc_dec_attention(
            x, enc_output, enc_output, enc_dec_mask)
        x = self.layer_norm2(x + self.dropout(enc_dec_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + self.dropout(ff_output))

        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, layer_count=6, dropout=0.1):
        """
        The full decoder stack.
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward hidden layer.
            layer_count (int): Number of decoder layers.
            dropout (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(layer_count)
        ])

    def forward(self, x, enc_output, self_mask=None, enc_dec_mask=None):
        """
        Forward pass through the decoder stack.
        Args:
            x (Tensor): Input tensor.
            enc_output (Tensor): Encoder output tensor.
            self_mask (Tensor, optional): Mask for preventing positions from attending to subsequent positions.
            enc_dec_mask (Tensor, optional): Mask for attention over encoder's output.
        Returns:
            Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, enc_dec_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=ff_dim, dropout=dropout
        )
        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))

        transformer_output = self.transformer(
            src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask
        )
        output = self.fc_out(transformer_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :].to(x.device)


class ModelService:
    def get_model(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        device: torch.device = 'cpu',
    ) -> Transformer:
        """
        Create and return a Transformer model with the specified parameters.

        Args:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward hidden layer.
            num_layers (int): Number of encoder and decoder layers.
            dropout (float): Dropout rate.

        Returns:
            Transformer: The instantiated Transformer model.
        """
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = nn.DataParallel(model)
        model.to(device)
        return model

    def load_model(
        self,
        model_path: str,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
        device: torch.device,
    ) -> nn.Module:
        """
        Load a trained Transformer model from a checkpoint.

        Args:
            model_path (str): Path to the saved model's state_dict.
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            ff_dim (int): The feedforward dimension.
            num_layers (int): The number of layers.
            dropout (float): The dropout rate.
            device (torch.device): The device to load the model on ('cuda' or 'cpu').

        Returns:
            nn.Module: The loaded model.
        """
        model = self.get_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = nn.DataParallel(model)
        model = model.to(device)

        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

        return model
