import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, val, mask = None):
        """
        query : tensor of shape (batch_size, num_heads, seq_len, d_k)
        key   : tensor of shape (batch_size, num_heads, seq_len, d_k)
        val : tensor of shape (batch_size, num_heads, seq_len, d_v)
        self_mask  : tensor of shape (batch_size, 1, seq_len, seq_len)

        - d_k is usually equal to d_v
        - d_k = dmodel(embedding space dimensionality) / # of heads
        """

        d_k = query.size(-1)
        similarity_scores = torch.matmul(query,key.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        attention_weights = F.softmax(similarity_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        if mask is not None:
            attention_weights[mask == 0] = float(0)

        output = torch.matmul(attention_weights, val)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.d_v = embed_dim // num_heads

        #Linear Layers for Query, key and Values
        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)

        #Output Projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        #scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, self_mask=None, cross_mask=None):
        """
        :param x: input tensor of shape (batch_size_input, seq_len_input, embed_dim)
        :param encoder_output : input tensor to calculate Query and key in cas of cross attention
        :param cross_mask: optional tensor for masking
        :param self_mask: optional Tensor for masking
        """

        batch_size_input, seq_len_input, _ = x.size()
        self_mask_expanded = None
        cross_mask_expanded = None
        if self_mask is not None and len(self_mask.shape) < 4:
            self_mask_expanded = self_mask.unsqueeze(0).unsqueeze(0).expand(batch_size_input, self.num_heads, -1, -1).bool()

        if encoder_output is None:
            #Linear Projections
            query_out   = (self.query(x).                                                       #(batch,seq_len_input,embed_dim)
                           view(batch_size_input, seq_len_input, self.num_heads, self.d_k).     #(batch,seq_len_input,num_heads,d_k)
                           transpose(1,2))                                                      #(batch,num_heads,seq_len_input,d_k)
            key_out     = (self.key(x).
                           view(batch_size_input, seq_len_input, self.num_heads, self.d_k).
                           transpose(1,2))
            value_out   = (self.value(x).
                           view(batch_size_input, seq_len_input, self.num_heads, self.d_k).
                           transpose(1,2))
        else:
            batch_size_encoder, seq_len_encoder, _ = encoder_output.size()
            if cross_mask is not None and len(cross_mask.shape) < 4:
                cross_mask_expanded = cross_mask.unsqueeze(0).unsqueeze(0).expand(batch_size_encoder, self.num_heads, -1,-1).bool()
            query_out   = (self.query(x).
                            view(batch_size_input, seq_len_input, self.num_heads, self.d_k).
                            transpose(1, 2))
            key_out     = (self.key(encoder_output).                                                # (batch_size_encoder, seq_len_encoder, embed_dim)
                           view(batch_size_encoder, seq_len_encoder, self.num_heads, self.d_k).     # (batch_size_encoder, seq_len_encoder, num_heads, d_k)
                           transpose(1, 2))                                                         # (batch_size_encoder, num_heads, seq_len_encoder, d_k)
            value_out   = (self.value(encoder_output).
                            view(batch_size_encoder, seq_len_encoder, self.num_heads, self.d_k).
                            transpose(1, 2))

        #attention_weights (batch_size_input,num_heads,seq_len_input,seq_len_input)
        #attention_output (batch_size_input,num_heads,seq_len_input,d_v)
        if encoder_output is None:
            attention_output, attention_weights = self.attention(query_out, key_out, value_out, self_mask_expanded)
        else:
            attention_output, attention_weights = self.attention(query_out, key_out, value_out, cross_mask_expanded)

        attention_output = (attention_output.transpose(1, 2).contiguous().              #(batch_size_input,seq_len_input,num_heads,d_v)
                            view(batch_size_input, seq_len_input, self.embed_dim))      #(batch_size_input,seq_len_input,embed_dim)

        output = self.out_proj(attention_output)

        return self.dropout(output), attention_weights











