class MyGRUAttention(nn.Module):
    def forward(self, x, hidden):
        attn_out, _ = self.attention(x, hidden)  # (B, T, hidden_dim)
        _, hidden_new = self.gru(attn_out, hidden)  # (B, T, hidden_dim)
        return attn_out, hidden_new