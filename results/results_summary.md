# Transformer Ablation Study Results

## Performance Summary

| Model | Parameters | Val Loss | Perplexity | Training Time | Description |
|-------|------------|----------|------------|---------------|-------------|
| no_pos_encoding | 30.4M | 3.1154 | 22.54 | 2240s | 移除位置编码 - 4层, 8注意力头, 256维 |
| baseline | 30.4M | 3.2487 | 25.76 | 1996s | 基准模型 - 4层, 8注意力头, 256维 |
| small_layers | 26.8M | 3.2653 | 26.19 | 1491s | 减少层数 - 2层, 8注意力头, 256维 |
| small_heads | 30.4M | 3.2784 | 26.53 | 1984s | 减少注意力头 - 4层, 4注意力头, 256维 |
| tiny_model | 12.5M | 3.6285 | 37.65 | 1004s | 超小模型 - 2层, 4注意力头, 128维 |
| small_dim | 13.4M | 3.6719 | 39.32 | 1438s | 减小维度 - 4层, 8注意力头, 128维 |

## Test Set Performance

| Model | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 |
|-------|----------|----------|----------|----------|----------|----------|
| no_pos_encoding | 18.37 | 15.40 | 17.68 | 17.08 | 23.42 | 20.07 |
| baseline | 20.97 | 17.60 | 20.01 | 19.31 | 26.18 | 22.45 |
| small_layers | 21.45 | 17.61 | 20.14 | 19.48 | 26.55 | 22.67 |
| small_heads | 21.90 | 18.14 | 20.86 | 20.16 | 26.81 | 23.22 |
| tiny_model | 30.56 | 25.28 | 28.89 | 28.15 | 37.77 | 32.58 |
| small_dim | 31.23 | 26.63 | 29.92 | 29.34 | 38.70 | 33.70 |

## Key Findings

- **Best Model**: no_pos_encoding (Perplexity: 22.54)
- **Most Efficient**: tiny_model (12.5M parameters)
- **Fastest Training**: tiny_model (1004 seconds)
- **Position Encoding Impact**: Removing positional encoding increases perplexity by -12.5%
