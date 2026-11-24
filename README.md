**Transformer**

This project is a full transformer written from scratch in C.


**Structs**

There are three important structs:

  transformer   learned weights
  runtime       activations + forward caches + buffers used in backward
  grads         a clone of transformer weights used for training updates


**Training Step**

  1. model_forward - fill runtime caches and logits
  2. cross_entropy_next_token - compute loss, write rt->dlogits
  3. grads_zero - clear gradients
  4. model_backward - create gradients from dlogits
  5. sgd_step - actual update: w -= lr * dw


**Constants**

  vocab_size - unique token count
  dim - residual / embedding width
  max_context - longest sequence we can allocate
  num_layers - number of transformer blocks
  num_heads - attention heads: head_dim = dim / num_heads
  hidden_dim - 4 * dim


**Loss and Dlogits**

the intuition, as always, is that probability for supervised next token(f(i) = y) should be 1 while all others are 0.
loss is determined with -log probability over the softmax: 

  p = softmax(z)
  contribution to loss:  -log(p[y])

softmax + cross entropy gradient for any position:

  dL/dz_i = p_i - 1[i == y]

raise the logit of the correct token, lower everything else


**Calculus**

Matmul:
Forward
  C = A @ B
  A is M x K,  B is K x N,  C is M x N
Backward
  dA = dC @ B^T      (M x N)(N x K) -> M x K
  dB = A^T @ dC      (K x M)(M x N) -> K x N

Residual add:
Forward
  residual = x + update
Backward
  d(update) = dresidual
  d(x)      = dresidual

Layer normalization:
Forward
  xhat_i = (x_i - mean) * inv_std
  y_i    = xhat_i * gamma[i] + beta[i]
Backward
  dgamma[i] += sum_over_rows  dout[i] * xhat[i]
  dbeta[i]  += sum_over_rows  dout[i]
  dxhat_i = dout_i * gamma[i]

Gelu grad:
Forward
  u = sqrt(2/pi) * (x + 0.044715 * x^3)
  tanh_u = tanh(u)
  sech2  = 1 - tanh_u^2
  du     = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
Backward
  gelu'(x) = 0.5 * (1 + tanh_u) + 0.5 * x * sech2 * du

Softmax:
Forward
  p = softmax(s)
Backward
  ds_i = p_i * ( dp_i - sum_j p_j * dp_j )


**Code**

  Makefile - builds with entry at main.c

  src/model.c - structs, init, forward pass
  src/train.c - grads, loss, backward, SGD
  src/helpers.c - matmul / softmax / gelu / randf
  src/normalization.c - layer_norm
  src/main.c - entrypoint

  include/model.h
  include/train.h
  include/helpers.h
  include/normalization.h
