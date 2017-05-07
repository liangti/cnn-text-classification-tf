## Result

With word embedding vocab_size 600K and dimension 300 trained on 300K corpus.

| Model | Features | Batch size | Filter | L2 | Dropout | N hidden | Best accuracy |
| ----- | -------- | ---------- | ------ | -- | ------- | -------- | ------------- |
| MLP | WV_skipgram | 200 | n/a | 1e-5 | 1 | 50 | 0.733 |
| CNN | WV_skipgram | 64 | 2,3,4 | 0 | 0.5 | 384 | 0.761 |
| CNN | WV_skipgram | 64 | 2,3,4 | 1e-2 | 0.5 | 384 | 0.757 |
