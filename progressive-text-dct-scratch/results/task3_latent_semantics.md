# Semantic preservation of representations (decoder-free probes)

N=512 test chunks. TF = teacher-forced token acc; RSA = Spearman(Z-space sims, MiniLM sims).

| K | enc_dct TF / RSA | enc_latent TF / RSA | emb_dct TF / RSA |
|---|---|---|---|
| 1 | 0.368 / 0.219 | 0.217 / 0.218 | 0.374 / 0.481 |
| 2 | 0.337 / 0.184 | 0.196 / 0.229 | 0.306 / 0.410 |
| 4 | 0.267 / 0.137 | 0.197 / 0.216 | 0.170 / 0.331 |
| 8 | 0.240 / 0.117 | 0.237 / 0.227 | 0.095 / 0.259 |
| 16 | 0.269 / 0.092 | 0.314 / 0.249 | 0.089 / 0.203 |
| 32 | 0.514 / 0.065 | 0.872 / 0.209 | 0.286 / 0.151 |
| 64 | 1.000 / 0.089 | 1.000 / 0.076 | 1.000 / 0.124 |