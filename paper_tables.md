

==== PAPER TABLES ====

### Table A — Headline (Day 1 Vanilla DDIM)
| Variant | Split | n | Steps | PSNR | SSIM | LPIPS | Latency/img (s) |
|---|---|---|---|---|---|---|---|
| Vanilla DDIM | eval15 | 15 | 5 | 18.550 | 0.7545 | 0.2526 | 0.499 |
| Vanilla DDIM | lolv2_real | 100 | 5 | 22.772 | 0.7909 | 0.2144 | 0.418 |
| Vanilla DDIM | eval15 | 15 | 20 | 18.468 | 0.7537 | 0.2533 | 2.783 |
| Vanilla DDIM | lolv2_real | 100 | 20 | 22.666 | 0.7900 | 0.2154 | 2.957 |

### Table B — Method ablation (Adaptive Residual Rescaling @ 5 DDIM steps, eval15)
ARR introduces an inference-time dynamic scaling of the residual.
| alpha | PSNR | SSIM |
|---|---|---|
| 0.00 | 18.556 | 0.7548 |
| 0.10 | 18.567 | 0.7546 |
| 0.20 | 18.544 | 0.7528 |
| 0.30 | 18.552 | 0.7515 |
| 0.40 | 18.385 | 0.7485 |
| 0.50 | 18.474 | 0.7481 |

### Table C — Sampler Comparison (@ 5 steps, eval15)
| Variant | Split | n | Steps | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| DDIM | eval15 | 15 | 5 | 18.550 | 0.7545 | 0.2525940219561259 |
| DPM-Posterior | eval15 | 15 | 5 | 16.135 | 0.6732 | - |

### Table D — Step ablation (Day 1 Vanilla DDIM)
| Steps | Split | PSNR | SSIM | Latency/img (s) |
|---|---|---|---|---|
| 5 | eval15 | 18.595 | 0.7555 | 0.501 |
| 5 | lolv2_real | 22.805 | 0.7910 | 0.494 |
| 10 | eval15 | 18.528 | 0.7539 | 1.337 |
| 10 | lolv2_real | 22.693 | 0.7901 | 1.312 |
| 20 | eval15 | 18.490 | 0.7541 | 2.999 |
| 20 | lolv2_real | 22.690 | 0.7903 | 2.953 |
| 50 | eval15 | 18.487 | 0.7529 | 7.926 |
| 50 | lolv2_real | 22.708 | 0.7899 | 7.880 |
| 100 | eval15 | 18.467 | 0.7535 | 16.014 |
| 100 | lolv2_real | 22.632 | 0.7899 | 16.105 |

### Table E — Efficiency (T4, 400x600, Day 1 Vanilla DDIM)
device,resolution,sampler,steps,latency_ms,throughput_imgs_per_s,peak_gpu_mb,total_flops_g,params_m
cuda,400x600,ddim,5,811.81,1.232,1051.8,1005.8,17.973
cuda,400x600,ddim,10,1685.55,0.593,1052.2,2011.59,17.973
cuda,400x600,ddim,20,3297.66,0.303,1052.2,4023.18,17.973

