# 260831-1400 brush-fit-deform-10k — 仅 phase 训练 10k 步 (time 已移除)

## 目的
验证移除 time 输入 (仅 phase 驱动, 52af82d2) 后 10k 步训练的收敛质量。
不 eval (eval-every=10000, 仅最后一步 eval 一次) 以节省时间。

## 运行命令 (commit: 52af82d2)
```bash
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/brush-fit deform images/rotate_dsa_raw_gamma_preprocessed.dcm \
  --points=5000 --iters=10000 --refine-every=400 --eval-split-every=5 \
  --eval-views=8 --eval-every=10000 --save-ply \
  --out=experiments/output/260831-1400_brush-fit-deform-10k
```

## 结果
| iter | loss | PSNR | SSIM | LPIPS | splats |
|---|---|---|---|---|---|
| init | - | 5.78 | 0.137 | 0.6824 | 5000 |
| 10000 | 0.0493 | **43.61** | 0.993 | **0.1440** | 39326 |

- 训练时间 ~9m51s (不含周期性 eval, 仅最后 8 views eval)
- volume phase=0: max|d|=50.27mm (5000 步 time 版为 79.5mm — 10k 步场更收敛)
- 对比 (time 版, 260831-1300 验证): 5000 步 PSNR 39.76 / LPIPS 0.2616 —
  仅 phase + 10k 步显著更好 (LPIPS 0.1440); 步数差为主要因素, time 移除的
  独立贡献需同步数对比确认
- 梯度平稳: rot 3.2e-5 / scale 6.3e-5 (无过拟合迹象, 无 time 编码过拟合)

## 后续
- 同 10k 步跑 time 版对比 (还原 enable_time) 定量确认 time 移除效果
- deform 场位移分布诊断 (网格场导出 --save-deform)
