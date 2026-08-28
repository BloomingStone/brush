# 260828-1316 rxa-static-refinefix / 5000p-10k

## 目的
验证 refine_until_frac 修复: 最后一步 refine 不再无条件剪枝, 导出的
bin/ply 与训练最后 eval 应一致 (消除 +0.05 proj 均匀雾)。

## 命令
```bash
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/fit_static images/RXA_chest.dcm --points=5000 \
  --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=1000 \
  --cull-density=0.001 --iters=10000 --dump-bin-every-eval \
  --out=experiments/output/260828-1316_rxa-static-refinefix/5000p-10k/
```

## 结果 / 根因定位 (00:47)

### 确认的事实 (现有 bin + 导出时 forward 重渲)
- forward vs backward(autodiff) 渲染同一 .bin: max diff 2.4e-7 (逐位一致) → **渲染函数等价**
- 训练 eval pred vs 导出 .bin forward: AVG PSNR 34.68 vs 23.95, Δ=10.7 dB; 各 view corr 0.80-0.93, mean(pg-pp)≈+0.03~0.05 **均匀雾** (非边缘错位)
- `refine_until_frac=0.9` 生效: 最后 refine 8800 (0.88<0.9), 9200/9600/10000 冻结, 导出 = eval = 14280 splat → **最后一步剪枝不是根因**
- 同一 canonical (14280 splat), 同相机, 同 GT:
  - 训练 eval_view (autodiff, lift 后渲染): PSNR(pred,GT)=34.52
  - 导出时 render_xray_forward 重渲: PSNR(FWD,GT)=23.23, max|FWD-pred|=0.33 (均匀雾)
- gs2volume forward(.bin) = 导出 forward(canonical) = 23 (一致); .bin == canonical (into_data 忠实)

### 结论
导出参数 (.bin/.ply) 在 **forward 路径** 渲染时带 ~+0.05 proj 均匀雾 (23 dB);
训练 **eval_view 的 autodiff/lift 原位渲染** 少了这段密度 (34.5 dB)。两者对同一
canonical 不一致 → 10.7 dB 差距。

tty 已排除: pass(Forward==Backward 逐位一致), backend(Autodiff<Wgpu>), 相机/GT(逐像素一致),
splat 数(相同 14280), .bin==canonical, lift(value-preserving from_inner)。

**尚存机制**: 训练循环内的 eval 渲染读到未完全同步/materialize 的 canonical 张量
(异步提交未 sync → 读到上一步的密度; 导出时 into_data/forward 强制 sync → 读到当前完整密度)。
需要: 让训练 eval 在渲染前同步/materialize canonical, 使其与导出一致。

### 未决
- 到底是 eval(autodiff) 少读密度(模型真带雾, 训练指标 34.5 虚高) 还是 forward 多读密度
  (导出参数带假雾)。需在训练循环内强制 sync 后对比 eval vs 导出, 或对已知单 splat 做
  forward/backward 逐像素差分对照。

## 根因定论 (异步写未 flush → eval 读到陈旧 canonical)
- canonical_10000 (in-loop eval 时 dump) 与 canonical_final (导出) **逐字节一致 (diff=0)** —
  canonical 在 eval 与导出时数值其实相同。
- 但同一 canonical:
  - 训练 eval (无论 lift+bwd 还是 forward) → PSNR ~34.5 (inflated)
  - 导出时 forward 重渲 / gs2volume forward → PSNR ~23 (+0.05 proj 均匀雾)
- 二者 render_xray_forward 逐位等价、同后端(Wgpu)、同相机、同 GT、同字节 → 唯一差异是
  **渲染时刻**: in-loop eval 在优化器步的 Wgpu 异步写**尚未 flush** 时读 canonical 的
  `.primitive`, 读到**上一层的陈旧/偏轻密度** (34.5); 导出时 (队列已排空/into_data 强制同步)
  读到**当前完整密度** (23, 真带 +0.05 fog)。
- 即 **模型真实渲染 (forward, 导出/gs2volume/voxelizer/DRR 一致) 带 +0.05 proj 雾 (~23dB)**;
  训练 eval/loss 因读到陈旧 canonical 而**低估该雾, 指标虚高 (34.5)**。故导出参数与
  训练 eval 之间 10.7 dB 的"雾"差异是 **训练循环异步写未 flush 的读数问题**, 非 voxelizer/DRR bug。

### 对 voxelizer/DRR 验证的意义
导出参数在 forward 路径 (gs2volume/voxelizer/DRR) 全部一致 (GS≈DRR), pipeline 本身正确。
gt_pred 的 pred 列由训练 eval 渲染 (陈旧读 → 34.5), 与 compare 的 forward GS (23) 不同，
是训练读数问题, 不是 pipeline 不一致。

### 建议修复 (需后端同步)
- 在 step() 更新 canonical 后、进入 eval/下一步前, 强制同步/物化 canonical
  (flush Wgpu 队列), 使 loss 与 eval 读到当前 (含雾) 值 → 训练才会对雾施加梯度并消除之。
- burn/wgpu 无直接 `Device::sync()`; 需通过 readback 或 backend API 实现, 属后端修改。
- 规避方案: 训练指标用导出参数经 gs2volume forward 重渲 (已一致), 不依赖 in-loop eval。

## 2026-08-28 深入排查 (debug 构建 + wgpu validation + 逐层 DIAG)

### 更正
- 之前把 compare_gs_proj.nrrd (raw proj 域) 当 intensity 比较 → 虚高 14 dB。
  修正后: 同一 .bin 在 fit_static 与 gs2volume 两进程渲染**逐位一致** (0.4825)。

### 已确认
- 训练循环内 eval/loss 渲染 (lift+bwd) 偏亮 (mean_int 0.494 vs 导出 0.480),
  与导出 forward 渲染不一致 (view4 PSNR 差 ~12 dB)。这是 34.5 vs 23 dB 现象的本质。
- canonical 数值 (into_data) 渲染前/后逐位正确 (nonzero=0)。
- 但**渲染输入 buffer 渲染后被覆盖 70% (maxdiff ~30)** → 渲染输出 buffer 复用了
  渲染输入 buffer 的地址 (内存池) → 内核读写别名 → 渲染结果损坏 (偏亮)。
- 根因: cubecl/burn 内存池在"已提交内核的输入 buffer"上复用 (排队内核的绑定
  不持有 buffer 生命周期) → 训练循环中 (内存压力大) 触发; 导出时刻无此问题。
- 已排除: 相机/GT/PLY/渲染函数/stream (单线程+固定 stream 0 均试)/voxelizer
  (probe_before/after 逐位一致)。

### 未解决
- 需要 cubecl/burn 层修复: 渲染输入 buffer 在已提交内核执行前不被内存池复用
  (drop queue / 分配规避 / 内核绑定持 Arc)。应用层无可靠绕过 (重建/_keep/sync
  均无效——free 不看 fusion count)。
- 过渡方案: 训练指标用导出 forward 重渲 (gt_pred_10000_FWD.nrrd / gs2volume) 度量。

## 2026-08-28 深度排查续 (应用层修复穷尽)

### 尝试过的修复 (全部无效)
- 渲染前重建 canonical (host 往返 from_raw, 全新 id): eval 仍偏亮。
- 保持渲染输入 CubeTensor (resolve, descriptor 计数≥2 → is_free=false): 无效。
- 渲染提交后 client.sync (等 GPU): 无效。
- tokio current_thread 单线程: 无效。
- cubecl-common StreamId::current() 固定 0 (消除跨 stream shared_view):
  编译生效确认 (rlib 重编, stream 均=0), 仍无效。

### 新增关键证据
- Adam 更新后立即 into_data (step 内): **读到新值** (quats 在变) → canonical 值
  渲染前是好的。
- 渲染输入 (lift 后) 与 canonical: **不同 FusionTensor id** (975238 vs 974952),
  内容 70% 不同 (maxdiff~28) → lift 拿到的不是 canonical 的张量。
- 重建 + forward 渲染 (与导出完全同路径同值): eval 仍偏亮 (0.493 vs 0.478)
  → 渲染输入 buffer 在渲染内核执行时仍被覆盖, 与代码路径无关。
- cubecl slice 复用只看 descriptor 计数 (is_free, ≤1 即复用), 不检查 GPU
  是否仍在读; drop queue 延迟释放的是存储, 不阻止 slice 复用。

### 结论 (诚实)
应用层无法根本修复: 渲染输入 buffer 在训练循环中被覆盖的机制在
cubecl/burn 的 buffer 生命周期 × 渲染 pipeline 交互层, 保持 descriptor/
重建/同步/单 stream 均不能阻止。需要 GPU 层调试 (RenderDoc 看 buffer
binding / watchpoint 定位写方) 或 cubecl 上游修复 (如内核绑定持 buffer
强引用直到执行 / slice 复用尊重 pending 内核)。当前评估口径: 训练指标
不可靠, 以导出 forward 重渲 (gt_pred_10000_FWD / gs2volume) 为准。
