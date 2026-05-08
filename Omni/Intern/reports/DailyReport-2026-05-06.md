# 娄雨轩 Daily Report — 2026-05-06

## 今日工作

1. **vllm-omni 代码阅读**：
   - 对比了 `vllm-omni` 主分支与 `vllm-omni-lingchaofan` 分支的差异，梳理了分支的主要修改内容。


2. **扩散模型与并行推理文献阅读**：
   - 了解了 FlowMatchEulerDiscreteScheduler 与 Flux 模型 muShift 策略的原理。
   - 阅读了 Accelerating Parallel Sampling of Diffusion Models、DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models、A PyTorch-native Inference Engine with Hybrid Cache Acceleration and Parallelism for DiTs 等关于 Diffusion 并行的文章。

3. **待确认问题**：
   - `multi_instance_scheduler.py` 中 `torch.device("meta")` 创建的 transformer 模型（line 309-314）被后续在 CPU 上的第二次创建（line 337-342）直接覆盖，疑似冗余代码。
<div style="page-break-after: always;"></div>
```python
def _init_diffusion_instance(self):
	...
	# multi_instance_scheduler.py: line308:
	# 关键修复：使用meta设备创建模型结构，避免CPU上分配float32内存
	logger.info(f"Rank {rank}: Creating transformer model on meta device")
	with torch.device("meta"):
		with set_current_vllm_config(vllm_config):
			transformer = QwenImageTransformer2DModel(
				od_config=self.od_config, **transformer_kwargs
			)
	...
	if weight_files:
		...
		# multi_instance_scheduler.py: line333:
		# 2. 设置默认dtype为bfloat16
		torch.set_default_dtype(torch.bfloat16)
		
		# 3. 在目标设备上创建模型
		logger.info(f"Rank {rank}: Creating transformer on {device} in bfloat16")
		try:
			with set_current_vllm_config(vllm_config):
				transformer = QwenImageTransformer2DModel(
					od_config=self.od_config, **transformer_kwargs
				)
		finally:
			torch.set_default_dtype(original_dtype)
		
		# 4. 验证模型dtype
		sample_param = next(transformer.parameters())
		logger.info(f"Rank {rank}: Model created with dtype: {sample_param.dtype}")
		...
```

## 明日计划

- 了解 diffusion 量化方案。
- 尝试跟踪源码设计中模型参数值、激活值的 dtype 改变情况。
