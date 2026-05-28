# Daily Report - 2026-05-28

## 遇到的问题

### 多实例并发 Benchmark 高负载下 HCCL 通信错误

**现象：** 在使用 `run_concurrency_benchmark.sh` 对多实例 Qwen-Image 服务进行并发压测时，不同并发级别的表现差异显著：

| 并发数 | 成功率 | QPS | 中位延迟 |
|--------|--------|-----|----------|
| 4 | 128/128 | 0.10 | 23.3s |
| 8 | 107/128 | 0.10 | 67.8s |
| 16 | 0/128 | 0.00 | - |
| 32 | 0/128 | 0.00 | - |

**错误日志：**
```
RuntimeError: copy_between_host_and_device_opapi ... NPU function error:
SUSPECT REMOTE ERROR, error code is 507057
ERR00100 PTA call acl api failed
```

错误发生在 Rank 0 的 text encoder 执行 `.to(device)` 将 tokenizer 输出搬移到 NPU 时。

**根因分析：**
1. `SUSPECT REMOTE ERROR 507057` 是华为 HCCL 通信层错误，表示 NPU 设备或通信链路进入不可恢复状态
2. 三个并发级别的 benchmark 在**同一进程内顺序执行**，未重启服务
3. 并发 8 时 HCCL 通信出错，NPU 设备状态被污染，导致并发 16/32 全部失败
4. 高并发下 Rank 0 多线程同时发起 HCCL 操作，可能超出 NPU 通信链路承载能力

**待验证方案：**
- 每轮 benchmark 之间重启服务，隔离 NPU 设备状态
- 限制同时在飞请求数（`VLLM_OMNI_MAX_INFLIGHT`），避免 Rank 0 过载
- 单独测试并发 16，确认是否为独立可复现问题
