---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/distributed_troubleshooting.md
---

# Troubleshooting Distributed Deployments（分布式部署故障排除）

## 验证节点间GPU通信

启动Ray集群后，验证跨节点的GPU到GPU通信。正确配置可能很复杂。

如果需要额外的环境变量进行通信配置，将它们追加到[examples/online_serving/run_cluster.sh](../../examples/online_serving/run_cluster.sh)，例如`-e NCCL_SOCKET_IFNAME=eth0`。

**建议**：在集群创建期间设置环境变量，因为变量会传播到所有节点。在shell中设置环境变量仅影响本地节点。

## 错误：No available node types can fulfill resource request

即使集群有足够的GPU，也可能出现此错误。问题通常发生在节点有多个IP地址且vLLM无法选择正确的IP时。

**解决方案**：通过在`run_cluster.sh`中设置`VLLM_HOST_IP`确保vLLM和Ray使用相同的IP地址（每个节点使用不同的值）。使用`ray status`和`ray list nodes`验证选择的IP地址。

## Ray可观测性

调试分布式系统可能因规模和复杂性而具有挑战性。Ray提供了一套工具来帮助监控、调试和优化Ray应用程序和集群。

### 相关资源

- [Ray可观测性文档](https://docs.ray.io/en/latest/ray-observability/index.html)
- [Ray调试指南](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/index.html)
- [KubeRay故障排除指南](https://docs.ray.io/en/latest/serve/advanced-guides/multi-node-gpu-troubleshooting.html)
