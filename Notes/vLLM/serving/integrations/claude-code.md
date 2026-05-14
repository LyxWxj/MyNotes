---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/integrations/claude_code.md
---

# Claude Code Integration

[Claude Code](https://code.claude.com/docs/en/quickstart)是Anthropic的官方代理编码工具，存在于终端中。它可以理解代码库、编辑文件、运行命令，并帮助更高效地编写代码。

通过将Claude Code指向vLLM服务器，可以使用自己的模型作为后端，而不是Anthropic API。

## 使用场景

- 运行完全本地/私有的编码辅助
- 使用具有工具调用能力的开放权重模型
- 使用自定义模型进行测试和开发

## 工作原理

vLLM实现Anthropic Messages API，这是Claude Code用于与Anthropic服务器通信的相同API。通过设置`ANTHROPIC_BASE_URL`指向vLLM服务器，Claude Code将其请求发送到vLLM而不是Anthropic。

这意味着任何由vLLM服务的具有适当工具调用支持的模型都可以作为Claude Code中Claude模型的直接替代品。

## 要求

Claude Code需要具有强大工具调用能力的模型。模型必须支持OpenAI兼容的工具调用API。

## 启动vLLM服务器

使用支持工具调用的模型启动vLLM：

```bash
vllm serve openai/gpt-oss-120b --served-model-name my-model --enable-auto-tool-choice --tool-call-parser openai
```

## 配置Claude Code

使用指向vLLM服务器的环境变量启动Claude Code：

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_OPUS_MODEL=my-model \
ANTHROPIC_DEFAULT_SONNET_MODEL=my-model \
ANTHROPIC_DEFAULT_HAIKU_MODEL=my-model \
claude
```

### 环境变量

| 变量 | 描述 |
|------|------|
| `ANTHROPIC_BASE_URL` | 指向vLLM服务器（默认端口8000） |
| `ANTHROPIC_API_KEY` | 可以是任何值，因为vLLM默认不需要身份验证 |
| `ANTHROPIC_AUTH_TOKEN` | 必需。可以是任何值。 |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Opus级请求的模型名称 |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Sonnet级请求的模型名称 |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Haiku级请求的模型名称 |

> **提示**：可以将这些环境变量添加到shell配置文件（如`.bashrc`、`.zshrc`）、Claude Code配置文件（`~/.claude/settings.json`）或创建包装脚本。

> **警告**：Claude Code最近开始在系统提示中注入每请求哈希，这可能破坏[前缀缓存](../../design/prefix_caching.md)。在vLLM版本> 0.17.1中自动解决，但对于旧版本，应在`~/.claude/settings.json`的`"env"`部分添加`"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"`。

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| **Connection refused** | 确保vLLM正在运行且可从指定URL访问。检查端口是否匹配。 |
| **Tool calls not working** | 验证模型支持工具调用，并使用正确的`--tool-call-parser`标志启用。 |
| **Model not found** | 确保`--served-model-name`与环境变量中的模型名称匹配。注意不能在模型名称中使用`/`。 |
