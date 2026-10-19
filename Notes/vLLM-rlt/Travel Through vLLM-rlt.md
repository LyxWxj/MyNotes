## Initialization

In `vllm_rlt/entripoints/llm.py`

```
LLM(model_path)
	-> model, tokenizer
	-> LLMEngine = StageEngine
```

### Model

In `vllm_rlt/models/__init__.py`

```text
model_path 
	-> model_name 
	-> AutoModelForCausalLM.from_pretained(model_name,...)
	-> dispatch_from_config
	-> specific_model_cls loads the model from model_path
	-> model, tokenizer
```

### LLMEngine/StageEngine

In `vllm_rlt/engine/stage_engine.py`

```text
StageEngine(model, configs..., StageConnector)
	-> Profiler(next(model.parameters()).device)
	-> Save and check configs(scheduler/execution/speculative...)
	-> Create attention from backend("torch"/"triton")
	-> Initialize KVCacheManager
	-> Initialize Scheduler
	-> Connector
	-> Initialize ModelRunner
	-> Initialize SpeculativeRunner / AsyncSpeculativeRunner
	-> Initialize Executor
	-> Initialize PreemptionManager
```

#### Profiler

In `vllm_rlt/profiling.py`

```python
def __init__(self, device):
	self.device = device
	self.current = None # lazy initialization when first `start()`.
```

#### Attention Backend

In `vllm_rlt/attention/backend.py`

```text
create_backend(
	backend: str,
	device: torch.device,
	dtype: torch.dtype,
	head_dim: int,
	block_size: int,
	*,
	required_capabilities: BackendCapabilities = BackendCapabilities()
)
	-> validate_backend_name(backend)
		-> "torch", "triton","flash_attn","flash_attn_2/3/4" are available
	-> pick candidates order= "flash_attn_4/3/2"-> "triton" -> "torch"
	-> for every candidate:
		-> Initialize an instance of candidate AttentionBackend
		-> check `required_capabilities`: "packed_prefill","cuda_graphs", "async_scheduling"
		-> find first candidate's implementation satisfying  `required_capabilities`
```

```python
class AttentionBackend:
	kernel: Callable
	capabilities: BackendCapabilities # packed_prefill, cudagraphs, async_scheduling
	info: dict
```

AttentionBackend forwards parameters to kernel.operator() or kernel.prefill()

But only FlashPagedAttention offers member function `prefill()`.

In `vllm_rlt/kernels/flash_attention.py`

```text
class FlashPagedAttention:
	-> initialize flash_attn generation # 2,3,4
	-> module, package, function = {
		2: ("flash_attn", "flash-attn", "flash_attn_with_kvcache"),
		3: ("flash_attn_3.flash_attn_interface", "flash-attn-3", "flash_attn_with_kvcache"),
		4: ("flash_attn.cute", "flash-attn-4", "flash_attn_varlen_func"),
	}[self.generation]
```

Only FlashAttention-4 offers member function `prefill()` for packed prefill.

>[!info]
>
> |Aspect|Packed Prefill|Non-Packed Prefill|
|---|---|---|
|Basic idea|Groups consecutive query tokens from each request into variable-length sequences|Treats each query token as an independent single-query row|
|Example layout|`A0 A1 A2 B0 B1` with `cu_seqlens_q = [0, 3, 5]`|Five independent rows: `A0`, `A1`, `A2`, `B0`, `B1`|
|Query metadata|One sequence boundary description per request/depth group|One block table and context length per token row|
|Attention input|Ragged multi-query input|Batch of single-query inputs|
|Attention kernel|Dedicated packed prefill kernel, such as FlashAttention-4 prefill|Standard paged single-query attention path|
|Causal masking|Kernel uses sequence boundaries and per-sequence prefix lengths|Each row uses its own context length|
|Cross-request isolation|Preserved by `cu_seqlens_q` and separate page tables|Preserved by separate page tables and row-level context lengths|
|Padding|Usually avoids padding within each variable-length query sequence|May require batch-level padding if static buffers or power-of-two padding are enabled|
|KV cache|Same per-request, depth-specific KV cache|Same per-request, depth-specific KV cache|
|Logical model computation|Same outputs for the same real tokens|Same outputs for the same real tokens|
|Metadata overhead|Lower: fewer sequence-level metadata entries|Higher: repeated metadata for every token row|
|Kernel efficiency|Can better exploit contiguous, variable-length prefill work|May do more per-row metadata processing and less specialized prefill work|
|Performance|Often better for long or highly variable prompt chunks, but hardware-dependent|May be simpler and competitive for short or uniform batches|
|Backend requirement|Requires a backend that supports packed prefill; in this repository, FA4|Works with ordinary Torch, Triton, FA2/FA3/FA4-compatible paths|
|Main trade-off|More specialized metadata and backend constraints|Simpler implementation and broader backend compatibility|

**Key point:** packed prefill changes more than metadata representation: it selects a different attention execution path. The actual speedup depends on prompt lengths, batch composition, backend, and hardware.

Otherwise, forward parameters to `operator()`

```python
def __call__(self, q, key_cache, value_cache, block_tables, context_lengths)
```

#### KVCache

In `vllm_rlt/engine/stage_engine.py`

```text
-> num_blocks. self.memory_plan = plan_cache(...)
-> self.cache_manager = KVCacheManager(
	**config,
	num_blocks,
	**cache_config,
	attention_backend,
	attention,
	...
)
```

In `memory.py` `plan_cache(model, cache, scheduler, execution, backend):`

1. How many bytes per block's Token KV：

```python
bytes_per_block = (
	2                            # every element for both K,V
	* config.num_hidden_layers   # num of layers
	* cache.block_size           # num of Tokens for every block
	* config.num_key_value_heads # num of heads for every key or value
	* config.head_dim            # dimension num of every head
	* parameter.element_size()   # precision 
)
```

Then calculate how many blocks should reserve (if no num_blocks provided).

```text
-> if offered num_blocks/kv_cache_memory_bytes or device.type != "cuda"
	num_blocks calculated by thoses configs and return.
-> snapshot cuda device status(free_before, total, reserved, allocated)
-> use `count // cache.block_size` as a temp `num_blocks` to initialize a temp KVCacheManager.
-> construct a biggest tensor and run (prelude->max_loops*recurrent->coda)
-> peak = max(0, torch.cuda.max_memory_allocated(device) - allocated)
-> buffers = execution_buffer_bytes()
-> calculate budget
-> calculate num_blocks.
-> return blocks, source="cuda_profile"
```

In `vllm_rlt/worker/buffer.py` `execution_buffer_bytes()`:

```text
calculates the total bytes of runtime buffer:
-> Workspace objects*4, each contains two double-buffered groups: core and boundary
-> Hidden_states for batching Tokens flatten
-> Device KV page tables
-> Runtime batching metadata: (tokens, positions, blocks, offsets, lengths)
-> One hidden-state for last token per request for coda.
-> RoutingBank objects
-> Routing descriptors
-> AsyncState
-> Two finalize descriptor banks
-> prefill_states for split-prefill hidden states across scheduling boundaries.
-> prefill_hidden Gathers scatters staged states into a contiguous recurrent input batch.
-> Prefill metadata pool.
```

#### KVCacheManager

In `vllm_rlt/core/kv_cache_manager.py`

```python
shape = (num_blocks,num_layers, block_size, num_kv_heads, head_dim)
self.key_cache = torch.empty(shape)
self.value_cache = torch.empty_like(self.key_cache)
```

#### Scheduler

In `vllm_rlt/core/scheduler/py`

```text
Scheduler(scheduler_config, cache_manager, speculative_config, accepts_prompt, produces_tokens):
-> Owns the reference of scheduler_config, cache_manager, speculative_config
-> Initialize queues:
	requests: dict[str, Request] # map request_id to Request instance.
	waiting: deque[str]
	work_queues: dict[WorkKey, deque[str]]
	admission_slots
-> Initialize SchedulingPolicy by configs.
```

In `vllm_rlt/core/execution.py`

```python
class WorkKey(Enum):
    PROMPT_ATOMIC = ("prompt_atomic", Operation.PRELUDE)
    PROMPT_PRELUDE = ("prompt_prelude", Operation.PRELUDE)
    PROMPT_RECURRENT = ("prompt_recurrent", Operation.RECURRENT)
    DECODE_PRELUDE = ("decode_prelude", Operation.PRELUDE)
    DECODE_RECURRENT = ("decode_recurrent", Operation.RECURRENT)
    DECODE_CODA = ("decode_coda", Operation.CODA)
    SPECULATIVE = ("speculative", None)
```

In `vllm_rlt/core/scheduling_policy.py`

We have `RefillPolicy`,`SplitPrefillPolicy` based on RefillPolicy, `MixedPolicy` based on `SplitPrefillPolicy` and `SpeculativePolicy` for Speculative on.

> [!info]
> AtomicPrefill vs SplitPrefill
> Setting by Executionconfig.atomic_prefill: bool
> When atomic_prefill is True, the scheduler schedule a batch by RefillPolicy/NorefillPolicy order from work_queues[PROMPT_ATOMIC] and runner executes once prelude and full times of recurrent as an atomic operation.
> When atomic_prefill is False, the scheduler schedule a batch by SplitPrefillPolicy order from work_queues[prompt_prelude] and work_queues[prompt_recurrent], then runner execute once prelude or recurrent(depend on batch.operation) and re-schedule requests in the next turn.
> `atmoic_prefill` prefills a prompt chunk as an atomic procedure, therefore we don't need to save activations across batch, only need to save the hidden of last token for coda, but in `split_prefill` mode we may need to maintain a buffer for those chunks not prefilled completely, counts in `execution_buffer_bytes()`.

> [!info]
> Self-Speculative
> Few loops decode for draft, full loops as varification.

### PreemptionManager

#### SpeculativeRunner

#### ModelRunner

In `vllm_rlt/worker/model_runner.py`

```text
ModelRunner(model, cache_manager, configs...):
	-> Owns the reference of model and cache_manager
	-> Initialize ExecutionRuntime(model, cache_manager)
	-> Initialize Sampler
	-> Initialize configs # exit/execution/scheduler configs
	-> Initialize graphs # RecurrentGraphs/CodaGraphs
	-> Initialize streams # core/boundary/copy/readback streams
	-> Initialize some runtime buffer pools # Complex
```

In `vllm_rlt/worker/execution.py`

```python
class ExecutionRuntime:
	def __init__(self, model, cache):
		self.model, self.cache = model, cache
	
	def forward(self, batch, ForwardBatch) -> ForwardResult:
		# execute different operation for batch.operation.
		
	def advance(
		self,
		hidden=None,
		metadata=(),
		*,
		input_tokens=None,
		input_kv=None,
		generators=None,
		hidden_buffer=None,
		compute_gate=False,
		graphs=None,
	)-> ExecutionResult:
		# wrap hidden and other metadata as ForwardBatch and pass it to forward().
```

In `vllm_rlt/worker/sampler.py`

```python
class Sampler:
	@staticmethod
	def probabilities(logits: torch.Tensor, params: SamplingParams)->torch.Tensor:
		# output probablities distribution based on temperature, top-k or top-p
		
	@staticmethod
	def draw(probs: torch.Tensor, generator:torch.Generator)->torch.Tensor:
		# sample a token use RNG from distribution
		
	def sample(
		self,
		logits: torch.Tensor,
		params: SamplingParams,
		generator: torch.Generator | None
	)->torch.Tensor:
		# a complete sample, greedy or probabilities + draw.
```

```python
class RngRegistry:
	def __init__(self):
		# initialize mapping: request-id->generator
		
	def peek(self, request_id: str) -> torch.Generator | None:
		# check its generator
		
	def acquire(
		self, request_id: str, params: SamplingParams, device: torch.device
	) -> torch.Generator
		# lazy init or reuse its generator
	
	def drop(self, request_id: str):
		# delete its generator
```

RecurrentGraphs

CodaGraphs

Streams

bufferpools

#### PreemptionManager
