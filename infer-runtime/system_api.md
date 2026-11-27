# Inference serving system API design draft

Status: proposed v1 contract for [the inference serving
runtime](https://github.com/kunxian-xia/llm-learning-journey/issues/47) and
[scheduler](https://github.com/kunxian-xia/llm-learning-journey/issues/56). This document specifies interfaces; it does
not implement them.

## Table of contents

- [Scope and invariants](#scope-and-invariants)
- [Shared types](#shared-types)
- [Scheduler API](#scheduler-api)
  - [`add_request`](#scheduler-add-request)
  - [`cancel`](#scheduler-cancel)
  - [`schedule`](#scheduler-schedule)
  - [`complete`](#scheduler-complete)
  - [`fail`](#scheduler-fail)
- [KV cache manager API](#kv-cache-manager-api-used-by-the-scheduler)
  - [`estimate_reservation`](#kv-estimate-reservation)
  - [`try_reserve`](#kv-try-reserve)
  - [`prepare_batch`](#kv-prepare-batch)
  - [`commit_batch`](#kv-commit-batch)
  - [`abort_batch`](#kv-abort-batch)
  - [`release`](#kv-release)
- [Model runner API](#model-runner-api-used-by-the-scheduler)
- [One-iteration lifecycle](#one-iteration-lifecycle)
- [Service-facing asynchronous API](#service-facing-asynchronous-api)

## Scope and invariants

The scheduler owns a waiting queue and a running pool. Each running request is in exactly one phase: `PREFILL` or
`DECODE`. There is one model batch in flight at a time. Normal scheduling does not preempt running requests.

Admission and batch planning have different memory checks:

1. **Admission:** Before moving a waiting request to the running pool, reserve enough KV capacity for its entire
   possible run. This lets the request finish without later KV preemption. A request can be admitted whenever capacity
   is available, including capacity left unused by current requests or freed by a completed request.
2. **Batch planning:** Select work that fits the per-forward-pass token, sequence, and execution-workspace limits.
   Allocate physical slots for every KV write before giving the plan to the model runner. A valid KV reservation
   guarantees that these slots can be allocated.

The KV block pool is sized after model weights and a worst-case execution workspace for the configured batch limits have
been accounted for. The runner and scheduler must validate those limits together at initialization; otherwise token and
sequence limits alone cannot guarantee that a forward pass fits GPU memory. A reservation is an accounting claim on the
KV pool; physical blocks may be assigned as the request progresses. The manager must count both allocated blocks and
outstanding reservation claims so that the pool cannot be overcommitted.

## Shared types

The following signatures are illustrative Python types. `SamplingConfig` describes the sampling policy and `RequestId`
is a unique string.

```python
RequestId = str
Phase = Literal["prefill", "decode"]
FinishReason = Literal["stop_token", "length", "cancelled", "unrunnable", "error"]

@dataclass(frozen=True)
class TokenizedRequest:
    request_id: RequestId
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    stop_token_ids: frozenset[int]
    sampling: SamplingConfig

@dataclass
class RunningRequest:
    request: TokenizedRequest
    phase: Phase
    prefill_cursor: int
    generated_token_count: int
    pending_decode_token_id: int | None
    cache_handle: CacheHandle
    cancel_requested: bool = False

@dataclass(frozen=True)
class SchedulerConfig:
    max_batch_tokens: int
    max_batch_seqs: int
    max_prefill_chunk_tokens: int
    max_running_requests: int | None = None
    max_admission_bypasses: int = 8
    max_batch_bypasses: int = 8

@dataclass(frozen=True)
class BatchItem:
    request_id: RequestId
    phase: Phase
    input_ids: tuple[int, ...]
    context_length_before: int
    cache_handle: CacheHandle
    sample_next_token: bool

@dataclass(frozen=True)
class ReadyBatch:
    items: tuple[BatchItem, ...]
    kv_layout: KVBatchLayout

@dataclass(frozen=True)
class ScheduleResult:
    batch: ReadyBatch | None
    events: tuple[Event, ...]

@dataclass(frozen=True)
class ItemOutput:
    request_id: RequestId
    sampled_token_id: int | None

@dataclass(frozen=True)
class RunnerOutput:
    items: tuple[ItemOutput, ...]

@dataclass(frozen=True)
class TokenEvent:
    request_id: RequestId
    token_id: int

@dataclass(frozen=True)
class FinishedEvent:
    request_id: RequestId
    reason: FinishReason

Event = TokenEvent | FinishedEvent
```

A prefill item consumes a nonempty chunk of remaining prompt tokens. Only the final prefill chunk has
`sample_next_token=True`; earlier chunks return no token. A decode item consumes exactly one previously sampled token
that is not yet in KV and has `sample_next_token=True`. After a successful final prefill or decode step, the sampled
token is streamed and held as the next decode input if generation continues. A terminal sampled token need not be
written to KV. The service layer owns text decoding; the scheduler emits token IDs and does not decode individual tokens
into text.

`KVBatchLayout` is an opaque runner-facing description of block tables, physical write slots, and lengths or positions.
The manager owns its physical representation. `RunnerOutput.items` contains exactly one result for each
`ReadyBatch.items` entry, matched by request ID. A partial prefill returns `sampled_token_id=None`; a final prefill
chunk or decode returns a token ID.

## Scheduler API

```python
class Scheduler:
    def add_request(self, request: TokenizedRequest) -> None: ...
    def cancel(self, request_id: RequestId) -> list[Event]: ...
    def schedule(self) -> ScheduleResult: ...
    def complete(self, batch: ReadyBatch, output: RunnerOutput) -> list[Event]: ...
    def fail(self, batch: ReadyBatch, error: Exception) -> list[Event]: ...
```

The running pool is scheduler-owned mutable state, keyed by request ID:

```python
running: dict[RequestId, RunningRequest]
```

`TokenizedRequest` remains immutable. `RunningRequest.phase` tracks the current phase, `prefill_cursor` is the number of
prompt tokens already committed to KV, and `generated_token_count` counts sampled output tokens, including a pending
token. `pending_decode_token_id` is the sampled token to consume on the next decode step; it is `None` before the first
sample or after termination. `cache_handle` owns the request's reservation and blocks. `cancel_requested` defers
cancellation until an in-flight model call no longer uses the request. `BatchItem` is a per-call snapshot derived from
this state, not an entry in the running pool.

<a id="scheduler-add-request"></a>
### `add_request(request) -> None`

Accepts an already tokenized request from the service layer and appends it to the waiting queue. It rejects duplicate
IDs, empty prompts, invalid output limits, requests exceeding model context, and requests whose maximum KV requirement
exceeds the entire pool. It does not reserve memory or move the request into the running pool.

<a id="scheduler-cancel"></a>
### `cancel(request_id) -> list[Event]`

Removes a waiting request immediately, or marks a running request for cancellation and releases its KV resources as soon
as no model call uses them. If the request is in the current batch, cancellation takes effect after that batch returns.
The method returns any `FinishedEvent(reason="cancelled")` ready now; an in-flight request's event is returned by
`complete` or `fail`. Unknown IDs are an error.

<a id="scheduler-schedule"></a>
### `schedule() -> ScheduleResult`

Scans waiting requests and calls `try_reserve` for requests that can fit; each admitted request moves to the running
pool with `phase=PREFILL`; this is a request state, not a separate prefill pool. A request skipped because capacity
is temporarily unavailable accrues one admission bypass for
that scheduling call. Once it reaches `max_admission_bypasses`, the scheduler must not admit newer requests ahead of
it until it is admitted or terminates. This bounds bypasses, not wall-clock time: wall-clock latency also depends on
batch execution time and resource release. It then plans prefill chunks and decode steps within `SchedulerConfig`;
these limits must have been validated against the runner's worst-case execution workspace at initialization.

If a request cannot fit in a batch with other work, the scheduler may defer it. If the request cannot make progress
even as the only item under the configured limits, it is individually unrunnable. `schedule` removes that request from
the active set, releases its cache handle, and includes `FinishedEvent(reason="unrunnable")` in `ScheduleResult.events`.
This check is made against the next required operation, so a request can become unrunnable during decoding. The
scheduler may continue planning other requests in the same call.

If runnable work remains, `schedule` calls `prepare_batch` and returns the fully executable batch in
`ScheduleResult.batch`. If no work can run, `batch` is `None`; `events` may still contain terminal events. Both fields
may be populated in one result. Calling `schedule` while a previous batch is in flight is an error.

The batch selection policy is internal to `schedule`. It must cap a single request's prefill chunk and use spare batch
capacity for other fitting work. Each running request that is eligible for work but omitted from a batch accrues one
batch bypass. Once it reaches `max_batch_bypasses`, the scheduler must prioritize it in the next batch where its next
operation can fit. This bounds scheduling opportunities, not wall-clock time. The scheduler must distinguish this
temporary batch pressure from an operation that cannot fit even by itself; only the latter is unrunnable. A one-token
decode can still be expensive when it reads a long KV history, so selection may use `context_length_before` as a cost
estimate in addition to input-token count.

<a id="scheduler-complete"></a>
### `complete(batch, output) -> list[Event]`

Verifies that output corresponds to the in-flight batch, commits its KV writes, advances each request's prefill cursor
or decode progress, and changes a request from `PREFILL` to `DECODE` after its final prompt chunk. It checks stop-token
and `max_new_tokens` limits, queues `TokenEvent` objects for sampled non-stop tokens, and releases every finished
request's blocks and reservation before returning its `FinishedEvent`. Partial prefills produce no token event. The
returned events are handed to the service layer for streaming.

<a id="scheduler-fail"></a>
### `fail(batch, error) -> list[Event]`

Handles an exception from the model runner. It aborts uncommitted KV allocations, terminates the affected requests with
`reason="error"`, releases their resources, and returns their completion events. This is an exceptional recovery path,
not normal preemption.

## KV cache manager API used by the scheduler

```python
class KVCacheManager:
    def estimate_reservation(self, request: TokenizedRequest) -> int: ...
    def try_reserve(self, request_id: RequestId, blocks: int) -> CacheHandle | None: ...
    def prepare_batch(self, items: tuple[BatchItem, ...]) -> KVBatchLayout: ...
    def commit_batch(self, layout: KVBatchLayout) -> None: ...
    def abort_batch(self, layout: KVBatchLayout) -> None: ...
    def release(self, handle: CacheHandle) -> None: ...
```

<a id="kv-estimate-reservation"></a>
### `estimate_reservation(request) -> int`

Returns the maximum KV block charge for the request, including block rounding and copy-on-write requirements. V1 may
conservatively calculate from `len(prompt_token_ids) + max_new_tokens`, even though the final sampled token may remain
outside KV. The calculation must be safe when prefix sharing is absent; prefix-cache savings can be added later without
weakening the guarantee.

<a id="kv-try-reserve"></a>
### `try_reserve(request_id, blocks) -> CacheHandle | None`

Atomically reserves `blocks` for an admitted request. Returns an opaque handle on success or `None` when other running
requests currently hold too much capacity. A request that can never fit is rejected by `add_request`, rather than left
waiting forever.

<a id="kv-prepare-batch"></a>
### `prepare_batch(items) -> KVBatchLayout`

Allocates the physical slots needed for all new KV writes in the proposed batch and returns the corresponding block
tables, write-slot mappings, and context metadata. Preparation is atomic. With valid reservations and a plan within the
batch limits, it must succeed; a capacity failure indicates an accounting or planning error, not a routine signal to
shrink the batch.

<a id="kv-commit-batch"></a>
### `commit_batch(layout) -> None`

Marks a successfully executed batch's KV writes and sequence lengths as committed. Only a committed batch can supply
history to a later batch.

<a id="kv-abort-batch"></a>
### `abort_batch(layout) -> None`

Discards that batch's uncommitted allocations after runner failure and restores the last committed logical KV state.
Slots written during a failed run must not be exposed as valid history.

<a id="kv-release"></a>
### `release(handle) -> None`

Frees the request's physical KV blocks and remaining reservation. The scheduler calls it on completion, cancellation, or
failure.

## Model runner API used by the scheduler

```python
class ModelRunner:
    def run(self, batch: ReadyBatch) -> RunnerOutput: ...
```

`run` receives a mixed batch whose KV writes already have physical slots. It builds flat, variable-length input tensors,
executes the forward pass with paged attention, writes the new KVs to the supplied slots, and samples where
`sample_next_token=True`. It returns one `ItemOutput` for every `BatchItem`. It does not alter scheduler queues, decide
admission, or release a request's KV resources.

## One-iteration lifecycle

```python
scheduler.add_request(tokenized_request)  # service layer, once per request

result = scheduler.schedule()
service.publish(result.events)
if result.batch is not None:
    try:
        output = model_runner.run(result.batch)
    except Exception as error:
        events = scheduler.fail(result.batch, error)
    else:
        events = scheduler.complete(result.batch, output)
    service.publish(events)
```

`complete` releases finished requests before `service.publish` runs. The next `schedule` call can therefore admit
waiting requests using the newly freed KV capacity.

## Service-facing asynchronous API

The service-facing mental model is a long-lived scheduler that accepts requests, owns the scheduling loop, and
delivers token and finish events back to the service. The service does not call `schedule`, `complete`, or `fail`
directly. Those operations remain internal scheduler steps. This section describes the intended boundary; exact
transport and shutdown semantics remain open design choices.

```python
class Scheduler:
    async def submit(self, request: TokenizedRequest) -> RequestStream: ...
    async def cancel(self, request_id: RequestId) -> None: ...
    async def run(self) -> None: ...

class ModelRunner:
    async def run(self, batch: ReadyBatch) -> RunnerOutput: ...

async def serve_request(scheduler: Scheduler, request: TokenizedRequest) -> None:
    events = await scheduler.submit(request)
    async for event in events:
        await service.publish(event)
```

`submit` makes the request available to the scheduler and returns a per-request stream. The scheduler's run loop admits
waiting requests, creates batches, awaits the model runner, commits or aborts KV work, and publishes resulting events
to the matching streams. A stream yields `TokenEvent`s in that request's token order and ends with exactly one
`FinishedEvent`. A scheduler failure or shutdown must also terminate open streams with a defined outcome.

Cancellation must be safe while the scheduler is planning or awaiting the runner. A request in an in-flight batch is
marked for cancellation and cleaned up after the runner returns; its stream then receives one cancellation finish
event. The implementation must define whether the runner call is awaited serially or can overlap with other work;
the current invariant remains one model batch in flight at a time.

This public async API is separate from the internal scheduling functions shown above. The current draft still needs to
reconcile their signatures and define `RequestStream`, stream backpressure, duplicate IDs, scheduler startup and
shutdown, and errors delivered after submission.
