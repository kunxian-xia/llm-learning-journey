"""Static contract scenario for three concurrent generation requests.

This describes an expected legal schedule. It is not bound to a concrete
scheduler implementation or a required batch-selection order.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SamplingConfig:
    """Placeholder until the public sampling contract is defined."""

    seed: int = 0


@dataclass(frozen=True)
class TokenizedRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    stop_token_ids: frozenset[int]
    sampling: SamplingConfig


@dataclass(frozen=True)
class BatchItem:
    request_id: str
    phase: str
    input_ids: tuple[int, ...]
    context_length_before: int
    sample_next_token: bool


@dataclass(frozen=True)
class ReadyBatch:
    items: tuple[BatchItem, ...]


@dataclass(frozen=True)
class ScheduleResult:
    batch: ReadyBatch | None
    events: tuple[object, ...]


@dataclass(frozen=True)
class ItemOutput:
    request_id: str
    sampled_token_id: int | None


@dataclass(frozen=True)
class RunnerOutput:
    items: tuple[ItemOutput, ...]


class Scheduler(Protocol):
    def add_request(self, request: TokenizedRequest) -> None: ...
    def schedule(self) -> ScheduleResult: ...
    def complete(self, batch: ReadyBatch, output: RunnerOutput) -> list[object]: ...
    def fail(self, batch: ReadyBatch, error: Exception) -> list[object]: ...


class ModelRunner(Protocol):
    def run(self, batch: ReadyBatch) -> RunnerOutput: ...


def make_request(request_id: str, prompt_length: int,
                 output_length: int) -> TokenizedRequest:
    return TokenizedRequest(
        request_id=request_id,
        prompt_token_ids=tuple(range(prompt_length)),
        max_new_tokens=output_length,
        stop_token_ids=frozenset(),
        sampling=SamplingConfig(seed=0),
    )


def make_requests() -> tuple[TokenizedRequest, ...]:
    return (
        make_request("long", prompt_length=12, output_length=8),
        make_request("medium", prompt_length=6, output_length=2),
        make_request("short", prompt_length=2, output_length=2),
    )


@dataclass(frozen=True)
class ExpectedStep:
    iteration: int
    work: tuple[str, ...]
    sampled: tuple[tuple[str, int], ...] = ()
    finished: tuple[str, ...] = ()


def expected_trace() -> tuple[ExpectedStep, ...]:
    """Illustrative legal trace; exact batch selection is not part of the API."""
    return (
        ExpectedStep(1, ("long:prefill:0-4", "medium:prefill:0-4")),
        ExpectedStep(2, ("short:prefill:0-2", "long:prefill:4-8",
                         "medium:prefill:4-6"),
                     (("short", 1000), ("medium", 1000))),
        ExpectedStep(3, ("long:prefill:8-12", "short:decode:1000",
                         "medium:decode:1000"),
                     (("long", 1000), ("short", 1001), ("medium", 1001)),
                     ("short", "medium")),
        ExpectedStep(4, ("long:decode:1000",), (("long", 1001),)),
        ExpectedStep(5, ("long:decode:1001",), (("long", 1002),)),
        ExpectedStep(6, ("long:decode:1002",), (("long", 1003),)),
        ExpectedStep(7, ("long:decode:1003",), (("long", 1004),)),
        ExpectedStep(8, ("long:decode:1004",), (("long", 1005),)),
        ExpectedStep(9, ("long:decode:1005",), (("long", 1006),)),
        ExpectedStep(10, ("long:decode:1006",), (("long", 1007),),
                     ("long",)),
    )


def run_contract_scenario(scheduler: Scheduler, runner: ModelRunner) -> None:
    """Drive the synchronous internal scheduler methods described in the draft."""
    for request in make_requests():
        scheduler.add_request(request)

    observed: list[object] = []
    while True:
        result = scheduler.schedule()
        observed.extend(result.events)
        if result.batch is None:
            break

        batch = result.batch
        try:
            output = runner.run(batch)
        except Exception as error:
            observed.extend(scheduler.fail(batch, error))
        else:
            observed.extend(scheduler.complete(batch, output))

    assert observed
