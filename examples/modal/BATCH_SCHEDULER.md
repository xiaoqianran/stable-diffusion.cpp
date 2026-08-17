# Batch GPU scheduler

The Web workbench separates **job scheduling** from **container fan-out**.

## Batch parallelism

Each batch job selects one of:

- `1` — strict serial execution; one GPU input at a time.
- `2` — at most two image inputs in flight.
- `4` — at most four image inputs in flight.

Other Web jobs remain behind the local GPU job scheduler while the active batch
owns the GPU stage. This prevents two independent batch jobs from multiplying
their requested parallelism together.

The Web recipe pool uses a stable `max_containers=4` Modal Cls variant. The
selected `parallelism` controls how many `.remote()` calls are actually in
flight. This lets serial / 2 / 4 jobs reuse the same recipe/GPU container pool.

## Model residency

Bundled Web recipes instantiate `SDEngine(recipe=...)`. The recipe is a Modal
class parameter, so each recipe has its own autoscaling pool. During
`@modal.enter`, the container starts `sd-server` with model paths from the
`sdcpp-models` Volume. The server process stays alive for the container lifetime
and handles subsequent prompts over its local HTTP API.

The generic CLI path still uses `sd-cli` so arbitrary CLI flags keep their old
behavior.

## Same-model affinity

After CPU/Volume staging, jobs enter the local GPU queue with an affinity key:

```text
<GPU>::<recipe>
```

When a recipe just ran, the scheduler may pull another matching recipe forward
from a bounded window. This raises the chance that Modal routes the next request
to an already-warm model pool. To avoid starvation, affinity is capped to a
short streak before FIFO order wins again.

## Scaling knobs

```text
SDCPP_GPU_JOB_MAX_ACTIVE=1   # independent Web jobs owning GPU stage at once
SDCPP_WEB_GPU_POOL_MAX=4     # max containers in one recipe/GPU pool
SDCPP_IDLE_SECONDS=10        # idle scale-down window
```

The UI exposes only per-job parallelism `1 / 2 / 4`. Increasing parallelism can
reduce wall-clock time but can also start additional billed GPU containers and
load another copy of the model in each container.
