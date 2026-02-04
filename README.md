# gym2048

Cython-accelerated 2048 environment for Gymnasium.

## Install (editable)

```bash
pip install -e .
```

## Quick usage

```python
import gymnasium as gym
import gym2048

env = gym.make("Gym2048-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)
```

## Tests

```bash
pytest
```

## Benchmark

```bash
python benchmarks/bench_step.py
```
