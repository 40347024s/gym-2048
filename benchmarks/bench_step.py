import time
import gymnasium as gym
import gym2048


def main():
    env = gym.make("Gym2048-v0")
    env.reset(seed=123)

    actions = [0, 1, 2, 3]
    steps = 200_000

    start = time.perf_counter()
    for i in range(steps):
        action = actions[i & 3]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    end = time.perf_counter()

    dt = end - start
    sps = steps / dt
    print(f"steps: {steps}")
    print(f"time: {dt:.4f}s")
    print(f"steps/sec: {sps:,.0f}")


if __name__ == "__main__":
    main()
