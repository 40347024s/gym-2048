# gym-2048 Usage
Use gym to create the environment of the 2048 game and accelerate it through numba.
## Installation
```
pip install -e /path/to/gym-2048
```
## Environments
|Env Name|Observation space|State Values|
|-|-|-|
|Game2048Env|4x4|2^1 to 2^20|
|NormGame2048Env_ver0|1x4x4|0 to 1|
|NormGame2048Env_ver1|20x4x4|0 to 1|

- ### Game2048Env
  - **Observation space**: 4x4
  - **State values**: 0 and 2^i, i from 1 to 20.
    - 0 for empty.
- ### NormGame2048Env_ver0
  - **Observation space**: 1x4x4
    - board shape: (C, W, H)
  - **State values**: 0~1
    - 0 for empty.
    - The power of two divided by the largest power, currently set to 20.
      - e.g. 2 -> 1/20
      - e.g. 2048 -> 11/20
- ### NormGame2048Env_ver1
  - **Observation space**: 20x4x4
    - board shape: (C, W, H)
      - C=0, this channel is for all grid positions with no value.
      - C=i, channel is for all grid positions with 2^i.
  - **State values**: 0~1
    - Same as NormGame2048Env_ver0


