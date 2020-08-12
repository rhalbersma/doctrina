# Experiments in Reinforcement Learning

[![Language](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)
[![Standard](https://img.shields.io/badge/Python-3.6-blue.svg)](https://en.wikipedia.org/wiki/History_of_Python)
[![License](https://img.shields.io/badge/license-Boost-blue.svg)](https://opensource.org/licenses/BSL-1.0)
[![Lines of Code](https://tokei.rs/b1/github/rhalbersma/doctrina?category=code)](https://github.com/rhalbersma/doctrina)

## Requirements

- Python version 3.6 or higher

## Installation

```bash
git clone https://github.com/rhalbersma/doctrina.git
cd doctrina
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -e .
```

## Examples

### Solving Blackjack using Value Iteration

```python
import gym
import pandas as pd

import gym_blackjack_v1 as bj
from doctrina import spaces
from doctrina.algorithms import dp

# Dynamic programming using Markov Chains or How to play Blackjack
# by Craig L. Zirbel (2001)
# https://www.dropbox.com/s/xrntclqyx36jhis/Blackjack_talk_2001.pdf

# The payoff (p. 3).
env = gym.make('Blackjack-v1', winning_blackjack=+1.5, model=True)

# Value iteration (p. 12)
V, policy, delta, iter = dp.V_value_iter(env)
```

```python
# The value matrix (p. 13).
pd.DataFrame(V.reshape(spaces.shape(env.state_space))[1:len(bj.Hand),:len(bj.Card)], index=bj.hand_labels[1:], columns=bj.card_labels).round(4)
```

```
           2       3       4       5       6       7       8       9       T       A
H2   -0.0759 -0.0498 -0.0221  0.0137  0.0389 -0.0273 -0.1032 -0.1900 -0.3003 -0.4485
H3   -0.1005 -0.0689 -0.0363  0.0002  0.0245 -0.0574 -0.1309 -0.2151 -0.3218 -0.4655
H4   -0.1149 -0.0826 -0.0494 -0.0124  0.0111 -0.0883 -0.1593 -0.2407 -0.3439 -0.4829
H5   -0.1282 -0.0953 -0.0615 -0.0240 -0.0012 -0.1194 -0.1881 -0.2666 -0.3662 -0.5006
H6   -0.1408 -0.1073 -0.0729 -0.0349 -0.0130 -0.1519 -0.2172 -0.2926 -0.3887 -0.5183
H7   -0.1092 -0.0766 -0.0430 -0.0073  0.0292 -0.0688 -0.2106 -0.2854 -0.3714 -0.5224
H8   -0.0218  0.0080  0.0388  0.0708  0.1150  0.0822 -0.0599 -0.2102 -0.3071 -0.4441
H9    0.0744  0.1013  0.1290  0.1580  0.1960  0.1719  0.0984 -0.0522 -0.2181 -0.3532
H10   0.1825  0.2061  0.2305  0.2563  0.2878  0.2569  0.1980  0.1165 -0.0536 -0.2513
H11   0.2384  0.2603  0.2830  0.3073  0.3337  0.2921  0.2300  0.1583  0.0334 -0.2087
H12  -0.2534 -0.2337 -0.2111 -0.1672 -0.1537 -0.2128 -0.2716 -0.3400 -0.4287 -0.5504
H13  -0.2928 -0.2523 -0.2111 -0.1672 -0.1537 -0.2691 -0.3236 -0.3872 -0.4695 -0.5825
H14  -0.2928 -0.2523 -0.2111 -0.1672 -0.1537 -0.3213 -0.3719 -0.4309 -0.5074 -0.6123
H15  -0.2928 -0.2523 -0.2111 -0.1672 -0.1537 -0.3698 -0.4168 -0.4716 -0.5425 -0.6400
H16  -0.2928 -0.2523 -0.2111 -0.1672 -0.1537 -0.4148 -0.4584 -0.5093 -0.5752 -0.6657
H17  -0.1530 -0.1172 -0.0806 -0.0449  0.0117 -0.1068 -0.3820 -0.4232 -0.4644 -0.6386
H18   0.1217  0.1483  0.1759  0.1996  0.2834  0.3996  0.1060 -0.1832 -0.2415 -0.3771
H19   0.3863  0.4044  0.4232  0.4395  0.4960  0.6160  0.5939  0.2876 -0.0187 -0.1155
H20   0.6400  0.6503  0.6610  0.6704  0.7040  0.7732  0.7918  0.7584  0.4350  0.1461
H21   0.8820  0.8853  0.8888  0.8918  0.9028  0.9259  0.9306  0.9392  0.8117  0.3307
T     0.2300  0.2534  0.2775  0.3030  0.3337  0.3011  0.2418  0.1597 -0.0095 -0.1969
A     0.5598  0.5768  0.5944  0.6129  0.6396  0.6340  0.5759  0.4940  0.3431  0.1168
S12   0.0818  0.1035  0.1266  0.1565  0.1860  0.1655  0.0951  0.0001 -0.1415 -0.3219
S13   0.0466  0.0741  0.1025  0.1334  0.1617  0.1224  0.0541 -0.0377 -0.1737 -0.3474
S14   0.0224  0.0508  0.0801  0.1119  0.1392  0.0795  0.0133 -0.0752 -0.2057 -0.3727
S15  -0.0001  0.0292  0.0593  0.0920  0.1182  0.0370 -0.0271 -0.1122 -0.2373 -0.3977
S16  -0.0210  0.0091  0.0400  0.0734  0.0988 -0.0049 -0.0668 -0.1486 -0.2684 -0.4224
S17  -0.0005  0.0290  0.0593  0.0912  0.1281  0.0538 -0.0729 -0.1498 -0.2586 -0.4320
S18   0.1217  0.1483  0.1759  0.1996  0.2834  0.3996  0.1060 -0.1007 -0.2097 -0.3720
S19   0.3863  0.4044  0.4232  0.4395  0.4960  0.6160  0.5939  0.2876 -0.0187 -0.1155
S20   0.6400  0.6503  0.6610  0.6704  0.7040  0.7732  0.7918  0.7584  0.4350  0.1461
S21   0.8820  0.8853  0.8888  0.8918  0.9028  0.9259  0.9306  0.9392  0.8117  0.3307
BJ    1.5000  1.5000  1.5000  1.5000  1.5000  1.5000  1.5000  1.5000  1.3846  1.0385
```

```python
# The optimal stopping strategy (p.14).
pd.DataFrame(policy.reshape(spaces.shape(env.state_space))[1:len(bj.Hand),:len(bj.Card)], index=bj.hand_labels[1:], columns=bj.card_labels).replace({0:' ', 1:'H'})
```

```
     2  3  4  5  6  7  8  9  T  A
H2   H  H  H  H  H  H  H  H  H  H
H3   H  H  H  H  H  H  H  H  H  H
H4   H  H  H  H  H  H  H  H  H  H
H5   H  H  H  H  H  H  H  H  H  H
H6   H  H  H  H  H  H  H  H  H  H
H7   H  H  H  H  H  H  H  H  H  H
H8   H  H  H  H  H  H  H  H  H  H
H9   H  H  H  H  H  H  H  H  H  H
H10  H  H  H  H  H  H  H  H  H  H
H11  H  H  H  H  H  H  H  H  H  H
H12  H  H           H  H  H  H  H
H13                 H  H  H  H  H
H14                 H  H  H  H  H
H15                 H  H  H  H  H
H16                 H  H  H  H  H
H17
H18
H19
H20
H21
T    H  H  H  H  H  H  H  H  H  H
A    H  H  H  H  H  H  H  H  H  H
S12  H  H  H  H  H  H  H  H  H  H
S13  H  H  H  H  H  H  H  H  H  H
S14  H  H  H  H  H  H  H  H  H  H
S15  H  H  H  H  H  H  H  H  H  H
S16  H  H  H  H  H  H  H  H  H  H
S17  H  H  H  H  H  H  H  H  H  H
S18                       H  H  H
S19
S20
S21
BJ
```

## License

Copyright Rein Halbersma 2020.  
Distributed under the [Boost Software License, Version 1.0](http://www.boost.org/users/license.html).  
(See accompanying file LICENSE_1_0.txt or copy at [http://www.boost.org/LICENSE_1_0.txt](http://www.boost.org/LICENSE_1_0.txt))
