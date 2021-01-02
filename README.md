# Exercises in Reinforcement Learning

[![Language](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)
[![Standard](https://img.shields.io/badge/Python-3.6-blue.svg)](https://en.wikipedia.org/wiki/History_of_Python)
[![License](https://img.shields.io/badge/license-Boost-blue.svg)](https://opensource.org/licenses/BSL-1.0)
[![Lines of Code](https://tokei.rs/b1/github/rhalbersma/doctrina?category=code)](https://github.com/rhalbersma/doctrina)

## Overview

This repository consists of several notebooks with my attempts at the (programming) exercises from the first few chapters of the book [Reinforcement Learning, an Introduction](http://incompleteideas.net/book/RLbook2020.pdf), second edition, by Richard S. Sutton and Andrew G. Barto (2018). The plan is to work through the book up to and including at least Chapter 6 (Temporal-Difference Learning), reproducing all the figures and completing all the exercises. The various environments have been implemented in Python using the [OpenAI Gym toolkit](https://gym.openai.com/).

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

## License

Copyright Rein Halbersma 2020-2021.  
Distributed under the [Boost Software License, Version 1.0](http://www.boost.org/users/license.html).  
(See accompanying file LICENSE_1_0.txt or copy at [http://www.boost.org/LICENSE_1_0.txt](http://www.boost.org/LICENSE_1_0.txt))
