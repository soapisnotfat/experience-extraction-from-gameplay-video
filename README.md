# Player Experience Extraction from Gameplay Video

Official code release for AIIDE 2018 oral presentation paper **[Player Experience Extraction from Gameplay Video]()**, authored by Zijin Luo, Matthew Guzdial, Nicholas Liao and Mark Riedl.

## Table of Contents

- [Getting Started](#getting-started)
- [License](#license)
- [Contributing](#contributing)

## Getting Started

This is the repo for the skyrim dataset experiment.

For the Gwario experiment, please checkout the gwario branch

### Prerequisites

- PyTorch ~0.4.0
- torchvision ~0.3.0
- numpy
- PIL
- tqdm

### Installation

```sh
git clone https://github.com/IvoryCandy/experience-extraction-from-gameplay-video.git
```

### Structure

```
- data
  - dataloader.py
  - download.sh
- model
  - model.py
  - download.sh
- main.py
- misc.py
- solver.py
```

`main.py` has an argument setting, you can run set your custom hyper-parameters here

`solver.py` has three solvers -- Backprop, ImageNet, and TeacherStudent. Each solver contains all reletive functions for loading data, training model, and tesing performance. Use them based on your purposes. 

### Usage

to download data

```sh
cd data
bash download.sh
```

to download pretrained model

```sh
cd models
bash download.sh
```

run under default config

```sh
python3 main.py
```

For customized experiments, select different solvers from `solver.py` and put it in `main.py`, and set your own hyoer-parameters in arguments.

## License

Distributed under the Apache2 license. See ``LICENSE`` for more information.

## Contributing

- [Fork it](https://github.com/IvoryCandy/experience-extraction-from-gameplay-video/fork)
