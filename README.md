# Player Experience Extraction from Gameplay Video

Official code release for AIIDE 2018 oral presentation paper **[Player Experience Extraction from Gameplay Video]()**

## Table of Contents

- [Getting Started](#getting-started)
- [License](#license)
- [Contributing](#contributing)

## Getting Started

This is the repo for the `gwario` dataset experiment

for Skyrim dataset experiment, please checkout `master` branch

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
  - log_file
  - frames (release soon)
- main.py
- dataset.py
- misc.py
- model.py
```

All hyper-parameters are global varaibles in `main.py`.

### Usage

run

```sh
python3 main.py
```

## License

Distributed under the Apache2 license. See ``LICENSE`` for more information.

## Contributing

- [Fork it](https://github.com/IvoryCandy/experience-extraction-from-gameplay-video/fork)
a
