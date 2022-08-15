- [1. What we are doing](#1-what-we-are-doing)
- [1.1. The problems:](#11-the-problems)
  - [What we are concerning about](#what-we-are-concerning-about)
  - [What we are doing here](#what-we-are-doing-here)
- [2. Share development env by VS code remote container](#2-share-development-env-by-vs-code-remote-container)
- [3. Steps to train the model](#3-steps-to-train-the-model)
- [4. Steps to test the model](#4-steps-to-test-the-model)
- [5. The result](#5-the-result)
  - [a. Model include in the work - No parameters search](#a-model-include-in-the-work---no-parameters-search)
  - [b. Model optimized with Optuna](#b-model-optimized-with-optuna)
  - [b. Model train with distillation loss](#b-model-train-with-distillation-loss)
- [6. Other Development setups](#6-other-development-setups)
  - [Poetry usage](#poetry-usage)
- [7. Link to trained model + resource](#7-link-to-trained-model--resource)


### 1. What we are doing

### 1.1. The problems:
- Key word spotting in audio.
- Dataset : Speech Command

#### What we are concerning about
- High accuracy on the test set
- Super small model size for edge deployment

#### What we are doing here
- We will use model distillation to pass knowledge from a big model to a small one
- We will use Optuna for parameters search
- We will use Torch lightling as boilerplace for this project
- We will use weight and bias as monitoring tool

### 2. Share development env by VS code remote container

Please read though some concept [here](https://code.visualistudio.com/docs/remote/containers-tutorial)

This will spin up the development environment with minimal setup.

1. Install and configure [git password manager](https://github.com/GitCredentialManager/git-credential-manager#linux) - this will help to share git configuration to the container

2. Run the "Remote-Containers: Reopen in Container" command


### 3. Steps to train the model

1. Train simple convolution
```shell
    python train.py
```

2. Train Bc ResNet model
```shell
    python train.py --model bc_resnet
```

### 4. Steps to test the model

1. Train simple convolution
```shell
    python test.py --pretrain path_to_pretrain
```

2. Train Bc ResNet model
```shell
    python test.py --model bc_resnet --pretrain path_to_pretrain
```


### 5. The result

#### a. Model include in the work - No parameters search
| Model      | Description |  Params-byte size | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 26900 - 108kb | 94.2% |
| BC Resnet   | Experiment logging        | 10600 - 42 kb | 95.6% |  |

#### b. Model optimized with Optuna
| Model      | Description |  Params-byte size | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 35000 - 140kb | 95.1% | |
| BC Resnet   | Experiment logging        |  |  | |

#### b. Model train with distillation loss
| Model      | Description |  Params-byte size | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 28600 - 114kb | 90.3% | |
| BC Resnet   | Experiment logging        |  |  | |

### 6. Other Development setups

#### [Poetry usage](https://python-poetry.org/docs/basic-usage/)

1. Install depenedencies
```shell
    pip install poetry
    poetry install
```

2. Add new dependencies
```shell
    poetry add package_name
```


### 7. Link to trained model + resource

