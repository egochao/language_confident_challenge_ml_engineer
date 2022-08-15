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
  - [c. Model train with distillation loss](#c-model-train-with-distillation-loss)
  - [d. Hightlight point](#d-hightlight-point)
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
| Model      | Description |  Params | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 26900 | 94.2% |
| BC Resnet   | Experiment logging        | 10600 | 95.6% |  |

#### b. Model optimized with Optuna
| Model      | Description |  Params | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 35000 | 95.1% | |
| BC Resnet   | Experiment logging        | 22000 | 98.3% - best | |

#### c. Model train with distillation loss
| Model      | Description |  Params | Model accuracy | |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| Simple Convolution      | A straight forward 1D convolution    | 28600 | 90.3% | |
| BC Resnet   | Experiment logging        |  |  | |


#### d. Hightlight point
- My best model have 22k parameters and accuracy on test set = 98.3% (Optuna optimized)
- Almost beat the state-of-art(98.5)
- The model size is superior compare with all other state-of-art model by some order of magnitude
- The distillation process is not success and it causing the model perform worst than non distill

![image](https://github.com/egochao/speech_commands_distillation_torch_lightling/blob/main/my_best_result.png)

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

- [Logit data](https://drive.google.com/file/d/1VhFxooFVE6Ph4V2QSM4PMyMPNW9iZg9g/view?usp=sharing)
- [Best simple convolution](https://drive.google.com/file/d/1UUiVjE6VYYbvXDA6bBnv4zCKe89vHLs1/view?usp=sharing) : 95.1% [link to wandb report](https://wandb.ai/johnoldman/ViT_experiments/reports/Simple-convolution-with-optuna-parameters--VmlldzoyNDc0Mjgy?accessToken=kl36zi4301sy0b880d3q20m7x8u5fty3mgxj6i7w5a1xxu40pum2lhci80fbp7yt)
- [Best Bc ResNet](https://drive.google.com/file/d/1yg8Aag0k_DMn4X25vaI7Vb8R32DuWBw7/view?usp=sharing) : 98.3% [link to wandb report](https://wandb.ai/johnoldman/ViT_experiments/reports/Best-bc-resnet---VmlldzoyNDc0NjAw?accessToken=0gusdixg3zt5aigffk9ueeca5gz3qsnbf7ofri1ex36jc1pfasx2ot31tm34743m)

