## 1. What we are doing

### 1.1. The problems:
- Key word spotting in audio.
- Dataset : Speech Command

### What we are concerning about
- High accuracy on the test set
- Super small model size for edge deployment

### What we are doing here
- We will use model distillation to pass knowledge from a big model to a small one
- We will use Optuna for parameters search
- We will use Torch lightling as boilerplace for this project
- We will use weight and bias as monitoring tool

## 2. Share development env by VS code remote container

Please read though some concept [here](https://code.visualistudio.com/docs/remote/containers-tutorial)

This will spin up the development environment with minimal setup.

1. Install and configure [git password manager](https://github.com/GitCredentialManager/git-credential-manager#linux) - this will help to share git configuration to the container

2. Run the "Remote-Containers: Reopen in Container" command


## 3. Steps to train the model

1. Train simple convolution
```shell
    python train.py
```

2. Train Bc ResNet model
```shell
    python train.py --model bc_resnet
```

## 4. Steps to test the model

1. Train simple convolution
```shell
    python test.py --pretrain path_to_pretrain
```

2. Train Bc ResNet model
```shell
    python train.py --model bc_resnet --pretrain path_to_pretrain
```


NOTE: You can run other training script with same args

## 4. Other Development setups

### 4.1. [Poetry usage](https://python-poetry.org/docs/basic-usage/)

1. Install depenedencies
```shell
    pip install poetry
    poetry install
```

2. Add new dependencies
```shell
    poetry add package_name
```