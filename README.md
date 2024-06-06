# Strategically Conservative Q Learning

Author's Pytorch implementation of **SCQ**.

## Install

1. Install Mujoco binaries
    - Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
    - Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
    - Set the environmental variable

      ```bash
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
      ```

2. Create Anaconda environment

    ```bash
    conda create -n scq python=3.8.5 pip
    conda activate scq
    ```

    We use `python=3.8.5` to do the experiments in the paper. But we believe `python=3.9` and `python=3.10` should also work as long as it is compatible with d4rl.

3. Install the dependencies

    ```python
    pip install -r requirements.txt
    ```

    For reference, authors use the following versions:

    - gym==0.18.3
    - d4rl==1.1
    - cython==0.29.37
    - torch==2.1.2
    - torchrl==0.2.1
    - numpy==1.24.4
    - wandb==0.16.4
    - tqdm==4.66.1
    - pyyaml==6.0.1
    - argparse==1.1


## How to reproduce experiments

All the necessary parameters to reproduce the result can be found in the `config` folder. You can just run each of the experiment by using the following command.

Ex1. HalfCheetah-medium-v2

```python
python main.py --config config/halfcheetah/halfcheetah-medium-v2.yaml --seed 0
```

Ex2. Antmaze-medium-play-v2

```python
python main.py --config config/antmaze/antmaze-medium-play-v2.yaml --seed 0
```

We use [Weights and Biases](https://wandb.ai/site) to visualize all of the logging data and results. In order to access these information, you need to set up your own W&B API keys at the first experiment.

## Credits

Some of our code come from the following repository. We appreciate these authors to share their valuable codes.

- CQL [CQL](https://github.com/aviralkumar2907/CQL).
- TD3-BC [TD3-BC](https://github.com/sfujim/TD3_BC)
- MCQ [MCQ](https://github.com/dmksjfl/MCQ)
- SAC-RND [SAC-RND](https://github.com/tinkoff-ai/sac-rnd/tree/main)

If you have any questions, suggestions and improvements, please feel free to send me message to purewater0901\[at\]berkeley\[dot\]edu
