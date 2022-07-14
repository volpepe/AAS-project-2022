# AAS-project-2022
Project for the Adaptive and Autonomous Systems master course at Alma Mater Studiorum - University of Bologna.

## Requirements

- Python3
- ViZDoom and its Gym wrapper (`pip install vizdoom vizdoom[gym]`). 
    - For installation on MacOS or Linux systems please follow [this guide](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md). There are some additional dependencies and libraries to install.
- TensorFlow 2
- OpenAI Gym
- NumPy
- tqdm

## Description

In this project, we implemented two deep reinforcement learning algorithms from scratch and let them learn how to solve a navigation and survival task in an environment based on the FPS game Doom (1993). A full analysis of the problem and our solutions can be found in the [report](report.pdf).

## Running the code

### Test
The [models](./models) folder already has trained weights and full logs of our experiments. To let the agent play the game using these trained weights you can run:

```python
python src/run_test.py -l -a <AGENT>
``` 
where `<AGENT>` is one between `a2c` (for the A2C agent) `dqn` (for the DQN agent) or `random` (default, for the random agent).

### Train
To train the models you can use 

```python
python src/run_training.py (-l) (-s) -a <AGENT> (-st <EPISODE_NUM>)
``` 

- The `-l` option asks the script to load the weights in the [models](models) folder. Without specifying it, the training will start from random weights.
- The `-s` option asks the script to save the weights and logs (by default once every 5 episodes) in the [models](models) folder. **Any pretrained weight in the respective [models](models) folder will be overwritten!**
- The `-st` option allows to specify an initial episode number for the training (default is 0).

### Plot
The notebook [imgs/create_graphs.ipynb](imgs/create_graphs.ipynb) can be used to plot the agents' average maximum reward as a function of the played frames. More information can be found in the [report](report.pdf).
