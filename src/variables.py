### EPISODES INFO
MAX_TIMESTEPS_PER_EPISODE = 2100
TRAINING_EPISODES = 1000
TESTING_EPISODES = 10

ENV_NAME = "VizdoomHealthGatheringSupreme-v0"

### MODEL SAVES
WEIGHTS_SAVE_FREQUENCY = 5

### PREPROCESSING
SKIP_FRAMES = STACK_FRAMES = 4
PREPROCESSING_SIZE = (84,84)      # Preprocessed images will be 84X84 black and white images

### WEIGHT PATHS
DQN_WEIGHTS_PATH = 'models/simpler/DQN/dqn.ckpt'
REINFORCE_WEIGHTS_PATH = 'models/simpler/reinforce/reinforce.ckpt'
ACTOR_CRITIC_WEIGHTS_PATH = 'models/simpler/actor_critic/actor_critic.ckpt'

### LOGS
LOGS_DIR = 'models/simpler/logs/'

### PARAMS
CLIP_NO = 40.0       # Gradient norm clipping
LR      = 1e-4       # Learning rate
GAMMA   = 0.99       # Discount for rewards
SIGMA   = 0.01       # Entropy coefficient
EPS_D   = 0.999      # Epsilon decay for each time step
EPS_S   = 0.99       # Starting epsilon value
EPS_MIN = 0.01       # Minimum epsilon value
### DQN
MAX_DQN_EXP_BUFFER_LEN = 5000   # Size of the experience buffer for DQN
DQN_BATCH_SIZE = 32  # Batch size for DQN experience sampling
DQN_START_UPDATES_EPISODE = 10   # In DQN updates start from episode 10.
### AC
MAX_AC_EXP_BUFFER_LEN = MAX_TIMESTEPS_PER_EPISODE
BATCH_SIZE_A2C = 32