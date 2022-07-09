MAP_EXTENSION = ".wad"
CONFIG_EXTENSION = ".cfg"

### EPISODES INFO
TIMESTEPS_PER_EPISODE = 1500
PRETRAINING_MAP_PATH = "wads/curiosity_training"
PRETRAINING_EPISODES = 200

### MAP PATHS
TESTING_MAP_PATH_DENSE = "wads/my_way_home_dense"
TESTING_MAP_PATH_SPARSE = "wads/my_way_home_sparse"
TESTING_MAP_PATH_VERY_SPARSE = "wads/my_way_home_verySparse"
TRAINING_EPISODES = 3000
TESTING_EPISODES = 10

### MODEL SAVES
STATS_UPDATE_FREQUENCY = 10
WEIGHTS_SAVE_FREQUENCY = 5

### PREPROCESSING
SKIP_FRAMES = STACK_FRAMES = 4
PREPROCESSING_SIZE = (42,42)      # Preprocessed images will be 42x42
PREPROCESSING_SIZE_RND = (84,84)  # Preprocessed images for RND will be 84x84

### ICM PATHS
ICM_WEIGHTS_PATH = 'models/ICM/icm_{}_{}.ckpt'
REINFORCE_WEIGHTS_PATH = 'models/reinforce/reinforce_{}_{}.ckpt'
REINFORCE_WEIGHTS_PATH_NO_ICM = 'models/reinforce_no_icm/reinforce_no_icm_{}_{}.ckpt'
# ### RND PATHS
# RND_WEIGHTS_PATH = 'models/RND/rnd_{}_{}.ckpt'
# ACTOR_CRITIC_WEIGHTS_PATH_RND = 'models/ac_rnd/ac_{}_{}.ckpt'
# ACTOR_CRITIC_WEIGHTS_PATH_NO_RND = 'models/ac_no_rnd/ac_no_rnd_{}_{}.ckpt'
### LOGS
LOGS_DIR = 'models/logs/{}_{}'

CLIP_NO = 40.0       # Gradient norm clipping
### ICM PARAMS
ETA     = 0.1        # Scaling factor for the intrinsic reward signal
BETA    = 0.8        # Weight of the forward model loss against the inverse model loss
GAMMA   = 0.99       # Discount for rewards
SIGMA   = 0.01       # Entropy coefficient
LAMBDA  = 0.1        # Actor-critic loss coefficient
CLIP_RE = 0.1        # Clip the intrinsic reward
ICM_LW  = 10.0       # Weight of the ICM loss with respect to the actor's loss
#############