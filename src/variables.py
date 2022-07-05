MAP_EXTENSION = ".wad"
CONFIG_EXTENSION = ".cfg"

TIMESTEPS_PER_EPISODE = 2100

PRETRAINING_MAP_PATH = "wads/curiosity_training"
PRETRAINING_EPISODES = 200

TESTING_MAP_PATH_DENSE = "wads/my_way_home_dense"
TESTING_MAP_PATH_SPARSE = "wads/my_way_home_sparse"
TESTING_MAP_PATH_VERY_SPARSE = "wads/my_way_home_verySparse"
TRAINING_EPISODES = 3000
TESTING_EPISODES = 10

STATS_UPDATE_FREQUENCY = 10
WEIGHTS_SAVE_FREQUENCY = 5

SKIP_FRAMES = STACK_FRAMES = 4
PREPROCESSING_SIZE = (42,42)      # Preprocessed images will be 42x42

ICM_WEIGHTS_PATH = 'models/ICM/icm.ckpt'
ACTOR_CRITIC_WEIGHTS_PATH = 'models/ac/ac.ckpt'
ACTOR_CRITIC_WEIGHTS_PATH_NO_ICM = 'models/ac_no_icm/ac_no_icm.ckpt'
LOGS_DIR = 'models/logs/'

ETA     = 0.1        # Scaling factor for the intrinsic reward signal
BETA    = 0.8        # Weight of the forward model loss against the inverse model loss
GAMMA   = 0.9        # Discount for rewards
SIGMA   = 0.05       # Entropy coefficient
LAMBDA  = 0.1        # Actor-critic loss coefficient
CLIP_RE = 0.1        # Clip the intrinsic reward
CLIP_NO = 40.0       # Gradient norm clipping
ICM_LW  = 10.0       # Weight of the ICM loss with respect to the actor's loss