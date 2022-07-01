MAP_EXTENSION = ".wad"
CONFIG_EXTENSION = ".cfg"

TIMESTEPS_PER_EPISODE = 2100

PRETRAINING_MAP_PATH = "wads/curiosity_training"
PRETRAINING_EPISODES = 1000

TESTING_MAP_PATH_DENSE = "wads/my_way_home_dense"
TESTING_MAP_PATH_SPARSE = "wads/my_way_home_sparse"
TESTING_MAP_PATH_VERY_SPARSE = "wads/my_way_home_verySparse"
TRAINING_EPISODES = 10_000

SKIP_FRAMES = STACK_FRAMES = 4
PREPROCESSING_SIZE = (42,42)      # Preprocessed images will be 42x42