NUM_POLICIES = 1
ACTION_DIM = 1
# TRAIN_STRATEGY = ‘Random’
TRAIN_STRATEGY = 'Divergence'
# OUTPUT_FILE = ‘cartpole_ddqn_s0_0.out’
OUTPUT_FILE = 'cartpole_pg_s1_0.out'
NUM_EPISODE = 3000
MEMORY_CAPACITY = 100000
START_LEARN_STEP = 200
BATCH_SIZE = 32
BIDIRECTION = False

# PG
PG_LEARNING_RATE = 0.001
PG_DECAY_RATE = 0.99

RENDER = False # rendering wastes time
GAMMA = 0.9 # reward discount in TD error
LR_A = 0.0002 # learning rate for actor
LR_C = 0.01 # learning rate for critic
TAU = 0.01 # soft replacement