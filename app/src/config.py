import os

ROOT_DIR = os.getcwd()

while not os.path.isfile(os.path.join(ROOT_DIR, "README.md")):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

TRAINING_FILE = ROOT_DIR + "/app/input/train.csv"
TEST_FILE = ROOT_DIR + "/app/input/test.csv"
OUTPUT_PATH = ROOT_DIR + "/app/models/model.bin"
VOCAB_PATH = ROOT_DIR + "/app/word2index.pkl"
METRICS_PATH = ROOT_DIR + "/app/metrics.json"

BATCH_SIZE = 4
EMB_DIM = 100
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 1 
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

