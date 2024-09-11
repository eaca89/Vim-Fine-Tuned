#!/bin/bash
# conda activate Vim
# cd /home/eh_abdol/Vim/vim

# Define paths and parameters
DATA_PATH="/home/eh_abdol/fine_tune/gold3"
TRAIN_FOLDER="${DATA_PATH}/train"
VALIDATION_FOLDER="${DATA_PATH}/validation"
TEST_FOLDER="${DATA_PATH}/test"
NUM_EPOCHS=30
BATCH_SIZE=16
LEARNING_RATE=5e-6

# Activate your Python environment if needed
# source /path/to/your/venv/bin/activate

# Run the Python script
python fine-tune.py \
    --train-folder ${TRAIN_FOLDER} \
    --validation-folder ${VALIDATION_FOLDER} \
    --test-folder ${TEST_FOLDER} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE}
