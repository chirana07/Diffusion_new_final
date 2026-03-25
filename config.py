import torch
import os

class Config:
    # Robust dataset root detection
    KAGGLE_INPUT = "/kaggle/input"
    LOCAL_INPUT = "./input"
    DATASET_ROOT = KAGGLE_INPUT if os.path.exists(KAGGLE_INPUT) else LOCAL_INPUT

    SAVE_DIR = "./checkpoints"
    RESULT_DIR = "./results"
    TEST_SAMPLES_DIR = "./test_samples"

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(TEST_SAMPLES_DIR, exist_ok=True)

    IMG_SIZE = 128
    BATCH_SIZE = 4
    EPOCHS = 300

    LR_START = 1e-4
    LR_MIN = 1e-6
    WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 1e-4
    BETAS = (0.9, 0.999)
    GRAD_CLIP = 1.0
    TIMESTEPS = 100
    INFERENCE_STEPS = 20
    CHANNELS = 32
    CHANNEL_MULT = [1, 2, 4, 8]
    RES_BLOCKS = 1

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"