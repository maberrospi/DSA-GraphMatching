import os
from dotenv import load_dotenv

load_dotenv()

IMG_DIR_PATH = os.getenv("IMG_DIR_PATH")
IMG_MIN_DIR_PATH = os.getenv("IMG_MIN_DIR_PATH")
IMG_SEQ_DIR_PATH = os.getenv("IMG_SEQ_DIR_PATH")
ANNOT_DIR_PATH = os.getenv("ANNOT_DIR_PATH")
