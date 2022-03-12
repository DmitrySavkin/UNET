import os
from datetime import date
from Download_utils import download_file_tqdm
from utils_folder import config

config = config.CocoConfig()

print(config.LEARNING_RATE)
# os.system(f"zip -rm experiments_rate_{config.LEARNING_RATE}_{date.today()}.zip experiments")
# os.system("rm expreriments")
os.system("python scripts/main/train.py")
