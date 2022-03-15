import os
from datetime import date
from Download_utils import download_file_tqdm
from utils_folder import config

config = config.CocoConfig()

print(config.LEARNING_RATE)
os.system(f"cp -rf experiments/ experiments_rate_0.001_{date.today()}")
# os.system("rm expreriments")
os.system(f"zip -rm models23_rate_0.001_{date.today()}.zip models23")
os.system("python scripts/main/train.py")
