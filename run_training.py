import os
from datetime import date, datetime
from Download_utils import download_file_tqdm
from utils_folder import config

config = config.CocoConfig()

print(config.LEARNING_RATE)
os.system(f"cp -rf experiments/ experiments_rate_good_0.001_{date.today()}_{datetime.now()}")
os.system("rm expreriments")
os.system(f"zip -rm models25__{date.today()}_{datetime.now()}.zip models25")
os.system("mkdir models26")
os.system("python scripts/main/train.py")
#os.system("rm -rf models")

