import datetime
import os

from logs.Logger import Logger
import time

# 生成当前的时间戳
log_date = datetime.datetime.now().strftime("%Y%m%d")
log_time = datetime.datetime.now().strftime("%H%M%S")

# 创建当日文件夹的日期的文件夹f"../logs/logfiles/{date}"
if not os.path.exists(f"./logs/logfiles/{log_date}"):
    os.makedirs(f"./logs/logfiles/{log_date}")

# 创建日志对象
logger = Logger(f"./logs/logfiles/{log_date}/{log_time}_Compared_VecSearch.log")

# 记录时间的操作
class Timer:
    def __init__(self, description="", dic_key=None):
        self.description = description
        self.dic_key = dic_key

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logger.info(f"{self.description} completed in {self.interval:.5f} seconds.")

        if self.dic_key: # 存储时间
            logger.dict_log[self.dic_key] = self.interval
