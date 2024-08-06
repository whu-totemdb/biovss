"""
用于激励实验过程中的日志记录的类.

代码写清楚中文注释，方便自己和他人阅读。
"""
import logging
import sys
import json
from collections import defaultdict
import numpy as np

class Logger:
    """
    This class is used for logging messages. It uses the built-in logging module in Python.

    Attributes:
        log_file_path (str): The path to the log file.
        logger (Logger): The logger object from the logging module.
        formatter (Formatter): The formatter object to format the log messages.
        file_handler (FileHandler): The file handler object to handle the writing of log messages to the file.
    """

    def __init__(self, log_file_path):
        """
        The constructor for the Logging class.
        下面使用logger, formatter, file_handler三个对象来记录日志，其中formatter用于格式化日志消息，file_handler用于将日志消息写入文件。
        使用logging包的基本思路：
        1. 获取logger对象，传入参数有name，如果不传入参数，则默认为root
        2. 设置日志级别为INFO，只有INFO级别及以上的日志才会被记录
        3. 设置日志格式，这里设置的格式为'%(asctime)s - %(levelname)s - %(message)s'
        4. 创建FileHandler对象，传入参数为日志文件的路径
        5. 设置FileHandler对象的格式为formatter
        6. 将FileHandler对象添加到logger对象中
        7. 使用logger对象的info、error、warning等方法记录日志

        Parameters:
            log_file_path (str): The path to the log file.
        """
        self.log_file_path = log_file_path
        self.logger = logging.getLogger() # 获取logger对象，传入参数有name，如果不传入参数，则默认为root
        self.logger.setLevel(logging.INFO) # 设置日志级别为INFO，只有INFO级别及以上的日志才会被记录
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                           datefmt='%m-%d %H:%M:%S')

        # 创建FileHandler对象，传入参数为日志文件的路径
        self.file_handler = logging.FileHandler(self.log_file_path) # 创建FileHandler对象，传入参数为日志文件的路径
        self.file_handler.setFormatter(self.formatter)
        # 创建StreamHandler对象，传入参数为sys.stdout
        stream_handler = logging.StreamHandler(sys.stdout) # 创建StreamHandler对象，传入参数为sys.stdout。
        stream_handler.setFormatter(self.formatter)

        # 将FileHandler对象和StreamHandler添加到logger对象中
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(stream_handler)

        # 存储日志变量: 用于存储实验的中间结果
        self.dict_log = defaultdict(list)


    def info(self, msg):
        """
        Logs an info level message.

        Parameters:
            msg (str): The message to be logged.
        """
        self.logger.info(msg)

    def error(self, msg):
        """
        Logs an error level message.

        Parameters:
            msg (str): The message to be logged.
        """
        self.logger.error(msg)

    def warning(self, msg):
        """
        Logs a warning level message.

        Parameters:
            msg (str): The message to be logged.
        """
        self.logger.error(msg)

    def dict_info(self):
        """
        Logs the dictionary log.
        """
        for key, value in self.dict_log.items():
            self.logger.info(f"{key}: {value}")

        # 同时保存到jsonl文件中，方便后续使用
        self.dic_jsonl()

    def dic_jsonl(self):
        """
        将self.dict_log中的所有数据转换为JSON Lines (JSONL) 格式并追加到文件中。
        """
        # 以追加模式打开修改后缀的文件
        with open(self.log_file_path[:-3] + "jsonl", 'a') as file:
            # 将整个字典转换为JSON字符串，使用自定义处理函数以支持int64类型
            json_str = json.dumps(self.dict_log, default = lambda o: int(o) if isinstance(o, np.int64) else o)
            # 写入文件，整个字典作为一行
            file.write(json_str + '\n')

    def dict_clear(self):
        """
        Clear the dictionary log.
        """
        self.dict_log.clear()






if __name__ == '__main__':
    # 测试Logging类
    log = Logging('./logfiles/test.log')
    log.info('This is an info message.')
