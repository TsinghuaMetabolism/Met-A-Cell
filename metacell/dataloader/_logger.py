import logging



class InMemoryHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []  # 日志信息存储在列表中

    def emit(self, record):
        log_entry = self.format(record)
        self.log_messages.append(log_entry)  # 将日志记录添加到列表中

# 初始化日志处理器并配置格式
def setup_logger():
    logger = logging.getLogger('in_memory_logger')
    logger.setLevel(logging.DEBUG)

    memory_handler = InMemoryHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    memory_handler.setFormatter(formatter)

    # 添加自定义的内存Handler
    logger.addHandler(memory_handler)

    return logger, memory_handler