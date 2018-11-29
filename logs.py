import logging

# initialize logger
logger = logging.getLogger("alpha_zero")
logger.setLevel(logging.INFO)

# init log handler
logger_handler = logging.FileHandler('logs/last_run.log', mode='w')
logger_handler.setLevel(logging.INFO)

logger_formatter = logging.Formatter(
    fmt='%(asctime)s %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S'
)

logger_handler.setFormatter(logger_formatter)

logger.addHandler(logger_handler)
logger.info('Completed configuring logger()!')


def get_logger():
    return logger