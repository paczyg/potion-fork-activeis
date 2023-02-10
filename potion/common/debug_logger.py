import logging
import sys
import functools

def setup_debug_logger(name, log_file, level=None, stream = None):
    # Helper to setup a logger based on the logging facility for Python:
    # See: https://docs.python.org/3/library/logging.html

    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Write on file
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optionally write also on stderr/stdout
    if 'stderr' == stream:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if 'stdout' == stream:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
