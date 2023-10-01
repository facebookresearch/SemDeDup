# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import os
import numpy as np
import random


def seed_everything(seed: int = 42):
    """
    Function to set seed for random number generators for reproducibility.

    Args:
        seed: The seed value to use for random number generators. Default is 42.

    Returns:
        None
    """
    # Set seed values for various random number generators
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CUDA algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def get_logger(
#     file_name="logger.log", level=logging.INFO, stdout=False
# ) -> logging.Logger:
#     """
#     When the level is set to "logging.INFO", the debugging logs will not be saved (lower level).
#     """
#     # See https://www.loggly.com/ultimate-guide/python-logging-basics/ for more information about pyhton logging module
#     logger = logging.getLogger()  # uses the module name
#     # set log level
#     logger.setLevel(level)
#     logger.handlers = []
#     # define file handler and set formatter
#     file_handler = logging.FileHandler(
#         file_name
#     )  # or use logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", file_name))
#     # define formatter
#     formatter = logging.Formatter(
#         "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
#     )  # or use logging.BASIC_FORMAT
#     file_handler.setFormatter(formatter)

#     stdout_handler = (
#         logging.StreamHandler()
#     )  # .setLevel(logging.DEBUG) #.setFormatter(CustomFormatter(fmt))

#     # add handler to logger
#     # if not logger.hasHandlers():
#     logger.addHandler(file_handler)
#     if stdout:
#         logger.addHandler(stdout_handler)

#     return logger


#     import logging


def get_logger(
    file_name: str = "logger.log", level: int = logging.INFO, stdout: bool = False
) -> logging.Logger:
    """
    Initialize and configure the logger object to save log entries to a file and optionally print to stdout.

    :param file_name: The name of the log file.
    :param level: The logging level to use (default: INFO).
    :param stdout: Whether to enable printing log entries to stdout (default: False).
    :return: A configured logging.Logger instance.
    """
    logger = logging.getLogger(__name__)

    # Set the logging level
    logger.setLevel(level)

    # Remove any existing handlers from the logger
    logger.handlers = []

    # Create a file handler for the logger
    file_handler = logging.FileHandler(file_name)

    # Define the formatter for the log entries
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Optionally add a stdout handler to the logger
    if stdout:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    # Return the configured logger instance
    return logger
