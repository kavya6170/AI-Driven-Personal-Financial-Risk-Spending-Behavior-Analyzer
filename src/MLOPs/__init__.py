import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs" # create the log file
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath), # this will store action in log file
        logging.StreamHandler(sys.stdout) # this will print the logged data on the terminal
    ]
)

logger = logging.getLogger("MLOPsLogger")