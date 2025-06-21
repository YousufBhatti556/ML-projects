import sys
from mlproject.logger import logging  # use the centralized logger

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_msg = error_message_details(error_msg, error_detail)

    def __str__(self):
        return self.error_msg

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info(CustomException(e, sys))
        # logging.error(CustomException(e, sys))  # logs the nicely formatted error
        raise CustomException(e, sys)  # if you want the program to crash/show traceback in terminal
