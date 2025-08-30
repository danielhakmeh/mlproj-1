import sys
import logging
from typing import Any

def error_message_detail(error: Exception, error_detail: Any) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"
    else:
        error_message = f"Error occurred: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: Any):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divided by zero error")
        raise CustomException(e, sys)