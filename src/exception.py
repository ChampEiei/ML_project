import sys
import logging
def error_message_details(error,error_detail:sys):
    _,_,exc_tb =error_detail.exc_info()
    file_name =exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occur python script name [{0}] line [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message


class Custom_exception(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        1/0
    except Exception as e:
        logging.info("cant divide by zero")
        raise Custom_exception(e,sys)