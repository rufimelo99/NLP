import sys
sys.append("..")

from src.exception import CustomException
import logging
import sys

if __name__=="__main__":
    try :
        a=1/0
    except Exception as e:
        logging.info("Hello World")
        raise CustomException(e, sys)