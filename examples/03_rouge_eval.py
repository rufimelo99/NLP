import logging
import sys
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
sys.append("..")
from src.exception import CustomException
from src.utils.rouge.evaluation import Rouge_Evaluator

def main():
    try:
        df = pd.read_csv('example_df.csv', encoding='utf-8')
        rouge_evaluator = Rouge_Evaluator(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        scores = rouge_evaluator.get_scores_from_dataframe(df, 'Generated_Summary', 'Sumário')
        mean_scores = rouge_evaluator.get_mean_scores(scores)
        print(mean_scores)

    except Exception as e:
        logging.info("Problem with rouge evaluation")
        raise CustomException(e, sys)

if __name__=="__main__":
    main()
