#Import the necessary libraries
from rouge_score import rouge_scorer
import numpy as np

class Rouge_Evaluator:
    def __init__(self, rouge_types, use_stemmer=False):
        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=use_stemmer)

    def score(self, prediction, reference):
        scores = self.scorer.score(prediction, reference)
        return scores

    def get_scores_from_dataframe(self, df, prediction_col, reference_col):
        scores = []
        for _, row in df.iterrows():
            scores.append(self.scorer.score(row[prediction_col], row[reference_col]))
        return scores

    def get_mean_scores(self, scores):   
        rouge_types=self.rouge_types
        mean_scores = {}
        for rouge_type in rouge_types:
            mean_scores[rouge_type] = {}
            mean_scores[rouge_type]['f1-score'] = np.mean([score[rouge_type].fmeasure for score in scores])
            mean_scores[rouge_type]['precision'] = np.mean([score[rouge_type].precision for score in scores])
            mean_scores[rouge_type]['recall'] = np.mean([score[rouge_type].recall for score in scores])
        return mean_scores
