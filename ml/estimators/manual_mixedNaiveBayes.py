import numpy as np


class MixedNaiveBayes:
    def __init__(self, *singleBayeses):
        self.singleBayeses = list(singleBayeses)

    def __str__(self):
        return f"MixedNaiveBayes({self.singleBayeses})"

    def predict_proba(self, X):
        eff_proba_predictions = self.singleBayeses[0].predict_proba(X[0])
        for i in range(1, len(self.singleBayeses)):
            eff_proba_predictions *= self.singleBayeses[i].predict_proba(X[i]) / (self.singleBayeses[i].class_count_ / np.sum(self.singleBayeses[i].class_count_))  # divide by prior (each classifier was computed using [multiplying by] prior - here I have to take that into account -> only one prior-multiplications stays)
        eff_proba_predictions /= np.vstack(np.sum(eff_proba_predictions, axis=1))  # normalization
        return eff_proba_predictions

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        bool_compare = y==y_pred
        return np.sum(bool_compare) / len(bool_compare)