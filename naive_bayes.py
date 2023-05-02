import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.prior_probs = None
        self.conditional_probs = None

    def calculate_prior(self, y):
        classes = np.unique(y)
        prior_probs = {}
        for label in classes:
            prior_probs[label] = np.sum(y == label) / len(y)
        return prior_probs

    def calculate_conditional_probabilities(self, X, y):
        features = X.columns
        classes = np.unique(y)
        conditional_probs = {}
        for feature in features:
            conditional_probs[feature] = {}
            for label in classes:
                label_rows = X[y == label]
                feature_counts = label_rows[feature].value_counts()
                feature_probs = {}
                for index, value in feature_counts.items():
                    feature_probs[index] = value / np.sum(y == label)
                conditional_probs[feature][label] = feature_probs
        return conditional_probs

    def calculate_conditional_probabilities_smoothing(self, X, y, k = 1):
        features = X.columns
        classes = np.unique(y)
        conditional_probs = {}
        for feature in features:
            conditional_probs[feature] = {}
            for label in classes:
                label_rows = X[y == label]
                feature_counts = label_rows[feature].value_counts()
                feature_probs = {}
                for index, value in feature_counts.items():
                    feature_probs[index] = (value + k) / (np.sum(y == label) + k*len(feature_counts))
                conditional_probs[feature][label] = feature_probs
        return conditional_probs

    def predict_class(self, x_new):
        posterior_probs = {}
        for label in self.prior_probs:
            posterior_probs[label] = self.prior_probs[label]
            for feature in x_new.index:
                value = x_new[feature]
                feature_probs = self.conditional_probs[feature][label]
                if value in feature_probs:
                    posterior_probs[label] *= feature_probs[value]
                else:
                    posterior_probs[label] *= (1 - sum(feature_probs.values()))
        return max(posterior_probs, key=posterior_probs.get)

    def fit(self, X_train, y_train, l):
        self.prior_probs = self.calculate_prior(y_train)
        if(l == 0):
            self.conditional_probs = self.calculate_conditional_probabilities(X_train, y_train)
        elif(l == 1):
            self.conditional_probs = self.calculate_conditional_probabilities_smoothing(X_train,y_train,l)
    def predict(self, X_test):
        y_preds = []
        for i in range(len(X_test)):
            x_new = X_test.iloc[i]
            y_pred = self.predict_class(x_new)
            y_preds.append(y_pred)
        return y_preds
    
    def convert(self,y_give):
        for i in range(len(y_give)):
            if(y_give[i] == ' >50K'):
                y_give[i] = 1
            else:
                y_give[i] = 0
                
        return y_give

    def accuracy(self, y_true, y_pred):
        count = 0
        for i in range(len(y_true)):
            if (y_true[i] == y_pred[i]):
                count += 1
        return count / len(y_true)
    
    def precision(self, y_true, y_pred):
        
            true_positives = 0
            false_positives = 0
            for i in range(len(y_true)):
                if y_pred[i] == 1 and y_true[i] == 1:
                    true_positives += 1
                elif y_pred[i] == 1 and y_true[i] != 1:
                    false_positives += 1
            if true_positives + false_positives == 0:
                return 0
            else:
                precision = true_positives / (true_positives + false_positives)
            return precision 

    def recall(self, y_true, y_pred):
        true_positives = 0
        false_negatives = 0
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                true_positives += 1
            elif y_pred[i] == 0 and y_true[i] == 1:
                false_negatives += 1
        if true_positives + false_negatives == 0:
            return 0
        return true_positives / (true_positives + false_negatives)