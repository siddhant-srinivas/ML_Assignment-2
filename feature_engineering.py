import pandas as pd
class FeatureEngineering:
    def __init__(self, ds):
        self.ds = ds

    def task1(self):
        columns = self.ds.columns
        for i in range(len(columns)):
            average = self.ds[columns[i]].mode()
            self.ds[columns[i]].fillna(average, inplace = True)
