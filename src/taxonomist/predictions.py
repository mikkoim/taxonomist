import pandas as pd
from scipy.special import softmax
import numpy as np

class TaxonomistPredictions():
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.fnames = None

    def set_y_true(self, y_true):
        self.y_true = y_true
    
    def set_y_pred(self, y_pred):
        self.y_pred = y_pred
    
    def set_fnames(self, fnames):
        self.fnames = fnames
    
    def set_logits(self, logits):
        self.logits = logits
        self.n_classes = logits.shape[1]
    
    def set_class_map(self, class_map):
        self.class_map = class_map
        n_in_class_map = len(class_map["fwd_dict"])
        if n_in_class_map != self.n_classes:
            self.classes = class_map["inv"](list(range(self.n_classes)))
        else:
            self.classes = range(self.n_classes)
    
    def get_y_true_y_pred(self):
        df = pd.DataFrame({"y_true": self.y_true,
                             "y_pred": self.y_pred})
        df.index = self.fnames
        df.index.name = "fname"
        return df
    
    def get_softmax(self):
        return softmax(self.logits, axis=1)
    
    def get_full_df(self, softmax=False):
        df_pred = self.get_y_true_y_pred()
        if softmax:
            df_prob = pd.DataFrame(self.get_softmax(), columns=self.classes)
        else:
            df_prob = pd.DataFrame(self.logits, columns=self.classes)

        df = pd.concat((df_pred.reset_index(drop=True),
                        df_prob.reset_index(drop=True)), axis=1)
        df.index = self.fnames
        df.index.name = "fname"
        return df

