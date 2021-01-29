import os
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    input_path = ""
    df = pd.read_csv(os.path.join(input_path, "train.csv"))
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)