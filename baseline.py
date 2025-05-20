from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path(__file__).parent / "cora"  
N_FOLDS  = 10


def load_cora():
    df = pd.read_csv(DATA_DIR / "cora.content", sep="\t", header=None)
    paper_ids = df.iloc[:, 0].astype(int).to_numpy()
    y_str     = df.iloc[:, -1].to_numpy()
    X_bin     = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)

    # TF-IDF  (use .toarray() – '.A' is deprecated)  :contentReference[oaicite:1]{index=1}
    X = TfidfTransformer().fit_transform(X_bin).toarray()
    return X, y_str, paper_ids


def main():
    warnings.filterwarnings("ignore")
    X, y, ids = load_cora()

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    y_pred = np.empty_like(y, dtype=object)

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=500, n_jobs=-1, C=4.0)
        clf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])

    acc = accuracy_score(y, y_pred)
    pd.DataFrame(
        {"paper_id": ids, "class_label": y_pred}
    ).to_csv("predictions.tsv", sep="\t", header=False, index=False)
    print(f"10-fold accuracy = {acc:.2%}")


if __name__ == "__main__":
    main()
