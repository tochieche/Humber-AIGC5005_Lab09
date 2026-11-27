import pandas as pd
from sklearn.model_selection import train_test_split

from tickets_data import make_tickets

LABELS = ["account", "bike_fault", "billing", "station"]


def split_data(seed=42):
    # Keep row numbers so we can check that no row is shared.
    data = pd.DataFrame(make_tickets())
    train, rest = train_test_split(
        data, train_size=0.6, stratify=data.label, random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest.label, random_state=seed
    )
    return train, val, test


def small_sample(train, n, seed):
    if n == len(train):
        return train.copy()
    sample, _ = train_test_split(
        train, train_size=n, stratify=train.label, random_state=seed
    )
    return sample


if __name__ == "__main__":
    data, _, _ = split_data()
    print(data.label.value_counts())
    print(data.groupby("label").head(12).to_string())