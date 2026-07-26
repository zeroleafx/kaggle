import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

train = pd.read_csv("titanicComp/titanic/train.csv")
test = pd.read_csv("titanicComp/titanic/test.csv")

TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "the Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dona": "Rare", "Dr": "Rare", "Major": "Rare",
    "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare",
}

def add_features(df):
    df = df.copy()
    df["Title"] = (df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
                             .str.strip().replace(TITLE_MAP))
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"]   = df["Cabin"].notna().astype(int)
    return df

train_f, test_f = add_features(train), add_features(test)

age_median  = train_f.groupby("Title")["Age"].median()
age_global  = train_f["Age"].median()
fare_median = train_f["Fare"].median()

for d in (train_f, test_f):
    d["Age"]      = d["Age"].fillna(d["Title"].map(age_median)).fillna(age_global)
    d["Fare"]     = d["Fare"].fillna(fare_median)
    d["FareLog"]  = np.log1p(d["Fare"])
    d["Embarked"] = d["Embarked"].fillna("S")

FEATURES = ["Pclass", "Sex", "Age", "FareLog", "Title",
            "FamilySize", "IsAlone", "HasCabin", "Embarked"]

combined = pd.concat([train_f[FEATURES], test_f[FEATURES]], keys=["tr", "te"])
combined = pd.get_dummies(combined, columns=["Sex", "Title", "Embarked"])

X_all      = combined.loc["tr"].values.astype(np.float32)
X_test_np  = combined.loc["te"].values.astype(np.float32)
y_all      = train["Survived"].values.astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

scaler = StandardScaler()
X_tr      = scaler.fit_transform(X_tr)
X_val     = scaler.transform(X_val)
X_test_np = scaler.transform(X_test_np)


class LRTitanic(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)
    
RANDOM_SEED = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'
t = lambda a: torch.tensor(a, dtype=torch.float32).to(device)

X_tr, X_val, X_test = t(X_tr), t(X_val), t(X_test_np)
y_tr  = t(y_tr).unsqueeze(1)
y_val = t(y_val).unsqueeze(1)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

model_0 = LRTitanic(n_features=X_tr.shape[1]).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

accuracy_fn = Accuracy(task="binary").to(device)

epochs = 2000
for epoch in range(epochs):
    model_0.train()
    logits = model_0(X_tr)
    loss   = loss_fn(logits, y_tr)
    acc    = accuracy_fn(torch.sigmoid(logits), y_tr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model_0.eval()
        with torch.inference_mode():
            val_logits = model_0(X_val)
            val_loss   = loss_fn(val_logits, y_val)
            val_acc    = accuracy_fn(torch.sigmoid(val_logits), y_val)
        print(f"Epoch {epoch:4d} | loss {loss:.4f} acc {acc:.4f} "
              f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

model_0.eval()
with torch.inference_mode():
    test_preds = torch.round(torch.sigmoid(model_0(X_test)))

pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds.int().squeeze().cpu().numpy()
}).to_csv("titanicComp/titanicSubmission.csv", index=False)
