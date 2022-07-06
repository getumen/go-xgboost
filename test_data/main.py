from sklearn import datasets
from sklearn import model_selection
import xgboost as xgb
import pandas as pd

breast_cancer = datasets.load_breast_cancer()

feature = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
target = pd.Series(breast_cancer.target)

train_x, test_x, train_y, test_y = model_selection.train_test_split(
    feature, target, test_size=0.2, shuffle=True
)

dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(test_x)

booster = xgb.train(
    {
        "max_depth": 2,
        "eta": 1,
        "objective": "binary:logistic",
    },
    dtrain,
    100,
)

booster.save_model("xgboost.model")

test_x.to_csv("feature.csv", index=False, header=False)
test_y.to_csv("label.csv", index=False)

scores = booster.predict(dvalid)
with open("score.csv", "w") as f:
    for x in scores:
        print(x, file=f)
