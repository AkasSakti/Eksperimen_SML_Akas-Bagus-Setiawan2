# modelling_tuning.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("Eksperimen_SML_Akas-Bagus-Setiawan2/preprocessing/olshopdatapreprocesed/online_shoppers_intention_preprocessed.csv")
X = df.drop("Revenue", axis=1)
y = df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, None]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid=params, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
