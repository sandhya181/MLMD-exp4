#step 1:Import libraries
import pandas as pd
import joblib
from sklearn.dataset import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipelilne import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.liner_model import logisticRegression
from sklearn.neighbors import KNeighborsClassifier
from skklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
  f1_score,
  accuracy_score,
  classification_report,
  confusion_matrix
)
#step 2: Load dataset 
data = load_breast_cancer()
X = pd.DataFrame( 
    data.data,
    columns=data.feature_names
)
y = pd.Series(data.target,name="target")
print("Data shape:",X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\nTarget values:")
print(y.value_counts())
#step 3: Split dataset
X_train,X_test,y_train,y_test =train_test_split(
  X,
  y,
  test_size=0.20,
  random_state=42,
  stratify=y
)
print("\n Training sample:", X_train.shape[0])
print("Testing sample:" X_test.shape[0])
#stpe 4:Define classifiction algorithms
model = {
  "Logistic Regression": LogisticRegression(
    max_iter=5000
  ),
  "KNN": KNeighborClassifier(),
  "Decision Tree":DecisionTreeClassifier(
    random_state=42
  ),
  "SVM": SVC()
}
#step 5: Define hyparameter grids
param_grids = {
  "Logistic Regression": {
    "model__C":[0.1,1,10],
    "model__solver":["liblinear"]
  },
  "KNN":{
    "model__n_neighbors":[3,5,7,9],
    "model__weights":["uniform","distance"]
  },
  "Decision Tree":{
    "model__max_depth":[None,3,5,10],
    "model__min_sample_split":[2,5,20]
  },
  "Random Forest":{
    "model__n_estimators":[100,200],
    "model__max_depth":[None,5,10],
    "model__min_samplpes_split":[2,5]
  }
  "SVM":{
  "model__C":[0.1,1,10],
  "model__kernel":["linear","rbf"],
  "model__gamma":["scale","auto"]
}
}
#stpe 6: Run Grid SearchCV for every algorithm
result =[]
best_model = {}
for model_name, model in models.itmes():
  print("\n" + "=" *60)
  print("Runnning:",model_name)
  print("=" * 60)
  pipeline =Pipeline([
  ("scaler",StandardScaler()),
  ("model",model)
  ])
  grid_search = GridSearchCV(
  estimator=pipeline,
  param_gird=param_grids[model_name],
  scoring="f1",
  cv=5,
  n_jobs=-1
  )
  gird_search.fit(X_train,y_train)
  best_models[model_name] = gird_search,best_estimator_result.append({
  "Model",model_name,
  "Best F1 Score";grid_search,best_score_,
  "Best Parameters"grid_search,best_params_,
  })
  print("Best F1 Score:"
        round(grid_seach.best_score_,4))
  print("Best Parameters:",
       grid_search.best_params_)
  #step 7:Compare all models
  result_df = pd.DtaFrame(results)
  result_df = result_df.sort_values(
  by="Best F1 SCore",
  ascending=False
  )
  print("\n" + "="*60)
  print("MODLE COMPARISION")
  print("=" * 60)
  print ( 
  result_df[
  ["Model",Best F1 Scire"]
  ].to_string(index=False)
             )
  #step 8: Select the best model 
  best_model_name =result_df.iloc[0]["Model"]
  best_model =best_models[best_model_name]
  best_cv_f1 = result_df.iloc[0]["Best F1 score"]
            
  print("\nBest model based on CV F1-score:")
  print(best_model_name)
  print(round(best_cv_f1,4))
  #step 9: Evaluate on unseen test data
  y_pred = best_model.predict(X_test)
  print("\n" + "=" * 60)
  print("FINAL MODEL EVALUATION")
  print("=" * 60)
  print("Accuracy:",accuracy_score(y_test,y_pred))
  print(F1 Score:", f1_score(y_test,y_pred))
  print("\n Classifiction Report:")
  prinnt( 
      classification_report(
  y_test,
  y_pred,
  target_names=data.target_names
      )
  )
  print("\n Confusion Matrix:")
  print(confusion_matrix(y_test,y_pred))
  #stpe 10: Save the best model
  joblib.dump(best_model,"best_model.joblib")
  print ("\n Best model saved successfully!")
  print)"Model:',best-model_name)
  print
