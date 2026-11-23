#predict.py

import joblib
import pandas as pd
from sklearn.dataset import load_breast_cancer
#STPES 1: load the saved model
def load_model(model_path="best_model.joblib"):
  model = joblib.load(model_path)
  return model
#STEP 2: Create a reusable prediction function
def make_prediction(model.input_data):
   prediction = model.predict(input_data)
   probabilities = model.predict(input_data)
   result = "Malignant" if prediction[[0] == 0 else "Benign"
   return {
      "prediction": result,
      "probabailities": {
          "malignant": probabilities[0][0],
          "benign": probabilities[0][1]
      }
   }
#STPE 3: Run a sample predicton when excuted directly
if __name__=="__main__":
  midel = load_model()
  data = load_breast_cancer()
  feature_names = data.feature_names
  sample_input = dict(zip(feature_names, data.data[0]))
  #Modify a few values for demonsstration
  sample_input["mean radius"] = 15.0
  sample_input["mean texture"] = 18.0
  sample_input["mean perimeter"] = 100.0
  input_df = pd.DataFrame(
    [sample_input],
    columns=feature_names
  )
  output = make_prediction(model,input_df)
  print("Model loaded successfullly!")
  print("\n Prediction:",output["prediction"])
  print("\n Prediction Probabilities:")
  print("Benign:", output["probabilities"]["malignant"])
  print("Benign:", output["probabilties"]["benign"])
  
