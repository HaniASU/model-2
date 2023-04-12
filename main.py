import json
from tensorflowtensorflow.keras.models import load_model
from flask import Flask, request

loaded_model = load_model("model.h5",compile = False)
app = Flask(__name__)

def getLabel(file):
   pred = loaded_model.predict(file).round()
   out_col = ['output_CAD', 'output_CHF', 'output_MI', 
              'output_Normal']
   output = {out_col[i]:pred[0][i] for i in range(4)}
   output_class = max(output, key=output.get).split('_')[-1]
   return output_class 
  
@app.route("/file", methods=["POST"])
def create_page():
  print("Entered")
  if 'messageFile' in request.files:
     file = request.files['messageFile']
     label = getLabel(file)
     data = {'label':f'{label}'}
     json_data = json.dumps(data)
     return json_data
  data = {'response':'file not found'}
  json_data = json.dumps(data)
  return json_data

if __name__ == "__main__":
  app.run("0.0.0.0",debug=True,port = 8000)