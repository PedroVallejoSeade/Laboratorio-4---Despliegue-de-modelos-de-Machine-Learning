from fastapi import FastAPI
from pydantic import BaseModel
import json

import pandas as pd
from joblib import load

class DataModel(BaseModel):
# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    serial_no: float
    gre_score: float
    toefl_score: float
    university_rating: float
    sop: float
    lor: float 
    cgpa: float
    research: float

    def columns(self):
        return ["Serial No.","GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]

app = FastAPI()

@app.get('/')
def read_root():
    return{"welcome":"Welcome to my API"}


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    lists = result.tolist()
    json_str = json.dumps(lists)
    #TODO: reescalar a valores reales de puntajes
    return json_str

@app.post("/score")
def make_predictions(dataModel: DataModel): ##Probar con DataModel y si no, cambiar por DataModel1
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    x = df[df.columns[:-1]]
    y = df.columns[-1]
    result = model.score(x, y)
    print(result)
    return result