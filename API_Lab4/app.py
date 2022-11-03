from fastapi import FastAPI, Query
from pydantic import BaseModel
import json
import numpy as np 
import pandas as pd
from joblib import load

class DataModel(BaseModel):
# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    serial_no: float | list[float] = Query(default=[])
    gre_score: float | list[float] = Query(default=[])
    toefl_score: float | list[float] = Query(default=[])
    university_rating: float | list[float] = Query(default=[])
    sop: float | list[float] = Query(default=[])
    lor: float | list[float] = Query(default=[])
    cgpa: float | list[float] = Query(default=[])
    research: float | list[float] = Query(default=[])

    def columns(self):
        return ["Serial No.","GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]

class DataModel1(BaseModel):

    # Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    serial_no: list | float
    gre_score: list | float
    toefl_score: list | float
    university_rating: list | float
    sop: list | float
    lor: list | float
    cgpa: list | float
    research: list | float
    admission_points: list | float

    # Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["Serial No.", "GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research", "Admission Points"]

if __name__ == "__main__":

   def square(X):
    for feature in X.columns:
        if feature=='CGPA' or feature=='GRE Score':
            X[feature]=np.sqrt(X[feature])
    return X

    def dropNa(X):
        return X.dropna()

app = FastAPI()

@app.get('/')
def read_root():
    return{"welcome":"Welcome to my API"}


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    try:
        df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
        df.columns = dataModel.columns()
        model = load("assets/modelo.joblib")
    
        result = model.predict(df)
        lists = result.tolist()
        json_str = json.dumps(lists)
        #TODO: reescalar a valores reales de puntajes
        return json_str
    except Exception as err:
        if str(err).startswith("columns are missing"):
            return "Falta alguna de las columnas: 'CGPA', 'University Rating', 'Research"
        else:
            return(err)

@app.post("/score")
def make_predictions(dataModel: DataModel1): ##Probar con DataModel y si no, cambiar por DataModel1
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    x = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    try:
        result = model.score(x, y)
        lists = result
        json_str = json.dumps(lists)
        if json_str=="NaN":
            return "Ingrese más valores"
        else:
            return json_str
    except Exception as err:
        return(err)