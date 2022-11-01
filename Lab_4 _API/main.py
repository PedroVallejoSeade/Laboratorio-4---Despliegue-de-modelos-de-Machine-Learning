from typing import Optional

import pandas as pd
from joblib import load

from fastapi import FastAPI

import DataModel, DataModel1

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    #TODO: reescalar a valores reales de puntajes
    return result


@app.post("/score")
def make_predictions(dataModel: DataModel): ##Probar con DataModel y si no, cambiar por DataModel1
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/modelo.joblib")
    x = df[df.columns[:-1]]
    y = df.columns[-1]
    result = model.score(x, y)
    return result
