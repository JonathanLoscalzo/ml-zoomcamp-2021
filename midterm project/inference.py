from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
from enum import Enum, IntEnum
import pickle
import pandas as pd

app = FastAPI()


class YesNo(str, Enum):
    yes = "Y"
    no = "N"


class MaleFemale(str, Enum):
    male = "M"
    female = "F"


class ChestPainType(str, Enum):
    TA = "TA"
    ATA = "ATA"
    NAP = "NAP"
    ASY = "ASY"


class FastingBS(IntEnum):
    yes = 1
    no = 0


class RestingECGType(str, Enum):
    Normal = "Normal"
    ST = "ST"
    LVH = "LVH"


class SlopeType(str, Enum):
    Up = "Up"
    Flat = "Flat"
    Down = "Down"


class InputModel(BaseModel):
    age: int
    sex: MaleFemale
    chest_pain_type: ChestPainType
    resting_bp: float
    cholesterol: float
    fasting_bs: FastingBS
    resting_ecg: RestingECGType
    max_heart_rate: int = Field(lt=202, gt=60)
    exercise_angina: YesNo
    old_peak: float
    st_slope: SlopeType

CATEGORICAL_COLUMNS = [
    "fasting_bs",
    "sex",
    "chest_pain_type",
    "resting_ecg",
    "exercise_angina",
    "st_slope",
]
NUMERICAL_COLUMNS = ["cholesterol", "age", "resting_bp", "max_heart_rate", "old_peak"]
TARGET_COLUMN = "heart_disease"

@app.post("/predict/", )
async def predict(inputData: InputModel):

    preprocessor = pickle.load(open('./data/models/preprocessor.bin', 'rb'))

    lr = pickle.load(open('./data/models/lr.bin', 'rb'))
    rf = pickle.load(open('./data/models/rf.bin', 'rb'))
    xgb = pickle.load(open('./data/models/xgb.bin', 'rb'))

    data = pd.DataFrame.from_dict([inputData.dict()])

    X = data[CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS]
    X = preprocessor.transform(X)

    return {
        "lr": lr.predict_proba(X)[0][1], 
        "rf": rf.predict_proba(X)[0][1], 
        "xgb": float(xgb.predict_proba(X)[0][1])
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
