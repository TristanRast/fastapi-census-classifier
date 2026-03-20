"""
FastAPI application for ML model inference.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import inference


# Define the input data schema
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status",
                                example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country",
                                example="United-States")

    class Config:
        populate_by_name = True


# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for income prediction using census data",
    version="1.0.0"
)

# Load model and encoder at startup
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("model/lb.pkl", "rb") as f:
    lb = pickle.load(f)

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@app.get("/")
async def root():
    """
    Welcome message for the API.
    """
    return {"greeting": "Welcome to the ML Model API!"}


@app.post("/predict")
async def predict(data: CensusData):
    """
    Perform model inference on input census data.

    Parameters
    ----------
    data : CensusData
        Input features for prediction

    Returns
    -------
    dict
        Prediction result (<=50K or >50K)
    """
    # Convert input to dataframe
    input_dict = {
        'age': [data.age],
        'workclass': [data.workclass],
        'fnlgt': [data.fnlgt],
        'education': [data.education],
        'education-num': [data.education_num],
        'marital-status': [data.marital_status],
        'occupation': [data.occupation],
        'relationship': [data.relationship],
        'race': [data.race],
        'sex': [data.sex],
        'capital-gain': [data.capital_gain],
        'capital-loss': [data.capital_loss],
        'hours-per-week': [data.hours_per_week],
        'native-country': [data.native_country]
    }

    input_df = pd.DataFrame(input_dict)

    # Process the input
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run inference
    pred = inference(model, X)

    # Convert prediction back to label
    prediction = lb.inverse_transform(pred)[0]

    return {"prediction": prediction}
