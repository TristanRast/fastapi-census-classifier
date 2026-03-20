# ML Model Deployment with FastAPI

This project demonstrates deploying a machine learning model to a cloud application platform using FastAPI, with complete CI/CD integration using GitHub Actions.

## Project Overview

This project implements a classification model trained on census data to predict income levels. The model is deployed as a REST API using FastAPI and includes comprehensive testing, continuous integration, and model performance analysis.

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── python-app.yml          # GitHub Actions CI/CD pipeline
├── ml/
│   ├── __init__.py
│   ├── data.py                      # Data processing functions
│   └── model.py                     # Model training and inference
├── model/
│   ├── model.pkl                    # Trained model (generated)
│   └── encoder.pkl                  # Label encoder (generated)
├── screenshots/
│   ├── continuous_integration.png   # CI passing screenshot
│   ├── unit_test.png               # Unit tests passing screenshot
│   └── local_api.png               # Local API test screenshot
├── tests/
│   └── test_model.py               # Unit tests
├── data/
│   └── census.csv                  # Training data
├── main.py                         # FastAPI application
├── train_model.py                  # Model training script
├── local_api.py                    # Local API testing script
├── model_card.md                   # Model documentation
├── slice_output.txt                # Model performance on data slices (generated)
├── requirements.txt                # Python dependencies
└── .gitignore
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Load and process the census data
- Train the classification model
- Save the model and encoder to `model/` directory
- Generate `slice_output.txt` with performance metrics on data slices

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Check Code Quality

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### 7. Run the API Locally

Terminal 1 - Start the server:
```bash
uvicorn main:app --reload
```

Terminal 2 - Test the API:
```bash
python local_api.py
```

## API Endpoints

### GET /
Returns a welcome message.

**Response:**
```json
{
  "greeting": "Welcome to the ML Model API!"
}
```

### POST /predict
Performs model inference on input data.

**Request Body:**
```json
{
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
```

**Response:**
```json
{
  "prediction": "<=50K"
}
```

## Model Performance

See `model_card.md` for detailed model documentation and performance metrics.

See `slice_output.txt` for performance breakdown across categorical feature slices.

## CI/CD

This project uses GitHub Actions for continuous integration. On every push to main/master:
- Code is checked with flake8
- All unit tests are run with pytest

See `.github/workflows/python-app.yml` for the CI configuration.

## License

MIT
