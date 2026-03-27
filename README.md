# Project Name: Rental price forcasting
test_change
other_branch_change

**Author:** Group 6    
**Course:** MLOps: Master in Business Analytics and Data Sciense  
**Status:** Session 1 (Initialization)

---

##  Business Objective


* **The Goal:** Accurately predict the rent price of rental properties in the city of Madrid to determine if a listing is overpriced, underpriced or fairly priced.

* **The User:** Tenants who are seaching for rental properties in Madrid.

---

##  Success Metrics

* **Business KPI (The "Why"):**
  > Increase conversion rate by 5% in one year.

* **Technical Metric (The "How"):**
  > Model MAPE (Mean Absolute Percentage Error) < 15% on the test set.

* **Acceptance Criteria:**
  > It outperforms the baseline of the district median price model.

---

##  The Data

* **Source:** CSC with retal prices, location and structural charactersitics of the properties.
* **Target Variable:** Rental price

---
## Overview

Rental prices in madrid has been increasing over the last couple of years. The challenge for users of real estate services is to find the best property at a fair price. But how can a user estimate the fair price of a property?. Thats the main goal of this project. based on Machine Lernaing models, this solutions allows renters from the city of Madrid to compare the current listing against what the fair price would be based on the characteristics of the property. 

Additionally this solution was built following MLOps priniciples what allows the owner of the application to ensure a stable service and continuos improvment adpating to the changes of the environment and the data available.

The project predicts the rental prices for properties in Madrid. I includes:

* Modular pipeline
* Centralize configuration
* Unit testing
* Experiment tracking and artifact logging with Weight and Biases
* a FastAPI service
* Stability of the service
* Continuos integration and continuos deployment 
* Deployment of an easy to use application 

## Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

.
├── Dockerfile
├── LICENSE
├── README.md
├── conda-lock.yml
├── config.yaml
├── environment.yml
├── pytest.ini
├── artifacts/
│   └── ret_prediction_model:v0/
│       └── model.joblib
├── data/
│   ├── raw/
│   │   └── opioid_raw_data.csv
│   ├── processed/
│   │   └── clean.csv
│   └── inference/
│       └── opioid_infer_01.csv
├── logs/
│   └── pipeline.log
├── models/
│   └── model.joblib
├── notebooks/
│   ├── Linear_Regression_Rent_prices_vf.ipynb
|   |── Linear_Regression_Rent_prices.ipynb
│   └── Sandbox.ipynb
├── reports/
│   └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── logger.py
│   ├── main.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_clean_data.py
    ├── test_evaluate.py
    ├── test_features.py
    ├── test_infer.py
    ├── test_load_data.py
    ├── test_main.py
    ├── test_train.py
    ├── test_utils.py
    └── test_validate.py

---
## Tech stack
* Python 
* Conda for environment management
* scikit-learn for model training
* FastAPI and Pydantic for serving and request validation
* Weights & Biases for experiment tracking and model artifacts
* Docker for containerized serving
* GitHub Actions for Continuous Integration
* Render for cloud deployment

---
## Configuration

config.yml is the single source of truth that includes all the paramters the model:

* file paths to data and artifacts
* linear regression settings
* split parameters for the training and testing of the models
* requeired features: categorical columns and numerical columns
* evaluation metrics
* W&B settings

.env contains the sensitive information:
* API keys

## Set Up

1) Create the environment
2) Run the unit tests
3) Run the full pipeline throught the orchestrator
4) Explore the notebook ´Sandbox.py´

## Usage
The application can be used in three different ways
1) Locally on the terminal by running the orchestrator main.py
2) Using the FastAPI service runnig the api.py
3) Accesing the deployment in Render:
 https://madrid-rentals-prediction.onrender.com/docs#/default/predict_predict_post:
 - Use the /docs endpoint for a more userfrendly experience

## Model Card

### Model Name

Madirid Rental price prediction.

### Model type

Linear regression pipeline built with scikitlearn.

### Inteded Use

Support user decision on the process of renting properties in the city of Madrid.

### Primary users

People looking for rental properties in Madrid.

### Prediction target


'rent_price'

### Inputs

Strucutre data containing features both categorical and numerical about the


`python -m src.main`



