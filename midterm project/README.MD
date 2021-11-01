# Midterm Project

## Dataset

Data taken from kaggle, link to dataset [here](https://www.kaggle.com/fedesoriano/heart-failure-prediction)


## Why heart disease dataset?

I am interested on these kind of datasets, because my grandfather dies due to a heart disease. 
I didn't have any heart issues but my father an me have some hypertension, so I suspect that my grandfather died as a consequence of that, 35 years ago common people weren't usually gone to the doctor. 


## Description

Common heart diseases are: heart-attacks, strokes, heart-failures, etc...
With ML we may train a model which helps detects the probability of heart-disease due to certain attributes information such age, cholesterol levels, sugar, heart-rate, and so on.

In particular, this dataset was created by combination of several datasets, so it could be interesting train a model with more information. 

It is important to note that, if we add a ml-service for inference to support a hypothetical software, we could experienced some bias, like people who dies but we don't register information about them (because it was suddenly and doctors didn't measure nothing).
I think that is the most important issue in datasets, the possibility of suffer bias related to "survival analysis". At the moment, I don't know how to work against that bias.

The target variable is HeartDisease, which have 1 if it is true, and 0 to false, so the problem is a binary classification. It could be a multiclass classification if the targets were diseases instead of actual values.

A longer description could be found at dataset's kaggle page.

## EDA
Information is contained on notebooks/00-EDA.ipybn, the other notebooks is the "training" one.

## Things Used

- 3 models: logistic regression, random forest and xgboost
- optuna to find hyperparameters
- serve with docker+fastapi
- poetry
- pyenv

## How To

For start the project, you need an environment. I've used pyenv with virtualenv plugin...

```bash
pyenv virtualenv 3.9.2 <your_environment>
pip install poetry
poetry install
```

Then, you will have all dependencies installed
For training, you need to change directory due to some relative paths on the script [fix me]

```
cd scripts
python train.py
```

The training will export 4 artifacts: preprocessor, lr, rf and xgb.

For inference:

```
cd ../ # if you are on scripts
python inference.py
# go to localhost:8000/docs
# use the /predict endpoint
```

It will returns, by each model, the probability for having a heart disease.

All models have accuracy between 0.86 and 0.90, this is showing when the training script is running as logging.

## Docker Image

Building docker image

```
docker build -t jloscalzo/midterm .
docker run -p 80:80 --rm -d jloscalzo/midterm

```

Running docker image

```
docker pull jloscalzo/mlzoomcamp-midtermproject:latest
docker run -p 80:80 --rm -d jloscalzo/mlzoomcamp-midtermproject:latest
```
