# Machine Learning Integrated API
API that takes as input Titanic passenger characteristics and outputs 
survival outcomes through logistic regression classifier.

## Requirements
* python3
    * pandas
    * numpy
    * sklearn
    * flask
    * traceback
* Postman (for API testing)

## Usage
Clone repo and go into directory.
```git clone https://github.com/matteo-esposito/ml-api.git```
```cd ml-api```

Run model

```python3 model.py```

Start flask app

```python3 api.py```

Go into postman API client and enter the url of the app and go to the `predict` path

```POST  localhost:12345/predict```

Enter values that you wish to test with the model

```
[
	{"Age": 85, "Sex": "female", "Embarked": "S"},
	{"Age": 22, "Sex": "male", "Embarked": "C"},
	{"Age": 29, "Sex": "female", "Embarked": "C"},
	{"Age": 55, "Sex": "male", "Embarked": "S"}
]
```

Get prediction

```
{
    "prediction": "[1, 0, 1, 0]"
}
```