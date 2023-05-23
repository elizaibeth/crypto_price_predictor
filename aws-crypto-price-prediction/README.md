# Bitcoin Price Prediction Engine

Lets all loose money together - using python and to be deployed as a lambda on AWS

# Install

Create a new virtual environment by running the command `python3 -m venv env`.

Activate the virtual environment by running the command `source env/bin/activate`.

Install all the required packages in the virtual environment by running the command `python3 -m pip install -r requirements.txt` ....

You must always run this pip install when you add new requirements

# Run Locally 
In the root run

```
serverless invoke local --function lambda_handler
```

or 

```
serverless invoke local --data '{"body": "{\"ticker\":\"BTC-USD\"\n}"}' --function lambda_handler
```

# Deploy to Lambda
```
serverless deploy --aws-profile default --region eu-west-2  
```