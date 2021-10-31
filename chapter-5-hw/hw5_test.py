import requests

url = 'http://0.0.0.0:5501/predict'

q3 = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

q4 = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}

q6 = {"contract": "two_year", "tenure": 12, "monthlycharges": 10}

req = requests.post(url, json= q6).json()

churn_str = ['churn' if req['churn'] == True else 'not churn']
churn_proba = req['churn_probability']

print(f'The customer will {churn_str[0]} with probability of {round(churn_proba, 3)} ')
