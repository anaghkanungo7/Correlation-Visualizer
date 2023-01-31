from flask import Flask, render_template, request

app = Flask(__name__)

# Visualizer imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
import base64



def fetchData(first, second):
    # Fetch data for S&P (temp)
    temp = yf.download(first, start="2020-01-07", end="2022-01-21")
    # Calculate adjusted return
    temp['X'] = ((temp['Adj Close'] - temp['Open']) / temp['Open']) * 100
    # Remove the other columns
    temp.drop(labels=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1, inplace=True)

    # Fetch data for Tesla and store in a temporary variable
    temp2 = yf.download(second, start="2020-01-07", end="2022-01-21")
    # Calculate adjusted return
    temp2['Y'] = ((temp2['Adj Close'] - temp2['Open']) / temp2['Open']) * 100
    # Remove the other columns
    temp2.drop(labels=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1, inplace=True)

    # Merge into one DataFrame (replace existing data)
    data = pd.merge(temp, temp2, left_index=True, right_index=True)
    
    return data


def fitRegressionModel(data):
    X = sm.add_constant(data['X'])
    model = sm.OLS(data['Y'], X).fit()
    
    # Display OLS summary
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 14}, fontproperties = 'monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.1)
    plt.savefig('static/images/summary.png', bbox_inches='tight', dpi=300);
    plt.close()
    
    # Display model and residuals
    data['y-hat'] = model.predict()
    data['residuals'] = model.resid
    ax = data.plot.scatter(x='X', y='Y', c='darkgrey', figsize=(14,6))
    data.plot.line(x='X', y='y-hat', ax=ax);
    for _, row in data.iterrows():
        plt.plot((row.X, row.X), (row.Y, row['y-hat']), 'k-')    
    sns.despine()
    plt.tight_layout();
    plt.savefig('static/images/model.png', bbox_inches='tight', dpi=300);
    plt.close()
    
    
@app.route("/", methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        first = request.form['first']
        second = request.form['second']
        data = fetchData(first, second)
        fitRegressionModel(data)
        
        return render_template('/index.html', results=first, summaryImage="/static/images/summary.png", modelImage="/static/images/model.png")
    else:
        return render_template("/index.html")


@app.route("/calculate", methods=['POST'])
def calculate():
    return render_template('/index.html')