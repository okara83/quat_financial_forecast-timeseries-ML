GluonTS stock market S&P500 prediction deep learning light-weight Jupyter Notebook. The data used is from 1970 to 2021


We will be predicting the AJUSTED CLOSE price for the trading days for 365 days from March 24, 2020 see the output below at 2000 epochs and a epoch loss rate of 1.98

Dependencies:
1) Install GluonTS & MXNet https://gluon-ts.mxnet.io/install.html
2) Install Jupyter Notebook + Python==<3.6

To change the time series prediction plot so you can see the predictions from a shorter previous time period than from 1970. Find <b>to_pandas(test_entry)[-30000:].plot(figsize=(40,20),linewidth=1)\n",</b> in the code and change <b>[-30000]</b> to another lesser negative number
