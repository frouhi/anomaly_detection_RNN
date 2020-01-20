## Anomaly Detection Using RNNs and TensorFlow
**Author**
* Farhang Rouhi

**Introduction**

This program is for anomaly detection in any financial data that includes [open,high,low,close,volume] fields.
First, we create a Recurrent Neural Network (RNN) to forecast [open,high,low,close] fields in the financial times series.
Note that the financial data used in this program is multivariate time series in order to increase accuracy of anomaly detection step.
After training the model, it is used to predict [open,high,low,close] fields. If error of all four fields pass a threshold, 
that data point is marked as anomalous. At the end, a plot visualizes the anomalies. 