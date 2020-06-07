# Step 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# step 2
df=pd.read_csv("E:\\Kaggle Project\\Project 3 - Airlines data\\Airlines.csv")
# Step 3 EDA
df.tail()
df.isnull().sum()
df["Month"]=pd.to_datetime(df["Month"])
df.Month[1]
df.head()
df.rename(columns={"#Passengers":"Passengers"},inplace=True)
df.columns
# we have to make the timestamp column as the index

df=df.set_index(["Month"])
df.head()

# We need to check if the data is stationary or not 

plt.plot(df)

# There are two tests to check the stationarity  1. Rolling Statistics 2. ADCF Test

#step-4


rolmean=df["Passengers"].rolling(window=12).mean()

rolstd=df["Passengers"].rolling(window=12).std()


plt.plot(df,color="black",label="Orginal")
plt.plot(rolmean,color="blue",label="rolmean")
plt.plot(rolstd,color="red",label="rolstd")
plt.legend(loc="best")
plt.title("Rolling mean & Standard Deviation")


#Step 5- ADCF Test

from statsmodels.tsa.stattools import adfuller
dftest1=adfuller(df["Passengers"],autolag="AIC")
print(dftest1)
for items in dftest1[0:4]:
    dfoutput=pd.Series(dftest1[0:4],index=["Test-Statistic","p-value","Number of lags","Number of Observations"])  

for keys,values in dftest1[4].items():
    dfoutput["The Critical value at %s"%keys]=values
    
print(dfoutput)

# Step -6- Diffrencing (If seasonal influence not there shift by 1,If seasonal influence there shift by 12) 

df["First Difference"]=df["Passengers"]-df["Passengers"].shift(1)



# DF test for the second time
dftest2=adfuller(df["First Difference"].dropna(),autolag="AIC")
print(dftest2)

# We have to shift it one more time 
df["Second Difference"]=df["Passengers"]-df["Passengers"].shift(2)


# DF test for the third time
dftest3=adfuller(df["Second Difference"].dropna(),autolag="AIC")
print(dftest3)
# Value of D can be Finalized to two

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig=plot_acf(df["First Difference"].dropna(),lags=40)
fig=plot_pacf(df["First Difference"].dropna(),lags=40)

# Order =(2,2,2) or (2,1,2)

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(df["Passengers"],order=(2,1,2))
results=model.fit()
results.summary()
df["Forecast1"]=results.predict(start=100,end=143,dynamic=True)
df[["Forecast1","Passengers"]].plot()
# Which clearly shows there is an issue of seasonality

df["First Seasonal difference"]=df["Passengers"]-df["Passengers"].shift(12)
plt.plot(df["First Seasonal difference"])

dftest4=adfuller(df["First Seasonal difference"].dropna(),autolag="AIC")
print(dftest4)
# Which Clearly shows there exists a seasonality issue so we should not be using ARIMA we should use SARIMAX



import statsmodels.api as sm
model1=sm.tsa.statespace.SARIMAX(df["Passengers"],order=(2,1,2),seasonal_order=(2,1,2,12))
result1=model1.fit()

result1.summary()
df.drop("Forecast",axis=1)
df["Forecast"]=result1.predict(start=100,end=143,dynamic=True)
df[["Passengers","Forecast"]].plot()

from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+DateOffset(months=x) for x in range(1,121)]
type(future_dates)
future_dates_df=pd.DataFrame(index=future_dates[0:],columns=df.columns)
future_dates_df

future_df=pd.concat([df,future_dates_df])
future_df.tail(5)

future_df["Forecast"]=result1.predict(start=120,end=263,dynamic=True)
future_df[["Passengers","Forecast"]].plot()
