#Simple Variance Covariance BackTesting (Multiassets)
rm(list=ls())
setwd("G:\\My Drive\\Asus\\Subjects_Preparation\\NuLearn\\AFRM\\Batch_12\\Sessions\\CampusWS_1")
price=read.table("dataset_5.csv", header=T, sep=",")
l=length(price[,1])
ret1=diff(price[,1])/lag(price[,1])
ret1=ret1[1:(l-1)]
ret2=diff(price[,2])/lag(price[,2])
ret2=ret2[1:(l-1)]
ret3=diff(price[,3])/lag(price[,3])
ret3=ret3[1:(l-1)]
ret4=diff(price[,4])/lag(price[,4])
ret4=ret4[1:(l-1)]
ret5=diff(price[,5])/lag(price[,5])
ret5=ret5[1:(l-1)]

ret_comb=cbind(ret1,ret2,ret3,ret4,ret5)  #Combine all returns series as matrix

#Backtesting
l=length(ret1)
n1=l-250
#declaring the variables that we will generate in loop
port_value=rowSums(price)   #Value of portfolio on different days
var_95=0
actual_pl=0
excess_cap=0
excess_loss=0
failure=0
for (i in 1:n1) {
  we_1=as.numeric(price[i+250,])   #Monetary weight
  var_covar=var(ret_comb[i:(i+250-1),])   #Estimate variance-covariance matrix
  port_vari=t(we_1)%*%var_covar%*%we_1  #Monetary Variance
  var_95[i]=qnorm(0.05)*sqrt(port_vari) #Monetary VaR 
  actual_pl[i]=port_value[i+250+1]-port_value[i+250]  #Actual Profit/Loss
  if(actual_pl[i]<=var_95[i]){
    failure[i]=1
    excess_loss[i]=actual_pl[i]-var_95[i]
    excess_cap[i]=0
  } else {
    failure[i]=0
    excess_loss[i]=0
    excess_cap[i]=actual_pl[i]-var_95[i]
  }
}
failure_rate=sum(failure)/n1
failure_rate
total_excess_cap=sum(excess_cap)
total_excess_cap
total_excess_loss=sum(excess_loss)
total_excess_loss

#Multivariate RiskMetrics BackTesting
rm(list=ls())
price=read.table("dataset_5.csv", header=T, sep=",")
l=length(price[,1])
ret1=diff(price[,1])/lag(price[,1])
ret1=ret1[1:(l-1)]
ret2=diff(price[,2])/lag(price[,2])
ret2=ret2[1:(l-1)]
ret3=diff(price[,3])/lag(price[,3])
ret3=ret3[1:(l-1)]
ret4=diff(price[,4])/lag(price[,4])
ret4=ret4[1:(l-1)]
ret5=diff(price[,5])/lag(price[,5])
ret5=ret5[1:(l-1)]

ret_comb=cbind(ret1,ret2,ret3,ret4,ret5)

library(RiskPortfolios)
var_covar_1=covEstimation(ret_comb, control = list(type = 'ewma'))   #Estimate variance-covariance matrix
write.csv(var_covar_1,"RM_Variance_Covariance_Matrix.csv")


#Backtesting
l=length(ret1)
n1=l-250
#declaring the variables that we will generate in loop
port_value=rowSums(price)
var_95=0
actual_pl=0
excess_cap=0
excess_loss=0
failure=0
for (i in 1:n1) {
  we_1=as.numeric(price[i+250,])   #Monetary weight
  var_covar=covEstimation(ret_comb[i:(i+250-1),], control = list(type = 'ewma'))   #Estimate variance-covariance matrix
  port_vari=t(we_1)%*%var_covar%*%we_1  #Monetary Variance
  var_95[i]=qnorm(0.05)*sqrt(port_vari)
  actual_pl[i]=port_value[i+250+1]-port_value[i+250]
  if(actual_pl[i]<=var_95[i]){
    failure[i]=1
    excess_loss[i]=actual_pl[i]-var_95[i]
    excess_cap[i]=0
  } else {
    failure[i]=0
    excess_loss[i]=0
    excess_cap[i]=actual_pl[i]-var_95[i]
  }
}
failure_rate=sum(failure)/n1
failure_rate
total_excess_cap=sum(excess_cap)
total_excess_cap
total_excess_loss=sum(excess_loss)
total_excess_loss

#DCC GARCH (Two-assets)
rm(list=ls())
price=read.table("dataset_1.csv", header=T, sep=",")
data=cbind(diff(price[,1])/lag(price[,1]),diff(price[,2])/lag(price[,2]))
l=length(data[,1])
data=data[1:(l-1),]

library(rugarch)   
library(rmgarch)
#Specify the GARCH model
garch_spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "norm")
dcc_spec=dccspec(uspec=multispec(replicate(2,garch_spec1)),dccOrder=c(1,1),distribution="mvnorm")
dcc_fit=dccfit(dcc_spec,data=data)
dcc_fit
fitted_corr=rcor(dcc_fit,type='R',output='matrix')  #Extract fitted correlations
fitted_vol=sigma(dcc_fit)     #Extract Fitted Volatilities
write.csv(fitted_corr,"corr_dcc_2.csv")
write.csv(fitted_vol,"Volatility_dcc_2.csv")

dcc_for=dccforecast(dcc_fit,n.ahead = 1)
corr_for=dcc_for@mforecast$R[[1]]
vol_for=sigma(dcc_for)[c(1,2)]
write.csv(corr_for,"For_corr_dcc_2.csv")
write.csv(vol_for,"For_Volatility_dcc_2.csv")

#DCC GARCH (Multi-assets)
rm(list=ls())
price=read.table("dataset_5.csv", header=T, sep=",")
data=cbind(diff(price[,1])/lag(price[,1]),diff(price[,2])/lag(price[,2]),diff(price[,3])/lag(price[,3]),diff(price[,4])/lag(price[,4]),diff(price[,5])/lag(price[,5]))
l=length(data[,1])
data=data[1:(l-1),]

library(rugarch)   
library(rmgarch)
#Specify the GARCH model
garch_spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "norm")
dcc_spec=dccspec(uspec=multispec(replicate(5,garch_spec1)),dccOrder=c(1,1),distribution="mvnorm")
dcc_fit=dccfit(dcc_spec,data=data)
dcc_fit
fitted_corr=rcor(dcc_fit,type='R',output='matrix')
fitted_vol=sigma(dcc_fit)
#write.csv(fitted_corr,"Corr_dcc_5.csv")
#write.csv(fitted_vol,"Volatility_dcc_5.csv")

dcc_for=dccforecast(dcc_fit,n.ahead = 1)
corr_for=dcc_for@mforecast$R[[1]]
corr_for
vol_for=diag(sigma(dcc_for)[c(1,2,3,4,5)],5,5)
vol_for
write.csv(corr_for,"For_corr_dcc_5.csv")
write.csv(vol_for,"For_Volatility_dcc_5.csv")

#Backtesting DCC-GARCH (Multi-assets)
rm(list=ls())
price=read.table("dataset_5.csv", header=T, sep=",")
data=cbind(diff(price[,1])/lag(price[,1]),diff(price[,2])/lag(price[,2]),diff(price[,3])/lag(price[,3]),diff(price[,4])/lag(price[,4]),diff(price[,5])/lag(price[,5]))
#write.csv(data,"Return_out.csv")
l=length(data[,1])
data=data[1:(l-1),]
l1=length(data[,1])

library(rugarch)   
library(rmgarch)
#Specify the GARCH model
garch_spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "norm")
dcc_spec=dccspec(uspec=multispec(replicate(5,garch_spec1)),dccOrder=c(1,1),distribution="mvnorm")
dcc_fit=dccfit(dcc_spec,data=data)
dcc_fit

#Backtesting
n1=l1-250
#declaring the variables that we will generate in loop
port_value=rowSums(price)
var_95=0
actual_pl=0
excess_cap=0
excess_loss=0
failure=0
for (i in 1:n1) {
  we_1=as.numeric(price[i+250,])  #Monetary Weight
  var_covar=dcc_fit@mfit$H[,,i+250-1]
  port_vari=t(we_1)%*%var_covar%*%we_1
  var_95[i]=qnorm(0.05)*sqrt(port_vari)
  actual_pl[i]=port_value[i+250+1]-port_value[i+250]
  if(actual_pl[i]<=var_95[i]){
    failure[i]=1
    excess_loss[i]=actual_pl[i]-var_95[i]
    excess_cap[i]=0
  } else {
    failure[i]=0
    excess_loss[i]=0
    excess_cap[i]=actual_pl[i]-var_95[i]
  }
}
failure_rate=sum(failure)/n1
failure_rate
total_excess_cap=sum(excess_cap)
total_excess_cap
total_excess_loss=sum(excess_loss)
total_excess_loss

#CCC GARCH (Two-assets)
rm(list=ls())
price=read.table("dataset_1.csv", header=T, sep=",")
data=cbind(diff(price[,1])/lag(price[,1]),diff(price[,2])/lag(price[,2]))
l=length(data[,1])
data=data[1:(l-1),]

library(rugarch)   
library(rmgarch)
#Specify the GARCH model
garch_spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "norm")
ccc_spec = cgarchspec(uspec = multispec(replicate(2, garch_spec1)),dccOrder = c(1, 1), distribution.model = list(copula = "mvnorm", method = "ML", time.varying = FALSE))
ccc_fit=cgarchfit(ccc_spec,data=data)
ccc_fit
fitted_corr=rcor(ccc_fit,output='matrix')  #Extract fitted correlations
fitted_vol=sigma(ccc_fit)     #Extract Fitted Volatilities
write.csv(fitted_corr,"corr_ccc_2.csv")
write.csv(fitted_vol,"Volatility_dcc_2.csv")

#Generate forecasts of volatility
dcc_spec=dccspec(uspec=multispec(replicate(2,garch_spec1)),dccOrder=c(1,1),distribution="mvnorm")
dcc_fit=dccfit(dcc_spec,data=data)
dcc_for=dccforecast(dcc_fit,n.ahead = 1)
vol_for=sigma(dcc_for)[c(1,2)]
write.csv(vol_for,"For_Volatility_dcc_2.csv")

#CCC GARCH (Multi-assets)
rm(list=ls())
price=read.table("dataset_5.csv", header=T, sep=",")
data=cbind(diff(price[,1])/lag(price[,1]),diff(price[,2])/lag(price[,2]),diff(price[,3])/lag(price[,3]),diff(price[,4])/lag(price[,4]),diff(price[,5])/lag(price[,5]))
l=length(data[,1])
data=data[1:(l-1),]

library(rugarch)   
library(rmgarch)
#Specify the GARCH model
garch_spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "norm")
ccc_spec = cgarchspec(uspec = multispec(replicate(5, garch_spec1)),dccOrder = c(1, 1), distribution.model = list(copula = "mvnorm", method = "ML", time.varying = FALSE))
ccc_fit=cgarchfit(ccc_spec,data=data)
ccc_fit
fitted_corr=rcor(ccc_fit,output='matrix')  #Extract fitted correlations
fitted_vol=sigma(ccc_fit)     #Extract Fitted Volatilities
write.csv(fitted_corr,"corr_ccc_5.csv")
write.csv(fitted_vol,"Volatility_dcc_5.csv")

dcc_spec=dccspec(uspec=multispec(replicate(5,garch_spec1)),dccOrder=c(1,1),distribution="mvnorm")
dcc_fit=dccfit(dcc_spec,data=data)
dcc_for=dccforecast(dcc_fit,n.ahead = 1)
vol_for=diag(sigma(dcc_for)[c(1,2,3,4,5)],5,5)
vol_for
write.csv(vol_for,"For_Volatility_dcc_5.csv")

#Backtesting CCC-GARCH (Multi-assets)
rm(list=ls())
price=read.table("dataset_5.csv", header=T, sep=",")
data=cbind(diff(price[,1])/lag(price[,1]),diff(price[,2])/lag(price[,2]),diff(price[,3])/lag(price[,3]),diff(price[,4])/lag(price[,4]),diff(price[,5])/lag(price[,5]))
#write.csv(data,"Return_out.csv")
l=length(data[,1])
data=data[1:(l-1),]
l1=length(data[,1])

library(rugarch)   
library(rmgarch)
#Specify the GARCH model
garch_spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "norm")
ccc_spec = cgarchspec(uspec = multispec(replicate(5, garch_spec1)),dccOrder = c(1, 1), distribution.model = list(copula = "mvnorm", method = "ML", time.varying = FALSE))
ccc_fit=cgarchfit(ccc_spec,data=data)

#Backtesting
n1=l1-250
#declaring the variables that we will generate in loop
port_value=rowSums(price)
var_95=0
actual_pl=0
excess_cap=0
excess_loss=0
failure=0
for (i in 1:n1) {
  we_1=as.numeric(price[i+250,])  #Monetary Weight
  var_covar=ccc_fit@mfit$H[,,i+250-1]
  port_vari=t(we_1)%*%var_covar%*%we_1
  var_95[i]=qnorm(0.05)*sqrt(port_vari)
  actual_pl[i]=port_value[i+250+1]-port_value[i+250]
  if(actual_pl[i]<=var_95[i]){
    failure[i]=1
    excess_loss[i]=actual_pl[i]-var_95[i]
    excess_cap[i]=0
  } else {
    failure[i]=0
    excess_loss[i]=0
    excess_cap[i]=actual_pl[i]-var_95[i]
  }
}
failure_rate=sum(failure)/n1
failure_rate
total_excess_cap=sum(excess_cap)
total_excess_cap
total_excess_loss=sum(excess_loss)
total_excess_loss

#Backtesting Monte Carlo Simulation for two assets
rm(list=ls())
price=read.table("dataset_1.csv", header=T, sep=",")
l=length(price[,1])
ret1=diff(price[,1])/lag(price[,1])
ret1=ret1[1:(l-1)]
ret2=diff(price[,2])/lag(price[,2])
ret2=ret2[1:(l-1)]
ret_comb=cbind(ret1,ret2)

#Backtesting
l=length(ret1)
n1=l-250
#declaring the variables that we will generate in loop
price2=price
price2[,1]=price[,1]*100
price2[,2]=price[,2]*150
port_value=rowSums(price2)
var_95=0
var_port=0
actual_pl=0
excess_cap=0
excess_loss=0
failure=0
z=0
source("stockprice1.R")
for (i in 1:n1) {
  corr_m=cor(ret_comb[i:(i+250-1),])
  chol_m=chol(corr_m)    #Cholesky decomposition
  aa=dim(ret_comb[i:(i+250-1),])
  m=1000
  z=matrix(rnorm(aa[2]*m,0,1),nrow = m,ncol = aa[2])
  xx=z%*%chol_m
  for (j1 in 1:aa[2]) {
    xxx=xx[,j1]
    s0=price[(i+250),j1]  #Give last day price
    r=mean(ret_comb[i:(i+250-1),j1]) #Average Return
    sig=sd(ret_comb[i:(i+250-1),j1])  #Standard deviation
    t=1             #Time horizon (1 day)
    n=1             #No. of steps in simulation
    price1=stockprice1(s0,r,sig,t,n,m,xxx)  #Using function to simulate prices
    pl_sim=price1[,2]-price1[,1]  #Generate P/L distribution
    var_95[j1]=quantile(pl_sim,0.05)
  }
  var_port[i]=var_95[1]*100+var_95[2]*150
  actual_pl[i]=port_value[i+250+1]-port_value[i+250]
  if(actual_pl[i]<=var_port[i]){
    failure[i]=1
    excess_loss[i]=actual_pl[i]-var_port[i]
    excess_cap[i]=0
  } else {
    failure[i]=0
    excess_loss[i]=0
    excess_cap[i]=actual_pl[i]-var_port[i]
  }
}
failure_rate=sum(failure)/n1
failure_rate
total_excess_cap=sum(excess_cap)
total_excess_cap
total_excess_loss=sum(excess_loss)
total_excess_loss

#Backtesting Monte Carlo Simulation (Multiassets)
rm(list=ls())
price=read.table("dataset_5.csv", header=T, sep=",")
l=length(price[,1])
ret1=diff(price[,1])/lag(price[,1])
ret1=ret1[1:(l-1)]
ret2=diff(price[,2])/lag(price[,2])
ret2=ret2[1:(l-1)]
ret3=diff(price[,3])/lag(price[,3])
ret3=ret3[1:(l-1)]
ret4=diff(price[,4])/lag(price[,4])
ret4=ret4[1:(l-1)]
ret5=diff(price[,5])/lag(price[,5])
ret5=ret5[1:(l-1)]
ret_comb=cbind(ret1,ret2,ret3,ret4,ret5)

#Backtesting
l=length(ret1)
n1=l-250
#declaring the variables that we will generate in loop
port_value=rowSums(price)
var_95=0
var_port=0
actual_pl=0
excess_cap=0
excess_loss=0
failure=0
z=0
source("stockprice1.R")
for (i in 1:n1) {
  corr_m=cor(ret_comb[i:(i+250-1),])
  chol_m=chol(corr_m)
  aa=dim(ret_comb[i:(i+250-1),])
  m=1000
  z=matrix(rnorm(aa[2]*m,0,1),nrow = m,ncol = aa[2])
  xx=z%*%chol_m
  for (j1 in 1:aa[2]) {
    xxx=xx[,j1]
    s0=price[(i+250),j1]  #Give last day price
    r=mean(ret_comb[i:(i+250-1),j1]) #Average Return
    sig=sd(ret_comb[i:(i+250-1),j1])  #Standard deviation
    t=1             #Time horizon (1 day)
    n=1             #No. of steps in simulation
    price1=stockprice1(s0,r,sig,t,n,m,xxx)  #Using function to simulate prices
    pl_sim=price1[,2]-price1[,1]  #Generate P/L distribution
    var_95[j1]=quantile(pl_sim,0.05)
  }
  var_port[i]=sum(var_95)
  actual_pl[i]=port_value[i+250+1]-port_value[i+250]
  if(actual_pl[i]<=var_port[i]){
    failure[i]=1
    excess_loss[i]=actual_pl[i]-var_port[i]
    excess_cap[i]=0
  } else {
    failure[i]=0
    excess_loss[i]=0
    excess_cap[i]=actual_pl[i]-var_port[i]
  }
}
failure_rate=sum(failure)/n1
failure_rate
total_excess_cap=sum(excess_cap)
total_excess_cap
total_excess_loss=sum(excess_loss)
total_excess_loss
