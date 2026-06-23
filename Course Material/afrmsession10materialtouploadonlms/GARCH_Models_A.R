rm(list=ls())   #Clearing the memory
#Setting a working directory
setwd("C:/projects/afrm_iim_kashipur/Course Material/afrmsession10materialtouploadonlms")
#Importing data from Nifty.csv file 
price=read.csv("reliance.csv", header=T)
#Separating the price series from second column
nifty_p=price[,2]

n=length(nifty_p)
#Estimate Simple return
nifty=diff(nifty_p)/lag(nifty_p)
nifty=nifty[1:(n-1)]
#nifty=diff(log(price[,2]))
library(rugarch)
#Specify a GARCH Model
spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "norm")
#Fit a GARCH model
garch_fit_1 = ugarchfit(spec = spec1, data = nifty)
garch_fit_1
sigma=garch_fit_1@fit$sigma
write.csv(sigma,file="reliance_vol_estimates_GARCH.csv")
for_c=ugarchforecast(garch_fit_1,n.ahead=1)
for_c1=for_c@forecast$sigmaFor[[1]]
for_c1

#GARCH model with Student t distribution
rm(list=ls())   #Clearing the memory
#Importing data from Nifty.csv file 
price=read.csv("reliance.csv", header=T)
#Separating the price series from second column
nifty_p=price[,2]

n=length(nifty_p)
#Estimate Simple return
nifty=diff(nifty_p)/lag(nifty_p)
nifty=nifty[1:(n-1)]
#nifty=diff(log(price[,2]))
library(rugarch)
#Specify a GARCH Model
spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "std")
#Fit a GARCH model
garch_fit_1 = ugarchfit(spec = spec1, data = nifty)
garch_fit_1
sigma=garch_fit_1@fit$sigma
write.csv(sigma,file="reliance_vol_estimates_GARCH_t.csv")
for_c=ugarchforecast(garch_fit_1,n.ahead=1)
for_c1=for_c@forecast$sigmaFor[[1]]
for_c1

#Student t quantiles
library(fGarch)
out1=stdFit(residuals(garch_fit_1,standardize=TRUE))
qstd(0.05,mean=0,sd=1,nu=out1$par[3])
qstd(0.01,mean=0,sd=1,nu=out1$par[3])
qstd(0.005,mean=0,sd=1,nu=out1$par[3])

#GARCH model with GED distribution
rm(list=ls())   #Clearing the memory
#Importing data from Nifty.csv file 
price=read.csv("reliance.csv", header=T)
#Separating the price series from second column
nifty_p=price[,2]

n=length(nifty_p)
#Estimate Simple return
nifty=diff(nifty_p)/lag(nifty_p)
nifty=nifty[1:(n-1)]
#nifty=diff(log(price[,2]))
library(rugarch)
#Specify a GARCH Model
spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "ged")
#Fit a GARCH model
garch_fit_1 = ugarchfit(spec = spec1, data = nifty)
garch_fit_1
sigma=garch_fit_1@fit$sigma
write.csv(sigma,file="reliance_vol_estimates_GARCH_ged.csv")
for_c=ugarchforecast(garch_fit_1,n.ahead=1)
for_c1=for_c@forecast$sigmaFor[[1]]
for_c1
#GED quantiles
library(fGarch)
out1=gedFit(residuals(garch_fit_1,standardize=TRUE))
qged(0.05,mean=0,sd=1,nu=out1$par[3])
qged(0.01,mean=0,sd=1,nu=out1$par[3])
qged(0.005,mean=0,sd=1,nu=out1$par[3])

#GARCH model with skewed normal distribution
rm(list=ls())   #Clearing the memory
#Importing data from Nifty.csv file 
price=read.csv("reliance.csv", header=T)
#Separating the price series from second column
nifty_p=price[,2]

n=length(nifty_p)
#Estimate Simple return
nifty=diff(nifty_p)/lag(nifty_p)
nifty=nifty[1:(n-1)]
#nifty=diff(log(price[,2]))
library(rugarch)
#Specify a GARCH Model
spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "snorm")
#Fit a GARCH model
garch_fit_1 = ugarchfit(spec = spec1, data = nifty)
garch_fit_1
sigma=garch_fit_1@fit$sigma
write.csv(sigma,file="reliance_vol_estimates_GARCH_snorm.csv")
for_c=ugarchforecast(garch_fit_1,n.ahead=1)
for_c1=for_c@forecast$sigmaFor[[1]]
for_c1

#Skewed Normal quantiles
library(fGarch)
out1=snormFit(residuals(garch_fit_1,standardize=TRUE))
qsnorm(0.05,mean=0,sd=1,xi=out1$par[3])
qsnorm(0.01,mean=0,sd=1,xi=out1$par[3])
qsnorm(0.005,mean=0,sd=1,xi=out1$par[3])
qsnorm(0.95,mean=0,sd=1,xi=out1$par[3])
qsnorm(0.99,mean=0,sd=1,xi=out1$par[3])
qsnorm(0.995,mean=0,sd=1,xi=out1$par[3])

#GARCH model with skewed t distribution
rm(list=ls())   #Clearing the memory
#Importing data from Nifty.csv file 
price=read.csv("reliance.csv", header=T)
#Separating the price series from second column
nifty_p=price[,2]

n=length(nifty_p)
#Estimate Simple return
nifty=diff(nifty_p)/lag(nifty_p)
nifty=nifty[1:(n-1)]
#nifty=diff(log(price[,2]))
library(rugarch)
#Specify a GARCH Model
spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "sstd")
#Fit a GARCH model
garch_fit_1 = ugarchfit(spec = spec1, data = nifty)
garch_fit_1
sigma=garch_fit_1@fit$sigma
write.csv(sigma,file="reliance_vol_estimates_GARCH_st.csv")
for_c=ugarchforecast(garch_fit_1,n.ahead=1)
for_c1=for_c@forecast$sigmaFor[[1]]
for_c1

#Skewed Student t quantiles
library(fGarch)
out1=sstdFit(residuals(garch_fit_1,standardize=TRUE))
qsstd(0.05,mean=0,sd=1,nu=out1$estimate[3],xi=out1$estimate[4])
qsstd(0.01,mean=0,sd=1,nu=out1$estimate[3],xi=out1$estimate[4])
qsstd(0.005,mean=0,sd=1,nu=out1$estimate[3],xi=out1$estimate[4])
qsstd(0.95,mean=0,sd=1,nu=out1$estimate[3],xi=out1$estimate[4])
qsstd(0.99,mean=0,sd=1,nu=out1$estimate[3],xi=out1$estimate[4])
qsstd(0.995,mean=0,sd=1,nu=out1$estimate[3],xi=out1$estimate[4])

#GARCH model with skewed GED distribution
rm(list=ls())   #Clearing the memory
price=read.csv("reliance.csv", header=T)
#Separating the price series from second column
nifty_p=price[,2]

n=length(nifty_p)
#Estimate Simple return
nifty=diff(nifty_p)/lag(nifty_p)
nifty=nifty[1:(n-1)]
#nifty=diff(log(price[,2]))
library(rugarch)
#Specify a GARCH Model
spec1=ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),mean.model = list(armaOrder = c(0, 0),include.mean = FALSE),distribution.model = "sged")
#Fit a GARCH model
garch_fit_1 = ugarchfit(spec = spec1, data = nifty)
garch_fit_1
sigma=garch_fit_1@fit$sigma
write.csv(sigma,file="reliance_vol_estimates_GARCH_sged.csv")
for_c=ugarchforecast(garch_fit_1,n.ahead=1)
for_c1=for_c@forecast$sigmaFor[[1]]
for_c1

#Skewed GED quantiles
library(fGarch)
out1=sgedFit(residuals(garch_fit_1,standardize=TRUE))
qsged(0.05,mean=0,sd=1,nu=out1$par[3],xi=out1$par[4])
qsged(0.01,mean=0,sd=1,nu=out1$par[3],xi=out1$par[4])
qsged(0.005,mean=0,sd=1,nu=out1$par[3],xi=out1$par[4])
qsged(0.95,mean=0,sd=1,nu=out1$par[3],xi=out1$par[4])
qsged(0.99,mean=0,sd=1,nu=out1$par[3],xi=out1$par[4])
qsged(0.995,mean=0,sd=1,nu=out1$par[3],xi=out1$par[4])

