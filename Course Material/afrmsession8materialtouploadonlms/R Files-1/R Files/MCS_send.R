# Monte Carlo Simulation with backtesting
# 1 day ahead VaR
rm(list=ls())   #Clears the memory
#Set a working directory
setwd("G:\\My Drive\\Asus\\Subjects_Preparation\\NuLearn\\AFRM\\Batch_12\\Sessions\\Session_8")
#Import csv file data in R
data=read.csv("nifty.csv", header=T)
#Separate Nifty Price data
nifty=data$Nifty
n=length(nifty)
ret1=diff(nifty)/lag(nifty)  #Computing Simple Return
write.csv(ret1,"Return.csv")  #Saving Return to file
nifty_r=ret1[1:(n-1)]  #Removing last extra garbage return value

#Call self made function
source("stockprice.R")

s0=nifty[length(nifty)]  #Give last day price
r=mean(nifty_r) #Average Return
sig=sd(nifty_r)  #Standard deviation
t=1             #Time horizon (1 day)
n=1             #No. of steps in simulation
m=100000   #Number of simulations
price=stockprice(s0,r,sig,t,n,m)  #Using function to simulate prices
pl_sim=price[,2]-price[,1]  #Generate P/L distribution
VaR_95_Nif=quantile(pl_sim,0.05)
VaR_99_Nif=quantile(pl_sim,0.01)
VaR_995_Nif=quantile(pl_sim,0.005)

#Backtesting
l=length(nifty_r)
n1=l-250
#declaring the variables that we will generate in loop
VaR_95_Nif_mc=0
actual_pl=0
violation=0
excess_cap=0
excess_loss=0
for (i in 1:n1) {
  s0=nifty[i+250]
  r=mean(nifty_r[i:(i+250-1)])
  sig=sd(nifty_r[i:(i+250-1)])
  t=1
  n=1
  m=1000
  price=stockprice(s0,r,sig,t,n,m)
  pl_sim=price[,2]-price[,1]
  
  VaR_95_Nif_mc[i]=quantile(pl_sim,0.05)
  actual_pl[i]=nifty[i+250+1]-nifty[i+250]
  if (actual_pl[i]<=VaR_95_Nif_mc[i]) {
    violation[i]=1
    excess_loss[i]=actual_pl[i]-VaR_95_Nif_mc[i]
    excess_cap[i]=0
  } else {
    violation[i]=0
    excess_loss[i]=0
    excess_cap[i]=actual_pl[i]-VaR_95_Nif_mc[i]
  }
}
failure_rate=sum(violation)/n1
failure_rate
tota_excess_cap=sum(excess_cap)
tota_excess_cap
total_excess_loss=sum(excess_loss)
total_excess_loss
out1=cbind(violation,excess_cap,excess_loss)
write.csv(out1,"MCS_Results.csv")
