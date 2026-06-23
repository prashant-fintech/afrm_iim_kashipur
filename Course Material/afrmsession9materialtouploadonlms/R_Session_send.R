#Deleting all variables from the memory
rm(list=ls())

# Setting a working directory (\\=Double backslash) or (/=single forward slash)
setwd("G:\\My Drive\\Asus\\Subjects_Preparation\\NuLearn\\AFRM\\Batch_12\\Sessions\\Session_9\\R_Session")
#OR
setwd("G:/My Drive/Asus/Subjects_Preparation/NuLearn/AFRM/Batch_12/Sessions/Session_9/R_Session")
# Importing data in R as dataframes (If you have already used the "setwd" command, then put the required data files in that working directory

# Importing data from csv file
data1=read.csv("nifty.csv", header=T)
nifty_price=data1$Nifty   #Separate Nifty Price
nifty=data1[,2]
n=length(nifty)   #Length of data series
#Matrix - [row,column]
# For all rows - [,column]
# For all columns - [row,]

#Continuously compounded Return=ln(P_t/P_t-1)=ln(P_t)-ln(P_t-1)
#Log(A/B)=Log(A)-Log(B)
#We can do the same things (Taking diff of Consecutive terms)=diff

nif_ret_cc=diff(log(nifty))
nif_ret_cc

nif_ret_simp=diff(nifty)/lag(nifty)
nif_ret_simp=nif_ret_simp[1:(n-1)]

#plot
plot(nif_ret_cc,type = "l")
plot(nif_ret_simp,type = "l")
plot(nifty,type = "l")

#Combine the data in columns
out=cbind(nif_ret_cc,nif_ret_simp)
# Exporting results
write.csv(out,"Return_All.csv")

# if statement
#Conditional statement
x=rnorm(1,0,1)
if(x>0){
  y=5*x
} else {
  y=0
}
y

# if - else if - else - statement
x=rnorm(1,0,1)   #Generate N(0,1) random number
if(x>0){
  y=5*x
} else if(x<0){
  y=10*x
} else {
  y=0
}
y

# For loop: Example 1:
for(i in 1:20){
  print(5*i)
}


y=0
for(i in 1:20){
  y[i]=5*i
}
y

for(i in 1:20){
  print(i*5+1)
}
i


# Elementary Maths function
n=25
o=sqrt(n)
o 
o=exp(2)
o  
o=log(25)
o   # Natural logarithm
o=abs(-5)
o  #Absolute value
o=round(1.5109)
o
o=sign(-3)
o

