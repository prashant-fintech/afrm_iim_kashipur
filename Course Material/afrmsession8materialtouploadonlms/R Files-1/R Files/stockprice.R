stockprice<-function(s0,r,sig,t,n,m){
  #n = No. of steps in simulation
  #m = No. of simulations
  del_t=t/n   #Length of one step
  st=matrix(NA,m,n+1)
  st[,1]=s0
  #Inner loop will generate one path and outer loop will generate m paths
  for (j in 1:m){
    for (i in 1:n) {
      st[j,i+1]=st[j,i]*exp((r-sig^2/2)*del_t+sig*sqrt(del_t)*rnorm(1,0,1))
    }
  }
  return(st)
}