# Entirely fabricated data. Does not read any SEER/project patient files.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)) .libPaths(c(args[1], .libPaths()))
for (p in c('survival','cmprsk')) if (!requireNamespace(p,quietly=TRUE)) stop('Missing R package: ',p)
set.seed(20260723)
n <- 200L
d <- data.frame(time=pmax(1,round(rexp(n,.03))), status=sample(0:3,n,replace=TRUE),x=rnorm(n))
fit <- survival::coxph(survival::Surv(time,status==1)~x,data=d)
stopifnot(all(is.finite(coef(fit))))
km <- survival::survfit(survival::Surv(time,status==2)~1,data=d)
stopifnot(all(diff(km$surv)<=1e-12),all(km$surv>=0),all(km$surv<=1))
aj <- survival::survfit(survival::Surv(time,factor(status,levels=0:3))~1,data=d)
ci <- cmprsk::cuminc(d$time,d$status,cencode=0)
at <- function(t,v,h) {i<-findInterval(h,t);if(i==0L) 0 else v[i]}
for (h in c(12,36,60)) {
  a <- at(aj$time,aj$pstate[,match('1',aj$states)],h)
  b <- at(ci[['1 1']]$time,ci[['1 1']]$est,h)
  stopifnot(abs(a-b)<1e-10)
}
retained <- d[d$time>=6,]
conditional <- cmprsk::cuminc(retained$time-5,retained$status,cencode=0)
base_risk <- at(aj$time,aj$pstate[,match('1',aj$states)],5)
surv5 <- at(aj$time,aj$pstate[,match('(s0)',aj$states)],5)
for (h in c(12,36,60)) {
  target <- (at(aj$time,aj$pstate[,match('1',aj$states)],h)-base_risk)/surv5
  estimate <- at(conditional[['1 1']]$time,conditional[['1 1']]$est,h-5)
  stopifnot(abs(target-estimate)<1e-10)
}
cat('PASS: fabricated-data Cox/KM, competing-risk estimator agreement, and conditional-CIF identity.\n')
print(sessionInfo())
