# Curated package: never execute against the original project directory.
mpn_run_guard <- Sys.getenv("MPN_RUN_ROOT")
if (!nzchar(mpn_run_guard) || !file.exists(file.path(mpn_run_guard, ".mpn_reproduction_run")))
  stop("Use run_pipeline.py with a new marked run directory.")
if (normalizePath(Sys.getenv("MPN_PROJECT_ROOT"), mustWork=TRUE) != normalizePath(mpn_run_guard, mustWork=TRUE))
  stop("Project and run roots must match.")
mpn_args_guard <- commandArgs(trailingOnly=TRUE)
if (length(mpn_args_guard) && normalizePath(mpn_args_guard[1],mustWork=TRUE) != normalizePath(mpn_run_guard,mustWork=TRUE))
  stop("Argument root differs from protected run root.")
# Supplementary minimum observed interval >=6 months. No original files changed.
# Rscript.exe --vanilla run_latency6.R [root] [new_run_name]
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) args[1] else Sys.getenv("MPN_PROJECT_ROOT")
run_name <- if(length(args)>=2) args[2] else format(Sys.time(),"run_latency6_%Y%m%d_%H%M%S")
stopifnot(grepl("^[A-Za-z0-9_-]+$",run_name))
base <- file.path(root,"_codex_workspace/09_event_timing_sensitivity")
out <- file.path(base,run_name)
if(dir.exists(out)) stop("Refusing to overwrite previous run")
dir.create(out,recursive=TRUE)
library(data.table)
library(survival)
library(ggplot2)
stopifnot(requireNamespace("cmprsk",quietly=TRUE))
set.seed(20260723)
start <- Sys.time()
log <- file(file.path(out,"execution_log.txt"),"wt")
sink(log,split=TRUE)
cat("Started:",format(start),"\nRun:",run_name,"\n")
save <- function(x,name) fwrite(x,file.path(out,paste0(name,".csv")),bom=TRUE,na="NA")
qc <- data.table(check=character(),value=numeric(),passed=logical())
check <- function(label,value,ok) {
  qc <<- rbind(qc,data.table(check=label,value=value,passed=ok))
  if(!isTRUE(ok)) { save(qc,"qc_checks");stop(label) }
}
rel <- c("ML/data/ml_competing_risk_v0.4_2004_2023.csv","MPN/patient_master_v0.4_2000_2023.csv",
  "_codex_workspace/09_event_timing_sensitivity/primary_run/input_hashes.csv",
  "_codex_workspace/09_event_timing_sensitivity/primary_run/CIF_all_events_12_36_60_months.csv",
  "_codex_workspace/09_event_timing_sensitivity/primary_run/scenario_event_counts.csv")
paths <- file.path(root,rel);stopifnot(all(file.exists(paths)))
hash0 <- unname(tools::md5sum(paths))
previous_hashes <- fread(paths[3])
check("Same ML and master inputs as independently reconstructed prior run",sum(hash0[1:2]!=previous_hashes$md5_after[match(rel[1:2],previous_hashes$relative_path)]),all(hash0[1:2]==previous_hashes$md5_after[match(rel[1:2],previous_hashes$relative_path)]))
m <- fread(paths[1],colClasses=list(character="patient_id"))
a <- fread(paths[2],colClasses=list(character="patient_id"))
save(data.table(source=rel[1:2],n=c(nrow(m),nrow(a)),columns=c(ncol(m),ncol(a))),"input_dimensions")
check("ML unique patient IDs",anyDuplicated(m$patient_id),!anyDuplicated(m$patient_id))
check("Master unique patient IDs",anyDuplicated(a$patient_id),!anyDuplicated(a$patient_id))
a <- a[match(m$patient_id,a$patient_id)]
check("Master and ML time/event agreement",sum(m$time_months!=a$time_months|m$event_type!=a$event_type),all(m$time_months==a$time_months&m$event_type==a$event_type))
check("Formal cohort size",nrow(m),nrow(m)==85680)
check("Formal event counts",sum(abs(tabulate(m$event_type+1,4)-c(50794,7421,24303,3162))),all(tabulate(m$event_type+1,4)==c(50794,7421,24303,3162)))
check("Times are complete positive integer months",sum(!is.finite(m$time_months)|m$time_months<1|m$time_months!=floor(m$time_months)),all(is.finite(m$time_months)&m$time_months>=1&m$time_months==floor(m$time_months)))
check("Unchanged declared split seed",length(unique(m$split_seed)),all(m$split_seed==20260723))

# Given integer recorded months, time >=6 means event-free and observed beyond
# month 5. Early first events AND early censoring are excluded, not relabelled.
x <- copy(m[time_months>=6])
x[,elapsed:=time_months-5]
labs <- c("Censored","Non-haematological SPM","Death","Haematological/MPN competing event")
counts <- data.table(scenario="latency_ge6_conditional",n=nrow(x),excluded_from_formal=nrow(m)-nrow(x),
  event_type=0:3,disposition=labs,count=tabulate(x$event_type+1,4))
counts[,percent:=100*count/n]
save(counts,"scenario_event_counts")
excluded <- m[time_months<6,.(excluded_n=.N),by=.(event_type,time_months)][order(event_type,time_months)]
save(excluded,"exclusions_month1_to_month5")
save(excluded[,.(excluded_n=sum(excluded_n)),by=event_type],"exclusions_by_disposition")
save(x[,.(n=.N),by=.(data_split,event_type)],"retained_split_event_counts")
check("Retained plus excluded equals formal cohort",nrow(x)+sum(excluded$excluded_n),nrow(x)+sum(excluded$excluded_n)==nrow(m))
check("Conditional time starts at positive month",min(x$elapsed),min(x$elapsed)>=1)

ci_fun <- function(p,v) {
  if(p<=0) return(c(0,0));if(p>=1) return(c(1,1))
  eta<-log(-log1p(-p));se<-sqrt(v)/((1-p)*(-log1p(-p)))
  1-exp(-exp(eta+c(-1,1)*qnorm(.975)*se))
}
fit <- cmprsk::cuminc(x$elapsed,x$event_type,cencode=0)
aj <- survfit(Surv(elapsed,factor(event_type,levels=0:3))~1,data=x)
full_aj <- survfit(Surv(time_months,factor(event_type,levels=0:3))~1,data=m)
at <- function(obj,t,state) obj$pstate[findInterval(t,obj$time),match(state,obj$states)]
rows<-list();curves<-list()
for(k in 1:3) {
  cc<-fit[[paste("1",k)]]
  for(h in c(12,36,60)) {
    j<-max(which(cc$time<=h-5));p<-cc$est[j];lim<-ci_fun(p,cc$var[j])
    independent<-at(aj,h-5,as.character(k))
    conditional<-(at(full_aj,h,as.character(k))-at(full_aj,5,as.character(k)))/at(full_aj,5,"(s0)")
    check(paste("Independent AJ agreement",k,h),abs(p-independent),abs(p-independent)<1e-10)
    check(paste("Conditional probability identity",k,h),abs(p-conditional),abs(p-conditional)<1e-10)
    rows[[length(rows)+1]]<-data.table(scenario="latency_ge6_conditional",n=nrow(x),event_type=k,event=labs[k+1],
      entry_month=5,horizon_months_since_MPN=h,cif=p,ci_lower=lim[1],ci_upper=lim[2],
      at_risk_just_before_horizon=sum(x$time_months>=h))
  }
  curves[[k]]<-data.table(event_type=k,month=cc$time+5,cif=cc$est)
}
results<-rbindlist(rows);save(results,"CIF_all_events_12_36_60_months")
prior<-fread(paths[4])[scenario %in% c("primary_tumour_first","latency_ge3_conditional") & event_type==1]
comparison<-rbindlist(list(prior[,.(scenario,n,entry_month,horizon_months_since_MPN,cif,ci_lower,ci_upper)],
  results[event_type==1,.(scenario,n,entry_month,horizon_months_since_MPN,cif,ci_lower,ci_upper)]))
comparison[,estimand:=ifelse(entry_month==0,"Primary cohort from diagnosis","Conditional on being first-event-free and observed beyond entry month")]
save(comparison,"SPM_CIF_primary_ge3_ge6")
prior_counts<-fread(paths[5])[scenario %in% c("primary_tumour_first","latency_ge3_conditional")]
all_counts<-rbindlist(list(prior_counts,counts),use.names=TRUE)
compact<-rbindlist(lapply(unique(comparison$scenario),function(s){
  z<-comparison[scenario==s]; c<-all_counts[scenario==s]
  row<-data.table(Analysis=s,N=z$n[1],Entry_month=z$entry_month[1],SPM_events=c[event_type==1,count],
    Death_first_events=c[event_type==2,count],Haematological_first_events=c[event_type==3,count],Censored=c[event_type==0,count])
  for(h in c(12,36,60)) {
    v<-z[horizon_months_since_MPN==h]
    row[,paste0("SPM_CIF_",h,"m_percent_95CI"):=sprintf("%.2f (%.2f-%.2f)",100*v$cif,100*v$ci_lower,100*v$ci_upper)]
  }
  row
}))
save(compact,"Manuscript_Table_Latency_Comparison")
cr<-rbindlist(curves);save(cr,"CIF_curve_source_data")
p<-ggplot(cr[event_type==1 & month<=60],aes(month,100*cif))+geom_step(colour="#0072B2",linewidth=.8)+
  labs(x="Months since MPN diagnosis",y="Conditional SPM cumulative incidence (%)",
       title="Minimum observed event interval of 6 months",
       subtitle="Conditional on being event-free and still followed beyond month 5")+theme_classic(base_size=11)
ggsave(file.path(out,"Latency_ge6_conditional_CIF.pdf"),p,width=8,height=5,device=cairo_pdf)
ggsave(file.path(out,"Latency_ge6_conditional_CIF.png"),p,width=8,height=5,dpi=200)
hash1<-unname(tools::md5sum(paths))
check("Original inputs and prior outputs unchanged",sum(hash0!=hash1),identical(hash0,hash1))
save(data.table(relative_path=rel,md5_before=hash0,md5_after=hash1,unchanged=hash0==hash1),"input_hashes")
save(qc,"qc_checks")
writeLines(capture.output(sessionInfo()),file.path(out,"sessionInfo.txt"))
cat("\nEvents:\n");print(counts)
cat("\nExcluded:\n");print(excluded[,.(excluded_n=sum(excluded_n)),by=event_type])
cat("\nComparison:\n");print(compact)
cat("\nAll",nrow(qc),"QC checks passed. Elapsed seconds:",as.numeric(difftime(Sys.time(),start,units="secs")),"\n")
sink();close(log)
