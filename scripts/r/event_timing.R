# Curated package: never execute against the original project directory.
mpn_run_guard <- Sys.getenv("MPN_RUN_ROOT")
if (!nzchar(mpn_run_guard) || !file.exists(file.path(mpn_run_guard, ".mpn_reproduction_run")))
  stop("Use run_pipeline.py with a new marked run directory.")
if (normalizePath(Sys.getenv("MPN_PROJECT_ROOT"), mustWork=TRUE) != normalizePath(mpn_run_guard, mustWork=TRUE))
  stop("Project and run roots must match.")
mpn_args_guard <- commandArgs(trailingOnly=TRUE)
if (length(mpn_args_guard) && normalizePath(mpn_args_guard[1],mustWork=TRUE) != normalizePath(mpn_run_guard,mustWork=TRUE))
  stop("Argument root differs from protected run root.")
# User-authorised supplementary analyses; original files remain read-only.
# Run: Rscript.exe --vanilla run_supplementary_analyses.R [project_root] [new_run_name]
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[1] else Sys.getenv("MPN_PROJECT_ROOT")
run_name <- if (length(args) >= 2) args[2] else format(Sys.time(), "run_%Y%m%d_%H%M%S")
stopifnot(grepl("^[A-Za-z0-9_-]+$", run_name))
out <- file.path(root, "_codex_workspace/09_event_timing_sensitivity", run_name)
if (dir.exists(out)) stop("Refusing to overwrite an existing run directory")
dir.create(out, recursive = TRUE)
for (pkg in c("data.table", "survival", "cmprsk", "ggplot2")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Required installed package unavailable: ", pkg)
}
library(data.table)
library(survival)
library(ggplot2)
set.seed(20260723) # Deterministic estimators; no resampling/model fitting is performed.
log_con <- file(file.path(out, "execution_log.txt"), "wt")
sink(log_con, split = TRUE)
options(warn = 1)
cat("Started:", format(Sys.time()), "\nOutput:", out, "\n")
paths <- c(
  master = "MPN/patient_master_v0.4_2000_2023.csv",
  ml = "ML/data/ml_competing_risk_v0.4_2004_2023.csv",
  index = "MPN/SEER_MPN_raw_index_records_v0.4_85680.csv",
  raw = "SEER_MPN_first_primary_person_selection_2000_2023.txt",
  cif_reference = "ML/tables/source_data/Table_2_CIF_at_1_3_5_years.csv",
  python = "ML/script/scripts/python/01_build_datasets.py")
full <- file.path(root, paths)
stopifnot(all(file.exists(full)))
hash_before <- unname(tools::md5sum(full))
qc <- data.table(check = character(), value = numeric(), criterion = character(), passed = logical())
check <- function(label, value, ok, criterion = "zero discrepancies") {
  qc <<- rbind(qc, data.table(check=label, value=value, criterion=criterion, passed=ok))
  if (!isTRUE(ok)) { fwrite(qc, file.path(out,"qc_checks.csv")); stop(label, ": ", value) }
}
save_table <- function(x, name) fwrite(x, file.path(out, paste0(name,".csv")), bom=TRUE, na="NA")
ml <- fread(full[2], colClasses = list(character="patient_id"))
master <- fread(full[1], colClasses = list(character="patient_id"))
idx <- fread(full[3], colClasses = "character")
raw <- fread(full[4], sep="\t", colClasses="character", na.strings=NULL)
save_table(data.table(source=names(paths)[1:4], rows=c(nrow(master),nrow(ml),nrow(idx),nrow(raw)),
                     columns=c(ncol(master),ncol(ml),ncol(idx),ncol(raw))), "input_dimensions")
check("ML unique patient IDs", anyDuplicated(ml$patient_id), !anyDuplicated(ml$patient_id))
check("master unique patient IDs", anyDuplicated(master$patient_id), !anyDuplicated(master$patient_id))
check("index unique patient IDs", anyDuplicated(idx[["Patient ID"]]), !anyDuplicated(idx[["Patient ID"]]))
check("ML versus index patient set", length(setdiff(ml$patient_id,idx[["Patient ID"]]))+length(setdiff(idx[["Patient ID"]],ml$patient_id)), setequal(ml$patient_id,idx[["Patient ID"]]))
d <- master[match(ml$patient_id, master$patient_id)]
check("ML versus master event and time", sum(d$event_type!=ml$event_type | d$time_months!=ml$time_months), all(d$event_type==ml$event_type & d$time_months==ml$time_months))
d[, `:=`(os_months=as.numeric(index_survival_months_numeric),
         death_any=as.integer(trimws(get("index_Vital status recode (study cutoff used)"))=="Dead"),
         data_split=ml$data_split, subtype=ml$mpn_subtype)]
ii <- idx[match(d$patient_id,idx[["Patient ID"]])]
check("OS follow-up independently matches index export", sum(d$os_months != as.numeric(ii[["Survival months"]])), all(d$os_months==as.numeric(ii[["Survival months"]])))
check("OS death status independently matches index export", sum(d$death_any != as.integer(trimws(ii[["Vital status recode (study cutoff used)"]])=="Dead")), all(d$death_any==as.integer(trimws(ii[["Vital status recode (study cutoff used)"]])=="Dead")))
check("formal cohort size", nrow(d), nrow(d)==85680, "85680")
check("formal event counts 0/1/2/3", sum(abs(tabulate(d$event_type+1,4)-c(50794,7421,24303,3162))), identical(as.integer(tabulate(d$event_type+1,4)),c(50794L,7421L,24303L,3162L)))
check("OS deaths",sum(d$death_any),sum(d$death_any)==30526,"30526")
check("complete and ordered times",sum(!is.finite(d$os_months)|!is.finite(d$time_months)|d$time_months<1|d$os_months<d$time_months), all(is.finite(d$os_months)&is.finite(d$time_months)&d$time_months>=1&d$os_months>=d$time_months))
check("formal split seed",length(unique(ml$split_seed)),all(ml$split_seed==20260723),"all 20260723")
save_table(d[, .(n=.N), by=.(data_split,event_type)], "formal_split_event_counts")

# Independently reconstruct eligible later malignant records from tumour-level export.
rank_seq <- function(x) {
  z <- rep(NA_integer_,length(x)); y<-trimws(x)
  a<-grepl("^[0-9]+(st|nd|rd|th)",y,ignore.case=TRUE)
  z[a]<-as.integer(sub("^([0-9]+).*","\\1",y[a]))
  z[y %in% c("One primary only","Only one state registry-defined neoplasm")]<-1L
  z
}
num_month <- function(x) { z<-rep(NA_real_,length(x)); a<-grepl("^[0-9]+$",trimws(x)); z[a]<-as.numeric(trimws(x[a])); z }
r <- data.table(patient_id=raw[["Patient ID"]], record=raw[["Record number recode"]],
  seq_text=raw[["Sequence number"]], seq_rank=rank_seq(raw[["Sequence number"]]),
  year=as.integer(raw[["Year of diagnosis"]]), record_surv=num_month(raw[["Survival months"]]),
  hist=trimws(raw[["Histologic Type ICD-O-3"]]), hist_beh=raw[["ICD-O-3 Hist/behav"]],
  behaviour=tolower(trimws(raw[["Behavior code ICD-O-3"]])))
r[hist=="", hist:=sub("/.*$","",hist_beh)]
r[, hist_num:=suppressWarnings(as.integer(hist))]
lookup <- d[,.(patient_id,os_months,death_any,index_year,
  index_record=get("index_Record number recode"),index_rank=rank_seq(get("index_Sequence number")))]
check("formal index sequence rank",sum(lookup$index_rank!=1),all(lookup$index_rank==1))
r <- merge(r,lookup,by="patient_id",all=FALSE,sort=FALSE)
r[, delta:=os_months-record_surv]
cand <- r[record!=index_record & !is.na(seq_rank) & seq_rank>index_rank & year>=index_year &
          !is.na(delta) & delta>=1 & behaviour=="malignant"]
cand[, event:=fifelse(!is.na(hist_num)&hist_num>=9590&hist_num<=9999,3L,1L)]
# Python selects earliest within kind by time, year, original sequence label,
# then compares kind winners by time, numeric sequence rank, event code.
setorderv(cand,c("patient_id","event","delta","year","seq_text"))
first_kind<-cand[, .SD[1], by=.(patient_id,event)]
possible<-rbindlist(list(first_kind[,.(patient_id,t=delta,rank=seq_rank,event)],
                        d[death_any==1,.(patient_id,t=os_months,rank=999L,event=2L)]))
setorderv(possible,c("patient_id","t","rank","event"))
first<-possible[, .SD[1],by=patient_id]
rec<-merge(d[,.(patient_id,os_months,time_months,event_type)],first,by="patient_id",all.x=TRUE,sort=FALSE)
rec[is.na(event),`:=`(event=0L,t=os_months)]
check("raw tumour reconstruction versus formal first-event/time",sum(rec$event!=rec$event_type|rec$t!=rec$time_months),all(rec$event==rec$event_type&rec$t==rec$time_months))
tie_ids<-unique(cand[death_any==1&delta==os_months,patient_id])
d[, conflict:=event_type %in% c(1L,3L)&death_any==1&time_months==os_months]
check("effective tumour/death ties have raw-record evidence",sum(d$conflict & !d$patient_id %in% tie_ids),all(d$patient_id[d$conflict] %in% tie_ids))
broader_ties<-unique(r[record!=index_record & behaviour %in% c("malignant","in situ") &
                       delta>=1 & delta==os_months & death_any==1,patient_id])
tie_audit<-data.table(definition=c("first_event_tumour_death_tie_total","first_event_SPM_death_tie","first_event_haematological_death_tie",
  "any_eligible_later_malignant_tumour_death_tie","later_malignant_tie_after_an_earlier_first_event",
  "broader_malignant_or_in_situ_tie_matching_prior_tumour_count_audit"),
  n=c(sum(d$conflict),sum(d$conflict&d$event_type==1),sum(d$conflict&d$event_type==3),
      length(tie_ids),sum(d$patient_id %in% tie_ids & !d$conflict),length(broader_ties)))
save_table(tie_audit,"same_month_conflict_audit")
save_table(d[conflict==TRUE,.(n=.N),by=.(event_type,subtype)],"same_month_conflicts_by_type_subtype")

scenarios<-list(primary_tumour_first=copy(d),death_first=copy(d),exclude_first_event_ties=d[conflict==FALSE],
                latency_ge3_conditional=d[time_months>=3])
scenarios$death_first[conflict==TRUE,event_type:=2L]
event_labels<-c("Censored","Non-haematological SPM","Death","Haematological/MPN competing event")
counts<-rbindlist(lapply(names(scenarios),function(s){
  x<-scenarios[[s]]
  data.table(scenario=s,n=nrow(x),excluded_from_formal=nrow(d)-nrow(x),event_type=0:3,
    disposition=event_labels,count=tabulate(x$event_type+1,4),percent=100*tabulate(x$event_type+1,4)/nrow(x))
}))
save_table(counts,"scenario_event_counts")
save_table(d[time_months<3,.(excluded_n=.N),by=.(event_type, time_months)],"latency_exclusions_month1_month2")
check("latency excludes all early disposition types",nrow(scenarios$latency_ge3_conditional), nrow(scenarios$latency_ge3_conditional)+sum(d$time_months<3)==nrow(d),"retained plus excluded equals formal cohort")

# Aalen-Johansen / nonparametric competing-risk CIF; cmprsk variance with
# pointwise complementary-log-log 95% CIs. CI endpoints are not bootstrap CIs.
cif_ci<-function(p,v) {
  if(p<=0) return(c(0,0))
  if(p>=1) return(c(1,1))
  eta<-log(-log1p(-p)); se<-sqrt(v)/((1-p)*(-log1p(-p)))
  1-exp(-exp(eta+c(-1,1)*qnorm(.975)*se))
}
rows<-list(); curves<-list(); horizon<-c(12,36,60)
for(s in names(scenarios)) {
  x<-copy(scenarios[[s]])
  entry<-if(s=="latency_ge3_conditional") 2 else 0
  x[, elapsed:=time_months-entry]
  ci<-cmprsk::cuminc(x$elapsed,x$event_type,cencode=0)
  aj<-survfit(Surv(elapsed,factor(event_type,levels=0:3))~1,data=x)
  for(k in 1:3) {
    cc<-ci[[paste("1",k)]]
    for(h in horizon) {
      j<-max(which(cc$time<=h-entry)); p<-cc$est[j]; v<-cc$var[j]; lim<-cif_ci(p,v)
      jj<-findInterval(h-entry,aj$time)
      p2<-if(jj==0) 0 else aj$pstate[jj,match(as.character(k),aj$states)]
      check(paste("CIF independent AJ agreement",s,k,h),abs(p-p2),abs(p-p2)<1e-10,"absolute difference < 1e-10")
      rows[[length(rows)+1]]<-data.table(scenario=s,n=nrow(x),event_type=k,event=event_labels[k+1],
        entry_month=entry,horizon_months_since_MPN=h,cif=p,ci_lower=lim[1],ci_upper=lim[2],
        at_risk_just_before_horizon=sum(x$time_months>=h))
    }
    curves[[length(curves)+1]]<-data.table(scenario=s,event_type=k,month=cc$time+entry,cif=cc$est)
  }
}
cifs<-rbindlist(rows)
base<-cifs[scenario=="primary_tumour_first",.(event_type,horizon_months_since_MPN,primary_cif=cif)]
cifs<-merge(cifs,base,by=c("event_type","horizon_months_since_MPN"),all.x=TRUE,sort=FALSE)
cifs[,difference_percentage_points:=100*(cif-primary_cif)]
cifs[,comparability:=ifelse(scenario=="latency_ge3_conditional","Conditional estimand; descriptive contrast only","Alternative same-month rule")]
save_table(cifs,"CIF_all_events_12_36_60_months")
save_table(cifs[event_type==1],"CIF_SPM_comparison")
ref<-fread(full[5])[group=="Overall"]
cmp<-merge(cifs[scenario=="primary_tumour_first"],ref,by.x=c("event_type","horizon_months_since_MPN"),by.y=c("event_type","horizon_months"),suffixes=c("_R","_reference"))
cmp[,absolute_cif_difference:=abs(cif_R-cif_reference)]
check("all nine original overall CIF point estimates reproduced",max(cmp$absolute_cif_difference),nrow(cmp)==9 & max(cmp$absolute_cif_difference)<1e-10,"9 rows, absolute difference < 1e-10")
save_table(cmp[,.(event_type,horizon_months_since_MPN,cif_R,cif_reference,absolute_cif_difference)],"original_CIF_replication")

# Reverse KM: observation-ending censoring is the reverse event; death (OS)
# or any first event (first-event follow-up) is censored in this estimator.
follow<-list()
groups<-c(list(Overall=d),lapply(split(d,d$data_split),identity),lapply(split(d,d$subtype),identity))
for(g in names(groups)) for(endpoint in c("OS","First_event")) {
  x<-groups[[g]]
  tm<-if(endpoint=="OS") x$os_months else x$time_months
  rev_status<-if(endpoint=="OS") 1L-x$death_any else as.integer(x$event_type==0)
  fit<-survfit(Surv(tm,rev_status)~1,conf.type="log-log")
  tt<-summary(fit)$table
  follow[[length(follow)+1]]<-data.table(population=g,followup_definition=endpoint,n=nrow(x),
    reverse_events=sum(rev_status),reverse_censored=sum(1-rev_status),
    median_months=unname(tt["median"]),ci_lower_months=unname(tt["0.95LCL"]),ci_upper_months=unname(tt["0.95UCL"]),
    median_years=unname(tt["median"])/12,maximum_observed_months=max(tm))
}
follow<-rbindlist(follow)
save_table(follow,"reverse_KM_followup")

# Figures deliberately separate unconditional same-month analyses from the
# two-month event-free conditional latency analysis (different estimand).
cr<-rbindlist(curves)
labels<-c(primary_tumour_first="Primary: tumour first",death_first="Death first",
          exclude_first_event_ties="Exclude first-event ties",latency_ge3_conditional="Minimum interval 3 months (conditional)")
pp<-ggplot(cr[event_type==1 & scenario!="latency_ge3_conditional" & month<=60],
           aes(month,100*cif,colour=scenario,linetype=scenario))+geom_step(linewidth=.75)+
  scale_colour_manual(values=c(primary_tumour_first="#222222",death_first="#D55E00",exclude_first_event_ties="#0072B2"),labels=labels)+
  scale_linetype_manual(values=c(primary_tumour_first="solid",death_first="dashed",exclude_first_event_ties="dotdash"),labels=labels)+
  labs(x="Months since MPN diagnosis",y="Non-haematological SPM cumulative incidence (%)",colour=NULL,linetype=NULL,
       title="Sensitivity to same-month tumour/death ordering")+theme_classic(base_size=11)+theme(legend.position="bottom")
qq<-ggplot(cr[event_type==1 & scenario=="latency_ge3_conditional" & month<=60],aes(month,100*cif))+
  geom_step(colour="#0072B2",linewidth=.8)+labs(x="Months since MPN diagnosis",y="Conditional SPM cumulative incidence (%)",
  title="Minimum observed event interval of 3 months",subtitle="Conditional on being event-free and still followed beyond month 2")+
  theme_classic(base_size=11)
for(nm in c("Same_month_CIF_sensitivity","Latency_ge3_conditional_CIF")) {
  p<-if(nm=="Same_month_CIF_sensitivity") pp else qq
  ggsave(file.path(out,paste0(nm,".pdf")),p,width=8,height=5,device=cairo_pdf)
  ggsave(file.path(out,paste0(nm,".png")),p,width=8,height=5,dpi=200)
}
save_table(cr,"CIF_curve_source_data")
hash_after<-unname(tools::md5sum(full))
check("source files unchanged",sum(hash_before!=hash_after),identical(hash_before,hash_after))
save_table(data.table(relative_path=paths,md5_before=hash_before,md5_after=hash_after,unchanged=hash_before==hash_after),"input_hashes")
save_table(qc,"qc_checks")
writeLines(capture.output(sessionInfo()),file.path(out,"sessionInfo.txt"))
cat("\nConflict audit:\n");print(tie_audit)
cat("\nEvent counts:\n");print(counts)
cat("\nSPM CIF comparison:\n");print(cifs[event_type==1,.(scenario,horizon_months_since_MPN,cif,ci_lower,ci_upper,difference_percentage_points)])
cat("\nReverse KM overall:\n");print(follow[population=="Overall"])
cat("\nAll QC checks passed:",nrow(qc),"\nCompleted:",format(Sys.time()),"\n")
sink();close(log_con)
