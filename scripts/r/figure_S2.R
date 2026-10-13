# Curated package: never execute against the original project directory.
mpn_run_guard <- Sys.getenv("MPN_RUN_ROOT")
if (!nzchar(mpn_run_guard) || !file.exists(file.path(mpn_run_guard, ".mpn_reproduction_run")))
  stop("Use run_pipeline.py with a new marked run directory.")
if (normalizePath(Sys.getenv("MPN_PROJECT_ROOT"), mustWork=TRUE) != normalizePath(mpn_run_guard, mustWork=TRUE))
  stop("Project and run roots must match.")
mpn_args_guard <- commandArgs(trailingOnly=TRUE)
if (length(mpn_args_guard) && normalizePath(mpn_args_guard[1],mustWork=TRUE) != normalizePath(mpn_run_guard,mustWork=TRUE))
  stop("Argument root differs from protected run root.")
#!/usr/bin/env Rscript
# User-authorized eight-panel KM redraw. No Cox/ML analyses or source edits.
options(stringsAsFactors = FALSE, warn = 1)
suppressPackageStartupMessages({
  library(data.table); library(dplyr); library(stringr)
  library(survival); library(survminer); library(ggplot2); library(cowplot)
})
root <- Sys.getenv("MPN_PROJECT_ROOT")
stopifnot(nzchar(root))
out <- file.path(root, '_codex_workspace/28_Figure_S2_all_KM_redraw_20261002/ready')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
reference <- file.path(root, '_codex_workspace/25_income_reanalysis_20261001/traditional_outputs/tables')
input <- file.path(root, 'MPN/SEER_MPN_raw_index_records_v0.4_85680.csv')
source(file.path(root, '_codex_workspace/21_sequence_income_integrity_20261001/income_recode_fix.R'))
hash_before <- unname(tools::md5sum(input))
raw <- fread(input, data.table = FALSE, colClasses = list(character = 'Patient ID'))
stopifnot(nrow(raw) == 85680L, !anyDuplicated(raw[['Patient ID']]))
age <- as.numeric(str_extract(raw[['Age recode with single ages and 85+']], '[0-9]+'))
year <- as.integer(raw[['Year of diagnosis']])
income <- parse_income(raw[['Median household income inflation adj to 2024']])
urban <- str_squish(str_to_lower(raw[['Rural-Urban Continuum Code']]))
chemo <- str_squish(str_to_lower(raw[['Chemotherapy recode (yes, no/unk) (2004+)']]))
stopifnot(all(chemo %in% c('yes', 'no/unknown', 'no/unk')))
d <- data.frame(
  os_months = as.numeric(raw[['Survival months']]),
  death_any = as.integer(raw[['Vital status recode (study cutoff used)']] == 'Dead'),
  age_grp = factor(case_when(age < 50 ~ '<50', age < 65 ~ '50-64', age < 75 ~ '65-74',
    age < 85 ~ '75-84', !is.na(age) ~ '85+'), levels = c('<50','50-64','65-74','75-84','85+')),
  sex = factor(raw[['Sex']], levels = c('Female','Male')),
  race = factor(raw[['Race recode (W, B, AI, API)']],
    levels = c('American Indian/Alaska Native','Asian or Pacific Islander','Black','Unknown','White')),
  mpn_subtype = factor(recode(str_extract(raw[['ICD-O-3 Hist/behav']], '^(9950|9961|9962)'),
    `9950`='PV', `9962`='ET', `9961`='PMF'), levels = c('PV','ET','PMF')),
  income_grp = factor(case_when(is.na(income) ~ 'Unknown', income < 60000 ~ '<$60,000',
    income < 75000 ~ '$60,000-$74,999', income < 90000 ~ '$75,000-$89,999', TRUE ~ '>=$90,000'),
    levels = c('<$60,000','$60,000-$74,999','$75,000-$89,999','>=$90,000','Unknown')),
  year_grp = factor(case_when(year <= 2008 ~ '2004-2008', year <= 2013 ~ '2009-2013',
    year <= 2018 ~ '2014-2018', TRUE ~ '2019-2023'),
    levels = c('2004-2008','2009-2013','2014-2018','2019-2023')),
  urban_grp = factor(case_when(str_detect(urban, 'nonmetropolitan') ~ 'Nonmetropolitan',
    str_detect(urban, 'metropolitan') ~ 'Metropolitan', TRUE ~ 'Unknown'),
    levels = c('Metropolitan','Nonmetropolitan','Unknown')),
  chemo_seer = factor(ifelse(chemo == 'yes','Yes','No/Unknown'), levels = c('No/Unknown','Yes'))
)
stopifnot(all(complete.cases(d)), all(year >= 2004 & year <= 2023), all(d$os_months >= 1),
  sum(d$death_any) == 30526L, sum(d$income_grp == 'Unknown') == 83L,
  sum(d$income_grp == '<$60,000') == 11569L)
rm(raw, age, year, income, urban, chemo)
variables <- c('age_grp','sex','race','mpn_subtype','income_grp','year_grp','urban_grp','chemo_seer')
titles <- c('Overall survival by age group','Overall survival by sex','Overall survival by race',
  'Overall survival by MPN subtype','Overall survival by area-level median household income',
  'Overall survival by diagnosis era','Overall survival by rural/urban residence',
  'Overall survival by initial chemotherapy')
display <- list(
  c('<50','50-64','65-74','75-84','>=85'), c('Female','Male'),
  c('AI/AN','Asian/Pacific Islander','Black','Unknown','White'), c('PV','ET','PMF'),
  c('<$60,000','$60,000-$74,999','$75,000-$89,999','>=$90,000','Unknown'),
  c('2004-2008','2009-2013','2014-2018','2019-2023'),
  c('Metropolitan','Nonmetropolitan','Unknown'), c('No/Unknown','Yes'))
palettes <- list(
  c('#F8766D','#A3A500','#00BF7D','#00B0F6','#E76BF3'), c('#F8766D','#00BFC4'),
  c('#F8766D','#A3A500','#00BF7D','#00B0F6','#E76BF3'), c('#F8766D','#00BA38','#619CFF'),
  c('#F8766D','#7CAE00','#00BFC4','#C77CFF','#619CFF'),
  c('#F8766D','#7CAE00','#00BFC4','#C77CFF'), c('#F8766D','#00BA38','#619CFF'),
  c('#F8766D','#00BFC4'))
ref_med <- fread(file.path(reference, 'KM_OS_corrected_medians.csv'), data.table = FALSE)
ref_global <- fread(file.path(reference, 'KM_OS_corrected_overall_logrank.csv'), data.table = FALSE)
ref_pw <- fread(file.path(reference, 'KM_OS_corrected_pairwise_Holm.csv'), data.table = FALSE)
font <- 'Times New Roman'
do.call(pdfFonts, setNames(list(pdfFonts('Times')[[1]]), font))
png(file.path(out,'layout_measurement.png'), width = 6600, height = 3360, res = 300, type = 'cairo', family = font)
lines_mm <- 0.28
base_theme <- theme_bw(base_size = 9, base_family = font) + theme(
  text = element_text(family = font, face = 'plain'),
  panel.grid.major = element_line(colour = 'grey92', linewidth = 0.15),
  panel.grid.minor = element_line(colour = 'grey95', linewidth = 0.08),
  panel.border = element_rect(fill = NA, colour = 'grey35', linewidth = 0.18),
  axis.ticks = element_line(linewidth = 0.18),
  axis.text = element_text(size = 8, colour = 'black'),
  axis.title = element_text(size = 9), legend.position = 'top',
  legend.text = element_text(size = 7), legend.title = element_blank(),
  legend.key.width = grid::unit(0.65,'lines'), legend.key.height = grid::unit(0.65,'lines'),
  legend.spacing.x = grid::unit(0.08,'cm'), plot.margin = margin(3,16,3,3))
main_plots <- risk_plots <- headers <- notes <- list()
all_risk <- all_curves <- all_medians <- all_global <- all_pairs <- checks <- list()
fmt <- function(p) ifelse(p < 1e-4,'<0.0001',format.pval(p,digits=4,eps=1e-4))
equal_num <- function(x,y) isTRUE(all.equal(as.numeric(x),as.numeric(y),tolerance=1e-10))
for (i in seq_along(variables)) {
  var <- variables[i]; lev <- levels(d[[var]]); cols <- setNames(palettes[[i]], lev)
  lab <- setNames(display[[i]], lev)
  f <- reformulate(var, response='Surv(os_months, death_any)')
  fit <- surv_fit(f, data=d); lr <- survdiff(f, data=d)
  p <- pchisq(lr$chisq, length(lr$n)-1L, lower.tail=FALSE)
  old_g <- ref_global[ref_global$variable == var,]
  stopifnot(equal_num(lr$chisq,old_g$chisq),equal_num(p,old_g$p_value))
  med <- as.data.frame(summary(fit)$table)
  med$group <- sub('^[^=]+=','',rownames(med)); med$variable <- var
  for (j in seq_len(nrow(med))) {
    old <- ref_med[ref_med$variable == var & ref_med$group == med$group[j],]
    stopifnot(nrow(old)==1,old$n==med$records[j],old$events==med$events[j],
      equal_num(med$median[j],old$median_months),equal_num(med$`0.95LCL`[j],old$lower_95),
      equal_num(med$`0.95UCL`[j],old$upper_95))
  }
  pw <- data.frame()
  if(length(lev)>=3) {
    mat <- pairwise_survdiff(f,data=d,p.adjust.method='holm')$p.value
    pw <- as.data.frame(as.table(mat)); names(pw) <- c('group_1','group_2','p_holm')
    pw <- pw[!is.na(pw$p_holm),]; pw$group_1 <- as.character(pw$group_1); pw$group_2 <- as.character(pw$group_2)
    key <- function(a,b) paste(pmin(a,b),pmax(a,b),sep=' | ')
    old <- ref_pw[ref_pw$variable==var,]
    old <- old[match(key(pw$group_1,pw$group_2),key(old$group_1,old$group_2)),]
    stopifnot(nrow(old)==nrow(pw),equal_num(pw$p_holm,old$p_holm))
    pw$variable <- var
  }
  s <- summary(fit,censored=TRUE)
  curve <- data.frame(group=sub('^[^=]+=','',s$strata),time=s$time,survival=100*s$surv,n_censor=s$n.censor)
  curve <- bind_rows(data.frame(group=lev,time=0,survival=100,n_censor=0),curve)
  curve$group <- factor(curve$group,levels=lev)
  # Clip at 120 months without extending groups beyond their observed follow-up.
  horizon <- summary(fit,times=120,extend=FALSE)
  if(length(horizon$time)) curve <- bind_rows(curve,data.frame(
    group=factor(sub('^[^=]+=','',horizon$strata),levels=lev),time=120,
    survival=100*horizon$surv,n_censor=0))
  curve <- curve[curve$time<=120,]; curve <- curve[order(curve$group,curve$time),]
  main <- ggplot(curve,aes(time,survival,colour=group,group=group)) +
    geom_step(linewidth=lines_mm) +
    geom_point(data=curve[curve$n_censor>0,],shape=3,size=0.8,stroke=0.20,show.legend=FALSE) +
    scale_colour_manual(values=cols,breaks=lev,labels=lab) +
    scale_x_continuous(breaks=seq(0,120,24),minor_breaks=seq(0,120,12),limits=c(0,120),expand=expansion(mult=c(.015,.015))) +
    scale_y_continuous(breaks=seq(0,100,25),minor_breaks=seq(0,100,12.5),limits=c(-5,105),expand=c(0,0)) +
    labs(x='Months since index MPN diagnosis',y='Overall survival (%)') +
    annotate('text',x=4,y=1,label=paste0('Overall log-rank P ',ifelse(p<1e-4,'< 0.0001',paste('=',fmt(p)))),
             hjust=0,vjust=0,size=2.7,family=font,colour='black') + base_theme +
    guides(colour=guide_legend(nrow=2,byrow=TRUE,override.aes=list(linewidth=.28)))
  legend <- get_legend(main)
  header <- plot_grid(ggdraw()+draw_label(titles[i],x=.52,size=10,fontfamily=font,fontface='plain'),
                      legend,ncol=1,rel_heights=c(.42,.58))
  main <- main+theme(legend.position='none')
  r <- summary(fit,times=seq(0,120,24),extend=TRUE)
  risk <- data.frame(group=sub('^[^=]+=','',r$strata),months=r$time,n_risk=r$n.risk)
  risk$y <- length(lev)+1-match(risk$group,lev)
  risk$hjust <- ifelse(risk$months==0,0,ifelse(risk$months==120,1,.5))
  risk_plot <- ggplot(risk,aes(months,y))+geom_text(aes(label=n_risk,hjust=hjust),size=2.75,family=font) +
    scale_x_continuous(breaks=seq(0,120,24),minor_breaks=seq(0,120,12),limits=c(0,120),expand=expansion(mult=c(.015,.015))) +
    scale_y_continuous(breaks=seq_along(lev),labels=rev(unname(lab)),limits=c(.5,length(lev)+.5),expand=c(0,0)) +
    labs(x=NULL,y=NULL,title='Number at risk') + base_theme + coord_cartesian(clip='off') +
    theme(axis.text.x=element_blank(),axis.ticks=element_blank(),axis.text.y=element_text(size=7,margin=margin(r=18)),
      plot.title=element_text(size=8.5),plot.margin=margin(2,16,2,3),panel.grid.minor=element_blank())
  note <- if(nrow(pw)) paste0('Pairwise log-rank (Holm-adjusted): ',
    paste0(lab[pw$group_1],' vs ',lab[pw$group_2],': ',fmt(pw$p_holm),collapse='; ')) else ''
  if(var=='race') note <- paste0(note,'\nAI/AN, American Indian/Alaska Native.')
  wrapped <- unlist(lapply(strsplit(note,'\n',fixed=TRUE)[[1]],strwrap,width=91))
  note_plot <- ggdraw()+draw_label(paste(wrapped,collapse='\n'),x=.04,y=.94,hjust=0,vjust=1,
    size=6.0,fontfamily=font,lineheight=1.08)
  main_plots[[i]] <- main; risk_plots[[i]] <- risk_plot; headers[[i]] <- header; notes[[i]] <- note_plot
  all_risk[[i]] <- transform(risk,variable=var); all_curves[[i]] <- transform(curve,variable=var)
  all_medians[[i]] <- med; all_global[[i]] <- data.frame(variable=var,n=nrow(d),deaths=sum(d$death_any),chisq=lr$chisq,p_value=p)
  if(nrow(pw)) all_pairs[[var]] <- pw
  checks[[i]] <- data.frame(panel=LETTERS[i],variable=var,groups=length(lev),n=nrow(d),
    counts_and_medians='PASS',global_logrank='PASS',pairwise_Holm=ifelse(length(lev)>=3,'PASS','Not applicable'))
  cat('Verified panel',LETTERS[i],var,'\n')
}
# Align all main charts and risk tables together: identical coordinate-frame edges.
aligned <- align_plots(plotlist=c(main_plots,risk_plots),align='v',axis='lr')
panels <- lapply(seq_along(variables),function(i) plot_grid(headers[[i]],aligned[[i]],
    aligned[[i+8]],notes[[i]],ncol=1,rel_heights=c(.13,.43,.25,.19)))
figure <- plot_grid(plotlist=panels,ncol=4,nrow=2,labels=LETTERS[1:8],
  label_size=14,label_fontfamily=font,label_fontface='bold',label_x=.005,label_y=.99,hjust=0,vjust=1)
pdf_path <- file.path(out,'Figure_S2_KM_all_eight_corrected.pdf')
png_path <- file.path(out,'Figure_S2_KM_all_eight_corrected.png')
jpg_path <- file.path(out,'Figure_S2_KM_all_eight_corrected.jpg')
stopifnot(!any(file.exists(c(pdf_path,png_path,jpg_path))))
ggsave(pdf_path,figure,device=cairo_pdf,family=font,width=22,height=11.2,bg='white',limitsize=FALSE)
ggsave(png_path,figure,device='png',type='cairo',width=22,height=11.2,dpi=300,bg='white',limitsize=FALSE)
ggsave(jpg_path,figure,device='jpeg',type='cairo',quality=100,width=22,height=11.2,dpi=300,bg='white',limitsize=FALSE)
dev.off()
fwrite(bind_rows(checks),file.path(out,'KM_reproduction_QA.csv'))
fwrite(bind_rows(all_risk)[,c('variable','group','months','n_risk')],file.path(out,'KM_number_at_risk.csv'))
fwrite(bind_rows(all_curves),file.path(out,'KM_curve_aggregate_source.csv'))
fwrite(bind_rows(all_global),file.path(out,'KM_overall_logrank.csv'))
fwrite(bind_rows(all_medians),file.path(out,'KM_medians.csv'))
fwrite(bind_rows(all_pairs),file.path(out,'KM_pairwise_Holm.csv'))
stopifnot(identical(hash_before,unname(tools::md5sum(input))))
fwrite(data.frame(input=input,md5_before=hash_before,md5_after=unname(tools::md5sum(input)),unchanged=TRUE),
  file.path(out,'input_preservation.csv'))
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
cat('PASS: eight-panel Figure S2 generated in R; all corrected KM summaries reproduced.\n')
