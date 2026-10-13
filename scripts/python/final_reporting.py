"""Export V4 analysis tables/figures from NEW run outputs only.

These are scientific source tables and regenerated plots, not a byte-identical
recreation of manually formatted Word/Illustrator submission files.
"""
from pathlib import Path
import argparse, csv, json, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from runtime_paths import work_root
import publication_functions as pub

ROOT=work_root()
OUT=ROOT/'final_results'
BOOT=ROOT/'revision/04_bootstrap5000/model_root/final_test_evaluation'
MODEL=ROOT/'output/models/model_development_v0.4'
SOURCE=ROOT/'deliverables/v0.4/tables/source_data'
TRAD=ROOT/'_codex_workspace/25_income_reanalysis_20261001/traditional_outputs'
REP=ROOT/'_codex_workspace/25_income_reanalysis_20261001/replacement_files'
EVENT=ROOT/'_codex_workspace/09_event_timing_sensitivity/primary_run'
LAT6=ROOT/'_codex_workspace/09_event_timing_sensitivity/latency6_run'
REV=ROOT/'revision'
CATALOG=[]

def emit(src,table,section):
    if not src.is_file():raise FileNotFoundError(f'Missing required result: {src.name}')
    dst=OUT/'tables'/table/(section+'.csv');dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    CATALOG.append({'item':table,'section':section,'source':str(src.relative_to(ROOT)),'output':str(dst.relative_to(ROOT))})

def complete_test_curve(metric, value, bootstrap_intervals):
    """Join monthly locked-test estimates to patient-bootstrap confidence limits."""
    points=pd.read_csv(MODEL/'final_test_evaluation'/f'test_{metric}_long.csv')[["model","times",value]]
    intervals=bootstrap_intervals[["model","times","lower","upper"]]
    curve=points.merge(intervals,on=["model","times"],how="inner",validate="one_to_one")
    for model in pub.SELECTED_MODELS:
        part=curve.loc[curve.model.eq(model)].sort_values("times")
        if len(part)!=49 or not np.array_equal(part.times.to_numpy(),np.arange(12,61)):
            raise ValueError(f'Incomplete 12–60-month {metric} curve for {model}')
        if part[[value,"lower","upper"]].isna().any().any() or (part.lower>part.upper).any():
            raise ValueError(f'Missing or invalid {metric} estimate/interval for {model}')
        sparse=bootstrap_intervals.loc[
            bootstrap_intervals.model.eq(model)&bootstrap_intervals[value].notna(),["times",value]
        ]
        check=part.loc[part.times.isin([12,36,60]),["times",value]].merge(
            sparse,on="times",suffixes=("_full","_bootstrap"),validate="one_to_one"
        )
        if len(check)!=3 or not np.allclose(
            check[f'{value}_full'],check[f'{value}_bootstrap'],rtol=0,atol=1e-12
        ):
            raise ValueError(f'Locked-test and bootstrap point estimates disagree for {model} {metric}')
    return curve

def figure4(frames,output_pdf=None):
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    perf=frames['performance'].set_index('model')
    curves={key:complete_test_curve(key,value,frames[key]) for key,value in [('auc','AUC'),('brier','Brier')]}
    for ax,metric,label in [(axes[0,0],'iAUC','A'),(axes[0,1],'IBS','B')]:
        for y,model in enumerate(reversed(pub.MODEL_ORDER)):
            row=perf.loc[model];v=row[metric];lo=row[metric+'_lower'];hi=row[metric+'_upper']
            ax.errorbar(v,y,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',capsize=2,color=pub.COLORS.get(model,'#777777'))
        ax.set_yticks(range(len(pub.MODEL_ORDER)),[pub.MODEL_LABELS[m] for m in reversed(pub.MODEL_ORDER)])
        ax.set_xlabel(metric+' (95% bootstrap CI)');pub.panel_label(ax,label)
    for ax,key,value,label in [(axes[1,0],'auc','AUC','C'),(axes[1,1],'brier','Brier','D')]:
        for model in pub.SELECTED_MODELS:
            z=curves[key].loc[curves[key].model.eq(model)].sort_values('times')
            ax.fill_between(z.times,z.lower,z.upper,alpha=.08,color=pub.COLORS[model],zorder=1)
            ax.plot(z.times,z[value],label=pub.MODEL_LABELS[model],color=pub.COLORS[model],
                    ls=pub.LINESTYLES[model],lw=1.7,zorder=3)
        ax.set(xlabel='Prediction horizon (months)',ylabel=value,xlim=(12,60));pub.panel_label(ax,label)
    axes[1,0].legend(fontsize=7);fig.tight_layout()
    if output_pdf is None:
        pub.save_figure(fig,OUT/'figures','Figure_4_locked_test_performance')
    else:
        output_pdf=Path(output_pdf)
        if output_pdf.exists():
            raise FileExistsError(f'Refusing to overwrite an existing Figure 4 PDF: {output_pdf}')
        output_pdf.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output_pdf,bbox_inches='tight')
        plt.close(fig)

def figure2(frames):
    curves=pd.read_csv(ROOT/'deliverables/v0.4/figures/source_data/Figure_2_and_S1_CIF_source_data.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    names={1:'Non-haematological SPM',2:'Death',3:'Haematological/MPN competing event'}
    for event,color in [(1,'#005A9C'),(2,'#777777'),(3,'#7A5195')]:
        z=curves.loc[curves.group.eq('Overall') & curves.event_type.eq(event)]
        axes[0].step(z.time_months,100*z.cif,where='post',label=names[event],color=color)
    for subtype in ['PV','ET','PMF']:
        z=curves.loc[curves.group.eq(subtype)&curves.event_type.eq(1)]
        axes[1].step(z.time_months,100*z.cif,where='post',label=subtype,color=pub.COLORS[subtype])
        axes[1].fill_between(z.time_months,100*z.ci_lower,100*z.ci_upper,step='post',alpha=.1,color=pub.COLORS[subtype])
        for horizon,shift,marker in [(12,-.16,'o'),(36,0,'s'),(60,.16,'D')]:
            r=frames['cif_summary'].loc[lambda x:x.group.eq(subtype)&x.event_type.eq(1)&x.horizon_months.eq(horizon)].iloc[0]
            axes[2].errorbar(100*r.cif,['PMF','ET','PV'].index(subtype)+shift,
                xerr=[[max(0,100*(r.cif-r.ci_lower))],[max(0,100*(r.ci_upper-r.cif))]],fmt=marker,
                color=pub.COLORS[subtype],capsize=2,label=f'{horizon//12} year' if subtype=='PV' else None)
    for ax,label in zip(axes,['A','B','C']):pub.panel_label(ax,label)
    for ax in axes[:2]:ax.set(xlim=(0,120),xlabel='Months since MPN diagnosis',ylabel='Cumulative incidence (%)');ax.legend(fontsize=7)
    axes[2].set_yticks([0,1,2],['PMF','ET','PV']);axes[2].set_xlabel('SPM cumulative incidence (%, 95% CI)');axes[2].legend(fontsize=7)
    fig.tight_layout();pub.save_figure(fig,OUT/'figures','Figure_2_CIF')

def figureS5(frames,output_pdf=None):
    """Plot local-rerun sensitivity and locked-subgroup point estimates."""
    sensitivity_order=[
      'no_chemotherapy_2004_2023',
      'no_chemotherapy_2000_2023',
      'temporal_2004_2015_to_2016_2018']
    sensitivity_labels=[
      '2004–2023, omit chemotherapy',
      '2000–2023, omit chemotherapy',
      'Temporal: 2004–2015 → 2016–2018']
    subtype_order=['locked_global_model_PV','locked_global_model_ET','locked_global_model_PMF']
    sensitivity=frames['sensitivity']
    subtype=frames['subtype']
    if (len(sensitivity)!=3 or set(sensitivity.analysis)!=set(sensitivity_order)
            or sensitivity.analysis.duplicated().any()):
        raise ValueError('Figure S5 sensitivity scenarios do not match the three expected analyses')
    if (len(subtype)!=3 or set(subtype.analysis)!=set(subtype_order)
            or subtype.analysis.duplicated().any()):
        raise ValueError('Figure S5 subtype rows do not match PV, ET, and PMF')
    sensitivity=sensitivity.set_index('analysis').loc[sensitivity_order]
    subtype=subtype.set_index('analysis').loc[subtype_order]
    if sensitivity[['iAUC','IBS']].isna().any().any() or subtype[['iAUC','IBS']].isna().any().any():
        raise ValueError('Figure S5 has a missing performance estimate')
    if int(subtype.n_test.sum())!=int(sensitivity.iloc[0].n_test) or int(subtype.events_test.sum())!=int(sensitivity.iloc[0].events_test):
        raise ValueError('Figure S5 subtype sample/event counts do not sum to the primary test set')
    performance=frames['performance']
    primary=performance.loc[performance.model.eq('xgboost_cox')]
    if len(primary)!=1 or primary[['iAUC','IBS']].isna().any().any():
        raise ValueError('Figure S5 requires one primary XGBoost–Cox locked-test estimate')
    primary=primary.iloc[0]

    fig,axes=plt.subplots(2,2,figsize=(12.5,7.5))
    y=np.arange(3)[::-1]
    for col,metric in enumerate(['iAUC','IBS']):
        ax=axes[0,col]
        ax.scatter(sensitivity[metric].to_numpy(),y,color=pub.COLORS['xgboost_cox'],s=38,zorder=3)
        ax.axvline(primary[metric],color=pub.COLORS['regularized_ridge_meta'],ls='--',lw=1.2,
                   label='Primary locked-test estimate',zorder=2)
        ax.set_yticks(y,sensitivity_labels)
        ax.set_ylim(-0.35,2.55)
        ax.set_xlabel(metric+(' (lower is better)' if metric=='IBS' else ''))
        ax.grid(axis='x',color='#E3E3E3',lw=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.07)
        ax.legend(loc='upper left' if col==0 else 'upper right',frameon=False,fontsize=7)
        pub.panel_label(ax,'AB'[col])

    subtype_labels=[
      f'{name} (n={int(row.n_test):,}; events={int(row.events_test):,})'
      for name,row in zip(['PV','ET','PMF'],subtype.itertuples())]
    subtype_colors=[pub.COLORS[name] for name in ['PV','ET','PMF']]
    for col,metric in enumerate(['iAUC','IBS']):
        ax=axes[1,col]
        ax.scatter(subtype[metric].to_numpy(),y,c=subtype_colors,s=42,zorder=3)
        ax.set_yticks(y,subtype_labels)
        ax.set_ylim(-0.35,2.55)
        ax.set_xlabel(metric+(' (lower is better)' if metric=='IBS' else ''))
        ax.grid(axis='x',color='#E3E3E3',lw=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.07)
        pub.panel_label(ax,'CD'[col])

    fig.tight_layout(pad=2.0,w_pad=3.0,h_pad=2.5)
    if output_pdf is None:
        pub.save_figure(fig,OUT/'figures','Figure_S5_sensitivity_and_subgroups')
    else:
        output_pdf=Path(output_pdf)
        if output_pdf.exists():
            raise FileExistsError(f'Refusing to overwrite an existing Figure S5 PDF: {output_pdf}')
        output_pdf.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output_pdf,bbox_inches='tight')
        plt.close(fig)

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    for d in [pub.MAIN_FIG,pub.SUPP_FIG,pub.FIG_SOURCE,pub.SUPP_TABLE]:d.mkdir(parents=True,exist_ok=True)
    frames={
      'performance':pd.read_csv(BOOT/'table_test_main_comparison_bootstrap5000.csv'),
      'auc':pd.read_csv(BOOT/'test_auc_bootstrap5000.csv'),'brier':pd.read_csv(BOOT/'test_brier_bootstrap5000.csv'),
      'calibration':pd.read_csv(MODEL/'final_test_evaluation/test_calibration_summary.csv'),
      'calibration_bins':pd.read_csv(MODEL/'final_test_evaluation/test_calibration_bins.csv'),
      'dca':pd.read_csv(MODEL/'final_test_evaluation/test_dca_long.csv'),
      'sensitivity':pd.read_csv(SOURCE/'Table_S3_primary_model_sensitivity.csv'),
      'subtype':pd.read_csv(SOURCE/'Table_S4_locked_model_by_MPN_subtype.csv'),
      'cif_summary':pd.read_csv(SOURCE/'Table_2_CIF_at_1_3_5_years.csv'),
      'gray':pd.read_csv(SOURCE/'Table_S1_subtype_Gray_tests.csv')}
    data=pd.read_csv(ROOT/'data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv')
    pub.setup_style();pub.figure1(data);figure2(frames);figure4(frames);pub.figure4(frames)
    curve,summary,tests=pub.risk_stratification(data)
    summary.to_csv(OUT/'risk_strata_summary.csv',index=False);tests.to_csv(OUT/'risk_strata_Gray.csv',index=False)
    pub.figure5(frames,curve,summary,tests);figureS5(frames)
    emit(REP/'Table1_complete_corrected.csv','Table_1','Baseline_characteristics')
    emit(REP/'Table2_corrected_time_varying_estimates.csv','Table_2','Time_varying_HR')
    emit(BOOT/'table_test_main_comparison_bootstrap5000.csv','Table_3','AUC_IBS_5000')
    emit(BOOT/'test_c_indices_bootstrap5000.csv','Table_3','Concordance_point_estimates')
    emit(MODEL/'final_test_evaluation/test_calibration_summary.csv','Table_4','Calibration')
    emit(SOURCE/'Table_1_baseline_characteristics.csv','Table_S1','A_raw_categories')
    emit(REP/'Table_S1B_Included_and_excluded_corrected.csv','Table_S1','B_included_excluded')
    emit(SOURCE/'Table_S2_outcomes_by_split.csv','Table_S1','Split_event_counts')
    emit(SOURCE/'Table_2_CIF_at_1_3_5_years.csv','Table_S2','CIF')
    emit(SOURCE/'Table_S1_subtype_Gray_tests.csv','Table_S2','Gray_tests')
    for name in ['Medians','Overall_logrank','Pairwise_Holm']:
        emit(REP/f'Table_S3_{name}_corrected.csv','Table_S3',name)
    for endpoint,label in [('Cause_specific_Cox_SPM','SPM'),('Cox_OS','OS')]:
        emit(REP/f'Corrected_{endpoint}_corrected_interval_specific_HR.csv','Table_S4',label+'_HR')
        emit(TRAD/f'tables/{endpoint}_corrected_PH_diagnostics.csv','Table_S4',label+'_PH')
    for name in ['group_counts','KM_medians','KM_overall_logrank','KM_pairwise_Holm']:
        emit(TRAD/f'tables/Tumor_count_landmark_{name}.csv','Table_S5',name)
    dev=pd.read_csv(MODEL/'base_oof_evaluation/table_base_model_main_comparison.csv')
    dh=json.loads((REV/'05_deephit_retraining/deephit_retraining_summary.json').read_text(encoding='utf-8'))
    for key,value in dh['development_oof_metrics'].items():
        if key in dev.columns:dev.loc[dev.model.eq('deephit_cr'),key]=value
    dev.to_csv(OUT/'development_comparison_final.csv',index=False)
    emit(OUT/'development_comparison_final.csv','Table_S6','Development_models')
    for f in ['table_ensemble_retention_decision.csv','ensemble_weights_and_coefficients_long.csv']:
        emit(MODEL/'ensemble_oof_evaluation'/f,'Table_S6',Path(f).stem)
    for f in ['Table_DeepHit_inner_candidate_results.csv','Table_DeepHit_outer_fold_results.csv']:
        path=REV/'05_deephit_retraining'/f
        if path.exists():emit(path,'Table_S6',Path(f).stem)
    (OUT/'tables/Table_S6/DeepHit_final_settings.json').write_text(json.dumps({k:dh[k] for k in ['candidates','maximum_epochs','patience','batch_size','selected_final_candidate','final_training_log']},indent=2))
    emit(BOOT/'table_test_main_comparison_bootstrap5000.csv','Table_S7','Performance_5000')
    emit(BOOT/'test_primary_paired_differences_bootstrap5000.csv','Table_S7','Paired_differences_5000')
    emit(OUT/'risk_strata_summary.csv','Table_S7','Risk_strata')
    emit(OUT/'risk_strata_Gray.csv','Table_S7','Risk_strata_Gray')
    emit(SOURCE/'Table_S3_primary_model_sensitivity.csv','Table_S8','A_model_sensitivity')
    emit(SOURCE/'Table_S4_locked_model_by_MPN_subtype.csv','Table_S8','B_subgroups')
    emit(EVENT/'scenario_event_counts.csv','Table_S8','C_event_counts_R')
    cif=pd.concat([pd.read_csv(EVENT/'CIF_all_events_12_36_60_months.csv'),pd.read_csv(LAT6/'CIF_all_events_12_36_60_months.csv')],ignore_index=True)
    cif=cif.loc[~cif.scenario.eq('primary_tumour_first')]
    cif.to_csv(OUT/'CIF_sensitivity_without_duplicate_primary.csv',index=False)
    emit(OUT/'CIF_sensitivity_without_duplicate_primary.csv','Table_S8','D_CIF_sensitivity')
    emit(REV/'01_event_sensitivity/Table_event_sensitivity_XGBoost_performance.csv','Table_S8','E_refitted_models')
    emit(REV/'01_event_sensitivity/Table_event_sensitivity_cohort_counts.csv','Table_S8','E_refit_population_counts')
    emit(EVENT/'reverse_KM_followup.csv','Supporting','Reverse_KM_followup')
    emit(TRAD/'tables/SPM_type_distribution_primary_endpoint.csv','Supporting','SPM_sites')
    figure_sources=[
      (REP/'Figure3_corrected.pdf','Figure_3.pdf'),(REP/'Figure_S3_combined_corrected.pdf','Figure_S3.pdf'),
      (TRAD/'figures/Bar_first_nonhematologic_SPM_type.pdf','Figure_S1.pdf'),
      (REV/'02_model_explainability/Figure_model_explainability.pdf','Figure_S6.pdf')]
    for month in [12,36,60]:figure_sources.append((TRAD/f'figures/KM_landmark_{month}m_tumor_count_primary.pdf',f'Figure_S4_{month}m.pdf'))
    km=ROOT/'_codex_workspace/28_Figure_S2_all_KM_redraw_20261002/ready'
    for p in km.glob('*.pdf'):figure_sources.append((p,'S2_'+p.name))
    for p in pub.MAIN_FIG.glob('*.pdf'):figure_sources.append((p,p.name))
    for src,name in figure_sources:
        if not src.exists():raise FileNotFoundError(src.name)
        shutil.copy2(src,OUT/'figures'/name)
    with (OUT/'RESULT_TO_SOURCE_MAP.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(CATALOG[0]));w.writeheader();w.writerows(CATALOG)
    print('Final scientific source outputs exported. Manual submission layout is not byte-identical.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Export publication figures and tables from a marked reproduction run.')
    single_figure=parser.add_mutually_exclusive_group()
    single_figure.add_argument('--figure4-only',action='store_true',help='Redraw Figure 4 without rerunning models or other reports.')
    single_figure.add_argument('--figureS5-only',action='store_true',help='Redraw Figure S5 without rerunning models or other reports.')
    parser.add_argument('--output-pdf',type=Path,help='New PDF path for a single-figure redraw.')
    args=parser.parse_args()
    if args.figure4_only:
        if args.output_pdf is None:parser.error('--figure4-only requires --output-pdf')
        frames={
          'performance':pd.read_csv(BOOT/'table_test_main_comparison_bootstrap5000.csv'),
          'auc':pd.read_csv(BOOT/'test_auc_bootstrap5000.csv'),
          'brier':pd.read_csv(BOOT/'test_brier_bootstrap5000.csv')}
        pub.setup_style()
        figure4(frames,args.output_pdf)
        print(f'Figure 4 saved to {args.output_pdf}')
    elif args.figureS5_only:
        if args.output_pdf is None:parser.error('--figureS5-only requires --output-pdf')
        frames={
          'performance':pd.read_csv(BOOT/'table_test_main_comparison_bootstrap5000.csv'),
          'sensitivity':pd.read_csv(SOURCE/'Table_S3_primary_model_sensitivity.csv'),
          'subtype':pd.read_csv(SOURCE/'Table_S4_locked_model_by_MPN_subtype.csv')}
        pub.setup_style()
        figureS5(frames,args.output_pdf)
        print(f'Figure S5 saved to {args.output_pdf}')
    else:
        if args.output_pdf is not None:parser.error('--output-pdf requires --figure4-only or --figureS5-only')
        main()
