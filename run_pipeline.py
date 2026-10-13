"""Explicit stage runner. Default action is a plan, never a full analysis."""
from pathlib import Path
import argparse, csv, hashlib, importlib.metadata, json, os, platform, shutil, subprocess, sys
from datetime import datetime, timezone

PKG=Path(__file__).resolve().parent
PY=PKG/'scripts/python'
REV=PKG/'scripts/revision'
R=PKG/'scripts/r'
STAGES={
 'data':([], [('py','01_build_datasets.py'),('py','02_validate_datasets.py'),('py','build_r_inputs.py')]),
 'descriptive':(['data'], [('py','20_v04_descriptive_delivery.py')]),
 'r_models':(['data'], [('r','traditional.R'),('r','exclusion_comparison.R')]),
 'r_figures':(['r_models'], [('r','publication_tables.R'),('r','figure_S2.R')]),
 'event_r':(['descriptive'], [('r','event_timing.R'),('r','latency6.R')]),
 'base_oof':(['data'], [('py','11_v03_nested_oof.py'),('py','12_v03_oof_audit.py'),('py','13_v03_aggregate_base_oof.py')]),
 'ensemble_oof':(['base_oof'], [('py','14_v03_nested_ensembles.py'),('py','15_v03_audit_and_evaluate_ensembles.py')]),
 'locked':(['ensemble_oof'], [('py','16_v03_fit_final_and_evaluate_test.py')]),
 'ml_sensitivity':(['locked','descriptive'], [('py','19_v04_build_sensitivity_datasets.py'),('py','21_v04_sensitivity_xgboost.py')]),
 'revision':(['data'], [('rev','01_core_revision_analyses.py')]),
 'deephit':(['locked'], [('rev','02_deephit_retraining.py')]),
 'bootstrap':(['deephit'], [('rev','04_prepare_bootstrap5000_inputs.py'),('rev','17_v03_bootstrap_test_metrics.py')]),
 'report':(['r_figures','event_r','ml_sensitivity','revision','bootstrap'], [('py','final_reporting.py')]),
}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def init_run(root,raw,reference=None):
    if root.exists():raise RuntimeError('Use a NEW, nonexistent run directory; no overwrite permitted.')
    raw=raw.resolve()
    if not raw.is_file():raise FileNotFoundError('Authorised SEER export not found')
    root.mkdir(parents=True)
    (root/'.mpn_reproduction_run').write_text('V4 controlled local reproduction; not for public release\n')
    for name in ['logs','config','data/raw','_codex_workspace/25_income_reanalysis_20261001',
                 '_codex_workspace/21_sequence_income_integrity_20261001','ML/script/scripts/python']:
        (root/name).mkdir(parents=True,exist_ok=True)
    raw_dest=root/'data/raw/SEER_MPN_first_primary_person_selection_2000_2023.txt'
    shutil.copy2(raw,raw_dest)
    # Legacy R readers use this alias; both copies are confined to the new run.
    shutil.copy2(raw,root/raw_dest.name)
    for p in (PKG/'config').glob('*.yml'):shutil.copy2(p,root/'config'/p.name)
    shutil.copy2(R/'income_recode.R',root/'_codex_workspace/21_sequence_income_integrity_20261001/income_recode_fix.R')
    shutil.copy2(PY/'01_build_datasets.py',root/'ML/script/scripts/python/01_build_datasets.py')
    save(root/'input_manifest.json',{'raw_source':str(raw),'raw_sha256':sha(raw),'copied_sha256':sha(raw_dest),
       'reference_cohort':str(reference.resolve()) if reference else None,
       'reference_cohort_sha256':sha(reference) if reference else None,
       'created_utc':datetime.now(timezone.utc).isoformat(),
       'restriction':'This run directory contains restricted patient-level data. Do not upload it.'})
    save(root/'status.json',{})

def compare_cohort(reference,new):
    def read(p):
        with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    a,b=read(reference),read(new)
    keys=['patient_id','data_split','split_seed','time_months','event_type']
    if len({x['patient_id'] for x in a})!=len(a) or len({x['patient_id'] for x in b})!=len(b):
        raise RuntimeError('Duplicate patient IDs in cohort comparison')
    if len(a)!=len(b):raise RuntimeError('Cohort row counts differ')
    # Order matters: fold generation and subsequent prediction arrays use this order.
    n=sum(any(x.get(k)!=y.get(k) for k in keys) for x,y in zip(a,b))
    if n:raise RuntimeError(f'{n} rows differ in identity/order/split/time/event; stop before modelling.')
    return {'status':'PASS','rows':len(a),'checked':keys,'order_checked':True}

def environment(root,rscript):
    packages=['numpy','pandas','scipy','scikit-learn','scikit-survival','xgboost','comprisk',
              'torch','pycox','torchtuples','joblib','matplotlib','PyYAML']
    versions={}
    for p in packages:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]='MISSING'
    save(root/'logs/execution_environment.json',{'python':sys.version,'executable':sys.executable,
      'platform':platform.platform(),'packages':versions,'Rscript':rscript,
      'note':'Actual execution environment; not proof of the historical environment.'})

def run_stage(root,stage,rscript,r_library):
    if not (root/'.mpn_reproduction_run').is_file():raise RuntimeError('Run init first')
    state=json.loads((root/'status.json').read_text())
    if stage in state:raise RuntimeError('Stage already attempted. Inspect its log; use a fresh run for a clean rerun.')
    deps,commands=STAGES[stage]
    missing=[x for x in deps if state.get(x,{}).get('status')!='PASS']
    if missing:raise RuntimeError('Unfinished dependencies: '+', '.join(missing))
    if any(k=='r' for k,_ in commands) and not shutil.which(rscript):raise RuntimeError('Rscript not found; provide --rscript')
    env=os.environ.copy()
    env.update(MPN_RUN_ROOT=str(root),MPN_PROJECT_ROOT=str(root),MPN_ANALYSIS_VERSION='v0.4',
      MPN_RANDOM_SEED='20260723',MPN_BOOTSTRAP_REPLICATES='5000',MPN_BOOTSTRAP_INCLUDE_C_INDICES='0',
      MPN_DATA_PATH=str(root/'data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv'),
      MPN_MODEL_ROOT=str(root/'output/models/model_development_v0.4'),
      MPN_RUN_SIGNATURE='v0.4_nested5x3_inner3_20260723',
      MPN_ENSEMBLE_SIGNATURE='v0.4_nested_ensemble_inner3_20260723',
      PYTHONPATH=os.pathsep.join([str(PY),str(REV)]),PYTHONUTF8='1',MPLBACKEND='Agg',
      MPLCONFIGDIR=str(root/'logs/matplotlib_cache'))
    if r_library:env['R_LIBS_USER']=str(Path(r_library).resolve())
    environment(root,rscript)
    started=datetime.now(timezone.utc).isoformat()
    state[stage]={'status':'RUNNING','started_utc':started};save(root/'status.json',state)
    try:
        if stage=='event_r':
            dest=root/'ML/tables/source_data';dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/'deliverables/v0.4/tables/source_data/Table_2_CIF_at_1_3_5_years.csv',dest/'Table_2_CIF_at_1_3_5_years.csv')
        for kind,name in commands:
            cmd=[rscript,'--vanilla',str(R/name)] if kind=='r' else [sys.executable,'-X','utf8',str((REV if kind=='rev' else PY)/name)]
            if name=='event_timing.R':cmd.extend([str(root),'primary_run'])
            if name=='latency6.R':cmd.extend([str(root),'latency6_run'])
            localenv=env.copy()
            if name=='17_v03_bootstrap_test_metrics.py':
                localenv['MPN_MODEL_ROOT']=str(root/'revision/04_bootstrap5000/model_root')
            logfile=root/'logs'/f'{stage}_{Path(name).stem}.log'
            print(f'Running {stage}: {name}; local log: {logfile}',flush=True)
            with logfile.open('w',encoding='utf-8') as log:
                result=subprocess.run(cmd,cwd=root,env=localenv,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{name} failed (exit {result.returncode}); inspect local log. No automatic retry.')
        if stage=='data':
            manifest=json.loads((root/'input_manifest.json').read_text())
            if manifest['reference_cohort']:
                if sha(manifest['reference_cohort'])!=manifest['reference_cohort_sha256']:raise RuntimeError('Reference input changed')
                check=compare_cohort(manifest['reference_cohort'],root/'data/processed/v0.4/ml_competing_risk_v0.4_2004_2023.csv')
            else:check={'status':'NOT_VERIFIED','reason':'No historical cohort supplied for row/order/split comparison'}
            save(root/'logs/cohort_equivalence.json',check)
        state[stage]['status']='PASS'
    except Exception as e:
        state[stage].update(status='FAIL',error=str(e));raise
    finally:
        state[stage]['finished_utc']=datetime.now(timezone.utc).isoformat();save(root/'status.json',state)
    print(stage+' completed. Execution success alone does not establish numerical reproduction.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','init','run'],nargs='?',default='plan')
    p.add_argument('--run-dir',type=Path);p.add_argument('--raw',type=Path);p.add_argument('--reference-cohort',type=Path)
    p.add_argument('--stage',choices=STAGES);p.add_argument('--rscript',default='Rscript');p.add_argument('--r-library')
    p.add_argument('--allow-analysis',action='store_true')
    a=p.parse_args()
    if a.action=='plan':
        for stage,(deps,_) in STAGES.items():print(stage+': after '+(', '.join(deps) or 'init'))
        return
    if not a.run_dir:p.error('--run-dir required')
    root=a.run_dir.resolve()
    if a.action=='init':
        if not a.raw:p.error('--raw required (authorised, labelled TSV with header)')
        init_run(root,a.raw,a.reference_cohort)
    else:
        if not a.stage or not a.allow_analysis:p.error('--stage and --allow-analysis required; no implicit full run')
        run_stage(root,a.stage,a.rscript,a.r_library)

if __name__=='__main__':main()
