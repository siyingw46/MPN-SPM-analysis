"""Construct the R index input from the same index rows saved by the ML builder.

No reselection of patients, sorting of the split, or income recoding occurs here.
Outputs remain restricted patient-level data and must not be publicly released.
"""
import csv, hashlib, json, shutil
from pathlib import Path
from runtime_paths import work_root

def reconstruct_index(master, ml):
    lookup={r['patient_id']:r for r in master}
    if len(lookup)!=len(master) or len({r['patient_id'] for r in ml})!=len(ml):
        raise ValueError('Duplicate patient keys')
    fields=[x[6:] for x in master[0] if x.startswith('index_')]
    # Raw exported labels contain capitals/spaces, derived index fields do not.
    fields=[x for x in fields if x in RAW_FIELDS]
    if set(fields)!=set(RAW_FIELDS):
        raise ValueError('Missing raw index columns in master')
    rows=[{name:lookup[r['patient_id']]['index_'+name] for name in RAW_FIELDS} for r in ml]
    if any(x['Patient ID']!=m['patient_id'] for x,m in zip(rows,ml)):
        raise ValueError('Index identity mismatch')
    return rows

# Same raw-column contract as 01_build_datasets; loaded without executing it.
RAW_FIELDS=[]
def main():
    import importlib
    global RAW_FIELDS
    RAW_FIELDS=importlib.import_module('01_build_datasets').RAW_FIELDS
    root=work_root();data=root/'data/processed/v0.4'
    def read(name):
        with (data/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    master=read('patient_master_v0.4_2000_2023.csv');ml=read('ml_competing_risk_v0.4_2004_2023.csv')
    rows=reconstruct_index(master,ml)
    if len(rows)!=85680: raise ValueError('V4 cohort size did not reproduce')
    target=root/'MPN/SEER_MPN_raw_index_records_v0.4_85680.csv';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=RAW_FIELDS);w.writeheader();w.writerows(rows)
    for src,dst in [(data/'patient_master_v0.4_2000_2023.csv',root/'MPN/patient_master_v0.4_2000_2023.csv'),
                    (data/'ml_competing_risk_v0.4_2004_2023.csv',root/'ML/data/ml_competing_risk_v0.4_2004_2023.csv')]:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    # Public-safe log contains only counts and a file checksum.
    (root/'logs/r_index_bridge.json').write_text(json.dumps({'rows':len(rows),'source':'raw index fields retained in patient master',
      'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'historical_index_rowwise_comparison':'NOT_RUN'},indent=2))
    print('R input bridge completed; historical input equivalence still requires local comparison.')

if __name__=='__main__':main()
