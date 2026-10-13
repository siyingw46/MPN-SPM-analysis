"""Compare only aggregate model metrics. Any missing/different result stays visible."""
import argparse, csv, json, math
from pathlib import Path

FILES={'table_test_main_comparison_bootstrap5000.csv':['model'],
       'test_auc_bootstrap5000.csv':['model','times'],
       'test_brier_bootstrap5000.csv':['model','times'],
       'test_primary_paired_differences_bootstrap5000.csv':['comparator_model','metric','times']}

def numeric(x):
    try:return float(x) if x!='' else float('nan')
    except ValueError:return None

def compare(a,b,keys,atol=1e-10,rtol=1e-8):
    def read(p):
        with p.open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        out={tuple(r[k] for k in keys):r for r in rows}
        if len(out)!=len(rows):raise ValueError('Duplicate aggregate keys')
        return out
    x,y=read(a),read(b);diff=[]
    if x.keys()!=y.keys():return [{'issue':'aggregate row keys differ'}]
    for key,row in x.items():
        for col,value in row.items():
            v=numeric(value)
            if v is None:continue
            u=numeric(y[key].get(col,''))
            same=u is not None and ((math.isnan(v) and math.isnan(u)) or math.isclose(v,u,rel_tol=rtol,abs_tol=atol))
            if not same:diff.append({'key':key,'column':col,'reference':value,'rerun':y[key].get(col),'issue':'numeric difference'})
    return diff

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--reference-dir',type=Path,default=Path(__file__).parent/'reference_metrics')
    a=p.parse_args();out={}
    for file,keys in FILES.items():
        ref=a.reference_dir/file;new=a.run_dir/'revision/04_bootstrap5000/model_root/final_test_evaluation'/file
        if not ref.exists() or not new.exists():out[file]={'status':'CANNOT_VERIFY','reason':'missing reference or rerun output'};continue
        try:d=compare(ref,new,keys);out[file]={'status':'MATCH' if not d else 'MISMATCH','differences':d}
        except (ValueError,KeyError) as e:out[file]={'status':'CANNOT_VERIFY','reason':str(e)}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    raise SystemExit(0 if all(x['status']=='MATCH' for x in out.values()) else 1)

if __name__=='__main__':main()
