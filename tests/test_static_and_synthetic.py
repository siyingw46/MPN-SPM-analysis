"""No SEER files are read. Syntax/configuration and fabricated-data tests only."""
import ast, csv, importlib.util, json, re, sys, tempfile, unittest
from pathlib import Path

PKG=Path(__file__).resolve().parents[1]
TEMP=PKG.parent/'test_artifacts'
TEMP.mkdir(exist_ok=True)
sys.path.insert(0,str(PKG/'scripts/python'))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def pure_function(name):
    tree=ast.parse((PKG/'scripts/python/01_build_datasets.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated pure function>','exec'),scope)
    return scope[name]

class Checks(unittest.TestCase):
    def test_all_python_parses(self):
        for p in PKG.rglob('*.py'):
            if 'runs' not in p.parts:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    def test_final_bootstrap_defaults(self):
        s=(PKG/'scripts/revision/17_v03_bootstrap_test_metrics.py').read_text()
        self.assertIn('"MPN_BOOTSTRAP_REPLICATES", "5000"',s)
        self.assertIn('"MPN_RANDOM_SEED", "20260723"',s)
        self.assertIn('"MPN_BOOTSTRAP_INCLUDE_C_INDICES", "0"',s)
    def test_no_old_bootstrap_readers(self):
        for p in (PKG/'scripts').rglob('*.py'):
            self.assertNotIn('bootstrap1000',p.read_text(encoding='utf-8'))
    def test_sequence_label_mapping(self):
        f=pure_function('sequence_rank')
        self.assertEqual(f('One primary only'),1)
        self.assertEqual(f('1st of 2 or more primaries'),1)
        self.assertEqual(f('2nd of 2 or more primaries'),2)
        self.assertIsNone(f('unknown'))
        # This export parser accepts value LABELS, not numeric 00/01.
        self.assertIsNone(f('00'))
    def test_age_topcoding(self):
        f=pure_function('parse_age')
        self.assertEqual(f('85+ years')[:2],(85.0,1))
        self.assertEqual(f('64 years')[:2],(64.0,0))
    def test_r_bridge_preserves_order_and_unknown(self):
        import build_r_inputs as b
        b.RAW_FIELDS=['Patient ID','Sequence number','Median household income inflation adj to 2024']
        unknown='Unknown/missing/no match/Not 1990-2024'
        master=[{'patient_id':'SYN_A','index_Patient ID':'SYN_A','index_Sequence number':'One primary only',
          'index_Median household income inflation adj to 2024':unknown},
          {'patient_id':'SYN_B','index_Patient ID':'SYN_B','index_Sequence number':'1st of 2 or more primaries',
          'index_Median household income inflation adj to 2024':'$60,000 - $64,999'}]
        rows=b.reconstruct_index(master,[{'patient_id':'SYN_B'},{'patient_id':'SYN_A'}])
        self.assertEqual([r['Patient ID'] for r in rows],['SYN_B','SYN_A'])
        self.assertEqual(rows[1]['Median household income inflation adj to 2024'],unknown)
        with self.assertRaises(ValueError):b.reconstruct_index(master+master,[{'patient_id':'SYN_A'}])
    def test_unsafe_direct_script_entry_rejected(self):
        import os
        from runtime_paths import work_root
        old=os.environ.pop('MPN_RUN_ROOT',None)
        try:
            with self.assertRaises(RuntimeError):work_root()
        finally:
            if old is not None:os.environ['MPN_RUN_ROOT']=old
    def test_stage_dependencies(self):
        runner=load('runner_test',PKG/'run_pipeline.py')
        seen=set()
        for stage,(deps,commands) in runner.STAGES.items():
            self.assertTrue(set(deps)<=seen)
            for kind,name in commands:
                folder={'py':'python','rev':'revision','r':'r'}[kind]
                self.assertTrue((PKG/'scripts'/folder/name).is_file(),name)
            seen.add(stage)
    def test_cohort_order_mismatch_is_rejected(self):
        runner=load('runner_cmp',PKG/'run_pipeline.py')
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            a=Path(tmp)/'a.csv';b=Path(tmp)/'b.csv'
            header='patient_id,data_split,split_seed,time_months,event_type\n'
            r1='SYN_A,development,20260723,30,1\n';r2='SYN_B,test,20260723,50,0\n'
            a.write_text(header+r1+r2);b.write_text(header+r1+r2)
            self.assertEqual(runner.compare_cohort(a,b)['status'],'PASS')
            b.write_text(header+r2+r1)
            with self.assertRaises(RuntimeError):runner.compare_cohort(a,b)
    def test_metric_comparison_detects_change(self):
        cmp=load('metrics_cmp',PKG/'compare_results.py')
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            a=Path(tmp)/'a.csv';b=Path(tmp)/'b.csv'
            a.write_text('model,iAUC\nsynthetic,0.65\n');b.write_text('model,iAUC\nsynthetic,0.65\n')
            self.assertEqual(cmp.compare(a,b,['model']),[])
            b.write_text('model,iAUC\nsynthetic,0.67\n')
            self.assertEqual(len(cmp.compare(a,b,['model'])),1)
    def test_table_reference_files_are_aggregate(self):
        for p in (PKG/'reference_metrics').glob('*.csv'):
            with p.open(encoding='utf-8-sig') as f:header=next(csv.reader(f))
            self.assertFalse(any('patient' in s.lower() for s in header))
    def test_reference_comparison_schema(self):
        cmp=load('metrics_schema',PKG/'compare_results.py')
        for name,keys in cmp.FILES.items():
            path=PKG/'reference_metrics'/name
            self.assertTrue(path.is_file())
            self.assertEqual(cmp.compare(path,path,keys),[])

if __name__=='__main__':unittest.main(verbosity=2)
