# Software and provenance of the controlled local reruns

| Component | Observed or confirmed version | Evidence/status |
|---|---|---|
| SEER*Stat extraction | 9.0.43.0 | Researcher-confirmed version of the original extraction; **not executed** in the local rerun. |
| SEER database | Research Limited-Field Data, 21 Registries excluding Illinois, November 2025 Submission, covering 2000–2023 | Manuscript/export setting; the restricted source file is not in this package. |
| Operating system | Windows 11 x64, build 26300 | `run02/logs/execution_environment.json` in controlled storage. |
| CPython | 3.12.10, 64-bit, MSC v.1943 | Restricted `run02` environment log; the local virtual environment is not distributed. |
| R / Rscript | R 4.6.1 (2026-06-24 ucrt) | Restricted `run01` environment log and `R_sessionInfo_run01.txt`. |
| Random seed | 20260723 | `run_pipeline.py` and 5,000-bootstrap summary. |
| Bootstrap | 5,000 patient-level resamples, fixed predictions/models; 95% confidence intervals | `run_pipeline.py` and `run02/revision/04_bootstrap5000/model_root/final_test_evaluation/test_bootstrap5000_summary.json`. |

Key Python distributions in the actual virtual environment: pandas **2.3.3**; NumPy **2.4.6**; SciPy **1.18.0**; scikit-learn **1.9.0**; scikit-survival **0.28.0**; XGBoost **3.3.0**; comprisk **0.7.0**; PyTorch **2.7.1+cpu**; pycox **0.3.0**; torchtuples **0.2.2**; joblib **1.5.3**; matplotlib **3.11.1**; PyYAML **6.0.3**. The complete **59-package** observed inventory is `requirements-observed-run02.txt`. `check_environment.py` compares every listed distribution with the current interpreter.

The CPU-only PyTorch wheel was recorded by `pip freeze --all` as:

```text
torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp312-cp312-win_amd64.whl#sha256=0bc887068772233f532b51a3e8c8cfc682ae62bef74bf4e0c53526c8b9e4138f
```

`requirements-rebuild-windows-cpu.txt` is a candidate installation recipe with that exact wheel, whereas `requirements-observed-run02.txt` is the authoritative **observed** environment inventory. A clean installation from the recipe was **not tested** in this phase and would require authorised package downloads.

Directly attached R packages in the traditional analysis session were: survival **3.8-6**, cmprsk **2.2-12**, survminer **0.5.2**, data.table **1.18.4**, dplyr **1.2.1**, stringr **1.6.0**, broom **1.0.13**, ggplot2 **4.0.3**, ggpubr **1.0.0**, cowplot **1.2.0**, and tableone **0.13.2**. `R_sessionInfo_run01.txt` additionally records loaded dependency namespaces, locale and platform. This is a session record, **not** a tested `renv.lock` or proof that fresh installation from public repositories currently succeeds.

The R stages used a separate locally installed package library through `--r-library` (`R_LIBS_USER`). The default R installation lacked `cmprsk` in the controlled run. A new user must provide a suitable local R library; installed software is **not** included in this code package.

The older code package's `requirements-lock.txt` came from a preliminary smoke-test environment and includes dependencies not installed in the controlled rerun. It must not be cited as the final computational environment. Run directories and original input hashes are retained in restricted storage; the restricted input itself is not distributed.
