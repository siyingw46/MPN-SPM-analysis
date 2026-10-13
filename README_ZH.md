# 经典 MPN 后第二原发恶性肿瘤：代码使用入口

**只从仓库根目录的 `run_pipeline.py` 开始。** 不要按文件名前面的数字逐个运行 `scripts/` 中的脚本；运行器会按依赖顺序调用它们。文件名中的 `v03`、`v04` 和 `revision` 是开发期名称，当前报告的分析配置是 **v0.4**，最终 bootstrap 为 **5,000 次患者级重抽样**。

## 先看什么

1. [阶段与脚本导航](docs/PIPELINE_MAP.md)：13 个阶段各做什么、依赖什么、调用什么脚本。
2. [输入要求](INPUT_CONTRACT.md)：SEER*Stat 导出字段、00/01 序列类别及研究对象定义。必须由研究者自行获得 SEER 数据使用授权。
3. [结果与来源对照](docs/RESULT_MAP.md)：正文及补充图表的科学源输出在哪里；手工排版文件不等同于脚本直接输出。
4. [复现范围](REPRODUCIBILITY.md)和[软件记录](environment/SOFTWARE_VERSIONS.md)：说明已经验证什么、尚未验证什么。

仓库不含 SEER 患者级数据、患者级预测、SHAP 个体值、已拟合模型或检查点。原始数据和运行目录必须放在仓库**之外**的受控位置，不得上传 GitHub。

## 从 PowerShell 开始

在仓库根目录运行。先把示例路径换成你电脑上的实际路径：

```powershell
$PythonExe = 'PATH_TO_PYTHON_3_12_10'
$RscriptExe = 'PATH_TO_RSCRIPT'
$RunDir = 'NEW_EMPTY_RESTRICTED_RUN_DIRECTORY'
$RawExport = 'AUTHORISED_LABELLED_SEER_EXPORT.txt'
$RLibrary = 'PATH_TO_LOCAL_R_LIBRARY'

& $PythonExe check_environment.py
& $PythonExe run_pipeline.py plan
```

上面两条只检查环境并显示阶段依赖，**不会训练模型**。确定输入满足 [输入要求](INPUT_CONTRACT.md)后，初始化一个尚不存在、位于仓库外的受控运行目录：

```powershell
& $PythonExe run_pipeline.py init --run-dir $RunDir --raw $RawExport
```

如持有同一导出的获授权历史患者级队列，可在 `init` 命令中另加 `--reference-cohort 'AUTHORISED_REFERENCE_COHORT.csv'`，以核对患者顺序、分组、事件时间及事件类型；不提供时，这项等价性**不能确认**。

之后先运行第一个阶段：

```powershell
& $PythonExe run_pipeline.py run --run-dir $RunDir --stage data `
  --rscript $RscriptExe --r-library $RLibrary --allow-analysis
```

只有确认当前阶段通过后，才将 `data` 换为下一阶段名称并单独运行。完整顺序为：`descriptive` → `r_models` → `r_figures` → `event_r` → `base_oof` → `ensemble_oof` → `locked` → `ml_sensitivity` → `revision` → `deephit` → `bootstrap` → `report`。运行器不会自动执行整条流水线，不会自动重试失败阶段，也不会覆盖已尝试的阶段。具体依赖及输出见[阶段与脚本导航](docs/PIPELINE_MAP.md)。

全部阶段完成后，可比较四张锁定测试集 bootstrap 汇总表：

```powershell
& $PythonExe compare_results.py --run-dir $RunDir
```

这**不是**全文所有数值及图表的自动核验，也不是外部验证。受控复跑使用 Windows 11 x64、Python 3.12.10、R 4.6.1 和随机种子 20260723。受控复跑已在作者本机完成，并对关键阶段进行了第二次重复核查；本公开代码包尚未在另一台独立电脑上完成从零安装和全流程复跑。

作者选择公开代码供查看，但**不授予代码许可证**。公开访问不自动授予一般性的使用、修改或再分发权利。发布前还应确认 SEER 协议、机构要求及目标期刊政策；仓库目前只是待上传候选版。
