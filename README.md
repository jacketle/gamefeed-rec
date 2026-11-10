# GameFeed-Rec：短视频召回与用户兴趣建模

基于KuaiRand公开日志，实现数据处理、时序历史、候选召回、曝光预测、分组评价和本地演示。项目包含两条实验线：**Pure比较推荐列表的召回与排序；1K研究历史信息怎样帮助曝光预测。**

- **Pure：** 热门与ItemCF组成候选，按RRF排序，测试Recall@100为**25.32%**，比ItemCF提高0.69个百分点。
- **1K：** 在B0上增加3个显式历史匹配特征，三个种子平均测试GAUC从**0.69076升至0.69377**，提升**0.300个百分点**，增加192个权重。
- **验证：** 时间切分、开发集选方案、配对种子、用户配对抽样和分组分析；47项单元测试覆盖历史边界、特征回退、指标与检查点恢复。

## 先看结果与讲解

- [最新优化报告](reports/optimization_v2/summary.md)：作者频次、初始化、显式匹配、三种子复核与训练曲线。
- [项目技术讲解](docs/PROJECT_GUIDE.md)：两条流程、关键设计、实验取舍和面试讲解要点。
- [Pure实验报告](reports/release/experiment_summary.md)：召回、双塔、行为预测和游戏内容分组。
- [离线图文演示](deliverables/GameFeed_实验结果图解.html)：下载后用浏览器打开，含结构图、最新1K结果与技术问答。

仓库提供代码、配置、测试、汇总结果和图表。原始数据从官方获取，模型和逐用户预测由本地复现生成。

## 主要结果

### Pure：固定100条候选的推荐列表

5,000名用户，272,354条行为，7,583条视频；测试集包含7,122个用户日期查询。

| 方法 | 测试Recall@100 | 内容覆盖率@100 |
| --- | ---: | ---: |
| 热门 | 22.22% | 1.38% |
| ItemCF | 24.63% | 66.95% |
| BGE＋历史均值 | 3.32% | 81.48% |
| 配额候选＋RRF排序 | **25.32%** | **66.95%** |

选定方案保留10条热门候选，ItemCF补齐100条；RRF结合各路名次排序，语义路在重叠内容上贡献分数。融合相对ItemCF的Recall@100差值95%区间为[0.441，0.941]个百分点。游戏组保留独立分析。

观看达标／点赞模型的测试AUC为0.7303／0.8375，但将模型用于候选重排后，Recall@10从4.71%降至3.23%。补充的场景一致性诊断也未得到足以替换默认方案的结果，**Pure演示继续使用RRF顺序**。

### 1K：常规曝光上的用户兴趣建模

覆盖全部1,000名日志用户，正式训练3,764,530条曝光、开发1,642,223条、测试1,188,322条。以下为三个配对种子的GAUC均值。

| 模型 | 开发GAUC | 测试GAUC | 用途 |
| --- | ---: | ---: | --- |
| B0：基础曝光预测 | 0.69490 | 0.69076 | 对照 |
| B1：历史向量均值 | 0.68743 | 0.68705 | 前期历史建模对照 |
| B2：候选相关历史加权 | 0.68794 | 0.68795 | 前期历史建模对照 |
| **B0＋3个显式历史匹配特征** | **0.69888** | **0.69377** | **当前优选方案** |

三个特征是历史同作者占比、是否出现同作者、与候选共享标签的历史占比。主对照采用相同样本、公共参数初始化与评价频率，开发集选方案后追加种子2027、2028。测试平均提升**0.300个百分点，用户配对95%区间[0.220，0.380]个百分点**；该数字是逐种子指标的均值。

全部作者ID直接加入时效果下降；筛除低频作者、减小初始化尺度后改善。显式历史匹配在开发集的训练出现／未见视频和六个作者频次组中均有正向点估计，最终三种子复核结果见[最新报告](reports/optimization_v2/summary.md)。

## 安装与测试

下面以PowerShell为例，在仓库根目录运行；建议使用Python 3.11或3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[semantic,report]"
python -m gamefeed.cli test
```

Pure可在CPU上运行。1K完整训练入口使用CUDA，需要安装与设备匹配的PyTorch版本，可参考[官方安装说明](https://pytorch.org/get-started/locally/)。训练前可用 `python -c "import torch; print(torch.cuda.is_available())"` 检查CUDA是否可用。

可选的本地浏览器交互检查使用 `python -m pip install -e ".[browser]"` 安装依赖。

## 官方数据与复现

数据说明见[KuaiRand官方仓库](https://github.com/chongminggao/KuaiRand)，压缩包来自[官方Zenodo记录](https://zenodo.org/records/10439422)。

### Pure流程

将 `KuaiRand-Pure.tar.gz` 放到 `data/raw/kuairand/`，准备命令会解压。再从官方仓库补充数据入口获取 `kuairand_video_captions.csv`、`kuairand_video_categories.csv`，放到 `data/raw/kuairand/supplement/`。

```powershell
python -m gamefeed.cli inspect --config configs/base.yaml
python -m gamefeed.cli prepare --config configs/base.yaml
python -m gamefeed.cli prepare-content --config configs/base.yaml

# 首次联网下载BGE编码器，后续复用本地缓存
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
python -m gamefeed.cli validate-models --config configs/release.yaml
python -m gamefeed.cli train-ranker --config configs/release.yaml
python -m gamefeed.cli train-single-rankers --config configs/release.yaml
python -m gamefeed.cli optimize-fusion --config configs/release.yaml
python -m gamefeed.cli report --config configs/release.yaml
python scripts/evaluate_release.py
python scripts/check_report_metrics.py
python scripts/evaluate_pure_scenes.py
```

生成候选与模型后，运行 `python -m gamefeed.cli serve --config configs/release.yaml`，打开 <http://127.0.0.1:8765> 查看推荐演示。

### 1K流程

解压 `KuaiRand-1K.tar.gz` 到 `data/raw/kuairand_1k/`，使日志与元数据位于 `data/raw/kuairand_1k/KuaiRand-1K/data/`。

```powershell
# 首次准备：全部用户、逐事件历史和时间切分
python -m gamefeed.sequence_data --config configs/kuairand_1k_prepare.yaml

# 五组开发筛选，优胜方案与B0追加两个配对种子，再评价测试集
python scripts/run_author_followup.py
python scripts/audit_followup_matches.py
python scripts/audit_followup_author_groups.py
```

数据准备写入 `data/processed/kuairand_1k/full1000/`，结果写入独立的 `reports/kuairand_1k/preparation/`；已有处理数据时从训练入口继续。训练配置为 `configs/kuairand_1k_followup.yaml`，新模型写入 `artifacts/kuairand_1k/author_v2/`，训练记录与比较结果写入 `reports/optimization_v2/sequence/`。

完成上述两条流程后，运行 `python scripts/report_author_followup.py` 重建联合总结与曲线。报告同时读取仓库保留的前期1K聚合记录，用于说明新旧对照的训练停点。训练入口复用同配置下已完成的训练结果。

## 代码导航

| 位置 | 作用 |
| --- | --- |
| `src/gamefeed/data.py`、`history.py`、`samples.py` | Pure清洗、按日历史和召回样本 |
| `baselines.py`、`fusion_experiment.py`、`text_retrieval.py` | 热门、ItemCF、BGE与融合 |
| `models.py`、`training.py` | 双塔、单任务与共享行为预测 |
| `sequence_data.py` | 1K逐事件历史、同时刻排除与数据切分 |
| `sequence_model.py`、`sequence_training.py` | B0/B1/B2、作者与匹配特征、训练和停点 |
| `metrics.py`、`final_evaluation.py`、`sequence_evaluation.py` | 召回、排序、用户GAUC与配对区间 |
| `scripts/run_author_followup.py` | 五组筛选、配对种子复核和测试评价 |
| `serving.py`、`web/` | Pure本地演示 |
| `tests/`、`reports/optimization_v2/` | 回归测试、最新汇总和图表 |

## 评价口径

- Pure按用户与日期评价推荐列表；1K按曝光预测 `long_view`，GAUC在有正负反馈的用户内计算，再按曝光数加权。两条线的指标分别解释。
- Pure训练／开发／测试为2022年4/8–4/21、4/22–4/28、4/29–5/8；1K用4/8–4/21积累历史，正式训练4/22–5/1、开发5/2–5/5、测试5/6–5/8。
- 历史上限均为20条。Pure只读前一天结束前的观看达标记录；1K只读严格早于请求的常规观看达标记录，同时间戳互不进入历史。缺失作者和未知标签不计为匹配。
- `long_view`表示短于或等于18秒的视频看完，较长视频至少观看18秒。用户配对区间使用2,000次抽样；1K结果覆盖三个配对种子。既有测试集用于本项目回顾性比较。

## 参考资料

- Gao et al., *KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos*, CIKM 2022. [论文](https://doi.org/10.1145/3511808.3557624)
- [BGE-small-zh-v1.5模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [DIN论文与作者代码](https://github.com/zhougr1993/DeepInterestNetwork)
- [FuxiCTR](https://github.com/reczoo/FuxiCTR)、[DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)：用于核对初始化、特征与训练实现。

数据和预训练模型遵循各自发布方许可。
