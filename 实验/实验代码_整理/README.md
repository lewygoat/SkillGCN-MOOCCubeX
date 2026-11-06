# 实验代码说明

开源仓库地址：[https://github.com/lewygoat/SkillGCN-MOOCCubeX](https://github.com/lewygoat/SkillGCN-MOOCCubeX)

本目录汇集「融合职业技能标准与课程知识图谱的图注意力学习路径推荐研究」全部实验代码，按功能分为九类。所有脚本沿用扁平布局，彼此通过顶层模块互相import（`common`、`guards`、`trainer`、`evaluator`），运行时假定当前工作目录为本目录，数据与产物以相对路径`../中间产物`、`../结果`、`../图表`访问，因此不要将文件移入子目录，否则import与路径均会断裂。

## 运行环境

Python配合PyTorch、NumPy、matplotlib、scipy。模型训练默认五个随机种子（42至46）。核验与绘图脚本依赖`../中间产物`下的图结构`s6_graph.pt`、技能权重`s9_skill_weight.npy`以及导出的嵌入。

## 一、共享核心模块

被多数脚本import的底层组件，不单独执行。

| 文件 | 作用 |
|------|------|
| `common.py` | 全局常量、路径、数据加载与通用工具，被22个脚本引用 |
| `guards.py` | 数据泄漏核验、损失与显存与参数与指标检查，被15个脚本引用 |
| `trainer.py` | 统一训练循环，被10个脚本引用 |
| `evaluator.py` | NDCG@10、HR@10、MRR、Recall@20、Precision@10计算，被9个脚本引用 |
| `monitor.py` | 心跳日志读取与进度监控 |

## 二、数据流水线（s1至s9）

从原始MOOCCubeX逐级构建训练图与技能权重，编号即执行顺序。

| 文件 | 作用 |
|------|------|
| `s1_concepts.py` | 概念实体抽取与信息技术领域过滤 |
| `s2_relations.py` | 先修、包含、章节序等关系集构建 |
| `s3_behavior.py` | 学习者交互行为抽取 |
| `s4_split.py` | 按学习者时间顺序划分训练验证测试 |
| `s5_filter.py` | 过滤并核验划分无学习者概念重叠 |
| `s6_graph.py` | 生成归一化交互图`s6_graph.pt` |
| `s7_skills.py` | 从职业技能标准文本解析技能信号 |
| `s8_align.py` | 技能与概念按名称对齐 |
| `s9_gen.py` | 生成最终技能权重向量`s9_skill_weight.npy` |

## 三、模型实现（m_*）

七个基线加一个技能加权模型，共享同一默认超参配置。

| 文件 | 作用 |
|------|------|
| `m_popularity.py` | 非参数Most-Popular排序 |
| `m_bprmf.py` | BPR-MF |
| `m_lightgcn.py` | LightGCN主干，被2个脚本引用 |
| `m_ngcf.py` | NGCF |
| `m_kgat.py` | KGAT |
| `m_conceptgcn.py` | ConceptGCN |
| `m_ackrec.py` | ACKRec |
| `m_skillgcn.py` | 技能加权缩放模型，含可学习门控g，被6个脚本引用 |

## 四、主实验：技能条件对照（RQ1）

T1零权重、T2随机置换、T3真实权重的五种子对照。

| 文件 | 作用 |
|------|------|
| `layer1_check.py` | T1与T3主对照运行 |
| `layer1_t2.py` | T2随机置换控制条件 |
| `layer1_early.py` | 早停诊断运行 |
| `smoke_skillgcn.py` | 技能模型快速连通性测试 |

## 五、门控与缩放诊断（RQ2）

定位效应在缩放幅度而非信号内容。

| 文件 | 作用 |
|------|------|
| `probe_gate.py` | 门控轨迹探测（关闭早停的扩展运行） |
| `exp5_hyperparam.py` | 定值门控扫描与超参敏感性 |
| `exp5_sparsity.py` | 按学习者活跃度分桶的稀疏性分析 |
| `supp_run.py` | 在学习到的终端门控值上重训练 |
| `supp_sparsity.py` | 补充稀疏性实验 |

## 六、路径构建与评估（RQ3）

验证技能覆盖优势是否在L2归一化下消失。

| 文件 | 作用 |
|------|------|
| `path_gen.py` | 贪心拓扑路径生成与违反率计算 |
| `path_l2norm.py` | L2归一化嵌入下的路径评估 |
| `path_pregate.py` | 门控前后嵌入的覆盖对比 |

## 七、嵌入导出与可复现核验

| 文件 | 作用 |
|------|------|
| `embed_export.py` | 导出各检查点嵌入供离线核验 |
| `verify_embed.py` | 从导出嵌入重算NDCG@10，核对与日志一致 |
| `verify_path.py` | 路径指标独立复算 |

## 八、权重分布与对齐诊断（RQ4）

| 文件 | 作用 |
|------|------|
| `bloom_map.py` | 技能词按布鲁姆认知层级映射 |
| `kappa_align.py` | 对齐标注一致性Kappa核验 |

## 九、结果聚合与可视化

| 文件 | 作用 |
|------|------|
| `agg_gate.py` | 门控轨迹结果聚合 |
| `agg_newS9.py` | 新版s9重跑结果多指标聚合 |
| `supp_stats_and_tables.py` | 补充实验全量统计汇总（Top-K曲线、技能覆盖缺口、掌握度负对照检验、冷启动单列评估、第8节五种子归因与配对检验） |
| `rerun_all.py` | T1/T2/T3全量重跑入口 |
| `merge_skillgcn_logs.py` | 技能模型多次运行日志合并 |
| `fig_common.py` | 绘图共享配色与样式 |
| `fig_draw.py` | 主结果图绘制 |
| `exp6_viz.py` | 技能权重、对齐、路径对比三图绘制 |

## 十、后台运行脚本（shell）

负载闸门下的守护式批量运行，内含随机种子循环与失败重试。注意部分脚本含绝对路径，迁移到新机器需改首行`cd`。

| 文件 | 作用 |
|------|------|
| `embed_runner.sh` | 嵌入导出守护 |
| `exp5_hyper_daemon.sh` | 超参扫描守护 |
| `night_probe.sh` | 门控轨迹探测守护 |
| `rerun_daemon.sh` | 多随机种子消融守护 |
| `supp_daemon.sh` | 补充实验守护 |
| `supp_daemon_g.sh` | 定值门控补充实验守护 |
| `supp_daemon_learned.sh` | 学习门控值重训练守护 |
| `supp_all_runner.sh` | 一键补充实验总控脚本（自动跳过已完成JSON，依次执行统计汇总、固定门控扫描、有界与正则化门控消融、路径评估与可视化） |

## 典型执行顺序

先依次运行数据流水线`s1`至`s9`生成`../中间产物`，再运行主对照`layer1_check.py`与`layer1_t2.py`，随后按需运行门控诊断、路径评估与补充实验，最后用`verify_embed.py`与`verify_path.py`复核，用`fig_draw.py`与`exp6_viz.py`出图。
