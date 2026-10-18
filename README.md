# rrt_test：哨兵导航工作空间与 RAG-RRT* 实车验证

本仓库包含原机器人工作空间 `src/` 和 RAG 单次规划验证包。验证流程为：定位与感知持续运行 → RViz 选择一个目标 → RAG 仅规划一次 → 原全向 PID 控制器执行 → 保存本次结果。执行中不重规划、不切换 Theta*、不自动更换种子；新试验重新启动。

原驱动、重定位、地形感知、控制、速度处理与串口全部保留。比赛决策与 Nav2 导航行为树不参与验证。

完整启动默认 `nav_profile:=original`，直接保留 `src/pb2025_sentry_nav-main/pb2025_nav_bringup/config/reality/nav2_params.yaml` 的算法与参数，只接入一次 RAG 全局规划。控制速度、PID、前瞻、代价地图外廓、膨胀和速度变换均不再被验证配置覆盖；RAG 净空半径从原全局代价地图外廓读取。此前的低速/矩形外廓验证设置可显式选择 `nav_profile:=validation`。

## 按顺序阅读

1. [环境安装与编译](docs/01-环境安装与编译.md)
2. [建图与地图保存](docs/02-建图与地图保存.md)
3. [定位与硬件接口检查](docs/03-定位与硬件接口检查.md)
4. [RViz 单次规划与执行](docs/04-RViz单次规划与执行.md)
5. [论文试验设计与结果整理](docs/05-论文试验与结果整理.md)
6. [常见问题](docs/06-常见问题.md)
7. [后续上传与同步](docs/07-上传与同步.md)

## 下载

PCD 地图和 ZIP 文件由 Git LFS 保存，必须取回实际内容：

```bash
git lfs install
git clone https://github.com/SpongeJun9/rrt_test.git
cd rrt_test
git lfs pull
git lfs fsck
```

## 机器人上的运行入口

Ubuntu 22.04 / ROS 2 Humble；安装及设备配置见第 1 份指南。

```bash
# 构建导航与 RAG 两个工作区
bash scripts/build_workspace.sh --install-deps

# 建图：先遥控采集并保存地图，不与验证入口同时运行
bash scripts/start_mapping.sh

# 地图保存并结束建图后，再启动单次验证
bash scripts/start_trial.sh \
  map:="$HOME/sentry_maps/test_site/test_site.yaml" \
  prior_pcd_file:="$HOME/sentry_maps/test_site/test_site.pcd"
```

`--method rag` 的试验脚本别名对应 TopK-Fallback 版本；完整启动的默认方法也是 `rag_bpc_conn_init_rel_only_topk_fallback`。无 TopK 版本需显式指定，详见第 4、5 份指南。

## 目录

| 目录 | 内容 |
|---|---|
| `src/` | 原 59 个 ROS 包，包括导航、串口、视觉、决策、仿真和依赖源码 |
| `rag_rrt_sentry_validation_v1/rag_rrt_sentry_validation_v1/` | 独立 RAG 工作区，3 个 ROS 包与离线验证程序 |
| `scripts/` | 根目录构建、建图、验证与完整源码暂存入口 |
| `docs/` | 按步骤执行的中文指南、来源与验证状态 |
| `rag_*.zip` | 历史验证交付，通过 LFS 保存；最新指南及修改以展开源码为准 |

14 个原嵌套仓库以普通源码快照纳入此仓库，下载无需初始化子模块。原本地 `.git` 元数据保留在原位置，不上传到主仓库。[来源记录](docs/upstream-repositories.json)包含各仓库 URL 和本地提交；以此仓库中的实际文件为运行版本。

不上传 `build/install/log`、缓存、Git 元数据与本地秘密文件。原始 `src.zip` 内含 603 个 Git 元数据条目，作为本地重复备份保留，不上传；完整展开源码 `src/` 已纳入仓库。既有 SDK 动态库、模型资源和地图保留。新增地图建议存到 `~/sentry_maps`，实验结果默认存到 `~/rag_rrt_real_results`。

## 验证边界

已进行 Python/配置/离线算法与模拟 ROS 流程检查。2026-10-18 另在 Ubuntu 22.04 / Humble x86_64 实机完成导航 21 个包及 RAG 3 个包的编译、真实 RAG 服务请求及 NoMachine RViz 显示检查；传感器、串口协议、实际建图、重定位和行驶尚未验证。详情见 [验证状态](docs/验证状态.md)。

实车必须核对雷达网络、串口设备、真实车体尺寸和 TF。原云台关节反馈接口不完整，详见第 3 份指南。现有 `small_gicp` 构建从上游 `master` 获取代码，首次编译需要网络且尚未锁定版本；源码内第三方许可证应继续保留。
