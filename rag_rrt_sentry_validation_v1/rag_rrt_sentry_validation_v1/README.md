# RAG-RRT* 哨兵全向机器人验证包 v1

目标环境：Ubuntu 22.04、ROS 2 Humble。保留设备拥有者已运行的雷达、定位、Nav2 和底盘驱动；本包负责单次规划、可选路径执行及结果记录。

请先阅读 **使用说明.md**。完整算法依赖已随包提供，不需要配置研究者电脑路径、CNN 权重或 Jetson。

| 操作 | 命令（包根目录） |
|---|---|
| 首次编译 | `bash scripts/build.sh` |
| 只检查接口和参数 | `bash scripts/run_trial.sh --check-only` |
| 预览低速/足迹配置修改 | `bash scripts/configure_trial.sh` |
| 在机器人停稳后应用配置 | `bash scripts/configure_trial.sh --apply` |
| RViz 选目标，只规划 | `bash scripts/run_trial.sh --clicked-goal --method rag --seed 0` |
| RViz 选目标，规划并执行 | `bash scripts/run_trial.sh --clicked-goal --method rag --seed 0 --execute` |

RViz 请使用 **Publish Point**，不使用会触发原导航任务的 **2D Goal Pose**。方法支持 `rrt`、`informed`、`rag`，结果中保存完整算法名称。

本包已完成离线实际算法检查、几何与流程测试，以及 Python 安装结构检查。当前助手环境没有 ROS，**完整 Humble 编译和实车验收仍需在设备上完成**。不能据此宣称已有实物实验结果。

首次先做一次检查和只规划，再做一条短而非零的执行任务。正式任务、种子和参数在验收通过后冻结；不自动切换种子、换算法、截断路径或重新规划以掩盖失败。
