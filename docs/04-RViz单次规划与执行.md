# 04 RViz 单次规划与执行

## 一次任务

完成建图并结束建图进程后：

```bash
cd ~/rrt_test
bash scripts/start_trial.sh \
  map:="$HOME/sentry_maps/test_site/test_site.yaml" \
  prior_pcd_file:="$HOME/sentry_maps/test_site/test_site.pcd"
```

或者，仅当现有同名地图确实是当前场地时：

```bash
bash scripts/start_trial.sh world:=rmul2026
```

统一入口启动原雷达、Point-LIO、重定位、模型 TF、地图、障碍感知、控制、速度平滑、速度变换、串口、RAG 后端及专用 RViz。不启动比赛决策、bt_navigator 或实时全局 SLAM。保留 planner_server 只为提供原全局代价地图及参数检查，没有请求端调用其 Theta*。

默认 `nav_profile:=original` 保留原导航参数，只改变全局路径的来源。RAG 从原全局代价地图的 `robot_radius`/footprint 读取净空，当前圆形半径为 0.2 m，在 0.05 m 地图中对应 4 格（同时保留算法最少 2 格的下限），不再套用此前预设外廓的 10 格。控制器和平滑器仍使用原速度参数。地图路径替换、空命名空间展开和 YAML 包路径替换属于启动输入处理，不改变这些算法参数。

如要使用此前的低速验证参数，在同一命令末尾增加 `nav_profile:=validation`。启动会保存 `navigation.yaml`、派生后的 `site.yaml` 和含 profile 的 `inputs.yaml`。原配置的 0.2 m 半径是否覆盖实车，需要用真实尺寸核对；恢复该参数不会自动解决 GICP 未收敛或地图起点不可通行。

1. 保持车体静止，用 **2D Pose Estimate** 给出当前位置/朝向，观察第 3 份指南的蓝色 PCD 与黄色实时点云对齐。黑白栅格地图仍会正常显示。
2. 等待终端提示 RViz 目标输入。默认等候目标 120 秒；定位调试较久时在 site.yaml 增大 `click_timeout_sec`，然后重启。
3. 在 `Global Costmap` / `Local Costmap` 的彩色叠加图上观察可通行区，使用专用 RViz 的 **2D Goal Pose** 点击、拖动、释放。固定坐标系保持 `map`；显示代价图不改变 Goal 的坐标系。
4. 程序冻结首个目标，获取当前定位起点，调用一次 RAG 服务，校验并加密返回折线，再发送一次 FollowPath。
5. 完成/失败后启动组退出，新任务重新运行命令。执行期间再次点击不会换目标、不会触发第二次规划。

工具必须是本包 `rviz_default_plugins/SetGoal`，话题 `/rag_rrt/goal_pose`。原 `nav2_rviz_plugins/GoalTool` 外观相近，却会发起原导航 Action，不能用于本试验。

`RAG single planned path` 显示绿色参考路径；黑白底图、彩色代价图和绿色路径可同时显示。当前 RAG 读取原始 `/map` 并按原配置车体半径做自身碰撞检查；代价图是原 Nav2 感知/膨胀结果的叠加显示，不代表 RAG 改为优化 Nav2 的膨胀代价值。

若日志出现 `Exec format error`，先停止本次启动并在根目录重新运行 `bash scripts/build_workspace.sh`。2026-10-18 实机发现两个 RAG 入口曾为空文件，导致后端和目标接收节点均未启动；现已加入构建后校验与 launch 启动前检查。等待终端出现 `waiting on /rag_rrt/goal_pose` 后再给目标；若超过目标等待时间，需要重启本次任务。

地图变化、执行异常、定位丢失等按原验证程序退出/取消，不自动重规划、不换算法、不换种子。Ctrl+C 尝试取消本次 FollowPath；若提示 `STOP NOT CONFIRMED`，使用原遥控接管/急停。

## 方法版本与参数

默认方法：`rag_bpc_conn_init_rel_only_topk_fallback`。

无 TopK 版本：

```bash
bash scripts/start_trial.sh world:=rmul2026 \
  method:=rag_bpc_conn_init_rel_only seed:=0
```

可用参数包括 `map`、`prior_pcd_file`、`params_file`、`nav_profile`、`serial_config`、`site_config`、`use_robot_state_pub`、`method`、`seed`、`output_dir`。完整启动只支持无命名空间实车接口。方法改变后，论文方法名也必须对应。

若原硬件系统已单独运行，只想先检查或只规划，可用独立入口：

```bash
export SENTRY_UNDERLAY_SETUP="$HOME/rrt_test/install/setup.bash"
cd rag_rrt_sentry_validation_v1/rag_rrt_sentry_validation_v1
bash scripts/run_trial.sh --check-only
# 打开本包专用 RViz 后，单次规划但不执行
bash scripts/run_trial.sh --rviz-goal --method rag --seed 0
```

它不启动原系统，不与 full_stack 重复运行。完整入口本身已经执行同样的参数预检。

## 结果目录

默认 `~/rag_rrt_real_results/stack_*/`，启动配置 `navigation.yaml`、`site.yaml` 和输入路径 `inputs.yaml` 位于该层；任务数据在其下独立目录。绿色 Path 是送入控制器的参考路径，并非实际行驶轨迹。实际轨迹在 CSV 中。
