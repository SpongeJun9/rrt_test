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

1. 保持车体静止，观察第 3 份指南的地图与定位对齐。
2. 等待终端提示 RViz 目标输入。默认等候目标 120 秒；定位调试较久时在 site.yaml 增大 `click_timeout_sec`，然后重启。
3. 使用专用 RViz 的 **2D Goal Pose**：在可通行区域点击、拖动、释放。
4. 程序冻结首个目标，获取当前定位起点，调用一次 RAG 服务，校验并加密返回折线，再发送一次 FollowPath。
5. 完成/失败后启动组退出，新任务重新运行命令。执行期间再次点击不会换目标、不会触发第二次规划。

工具必须是本包 `rviz_default_plugins/SetGoal`，话题 `/rag_rrt/goal_pose`。原 `nav2_rviz_plugins/GoalTool` 外观相近，却会发起原导航 Action，不能用于本试验。

地图变化、执行异常、定位丢失等按原验证程序退出/取消，不自动重规划、不换算法、不换种子。Ctrl+C 尝试取消本次 FollowPath；若提示 `STOP NOT CONFIRMED`，使用原遥控接管/急停。

## 方法版本与参数

默认方法：`rag_bpc_conn_init_rel_only_topk_fallback`。

无 TopK 版本：

```bash
bash scripts/start_trial.sh world:=rmul2026 \
  method:=rag_bpc_conn_init_rel_only seed:=0
```

可用参数包括 `map`、`prior_pcd_file`、`params_file`、`serial_config`、`site_config`、`use_robot_state_pub`、`method`、`seed`、`output_dir`。完整启动只支持无命名空间实车接口。方法改变后，论文方法名也必须对应。

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
