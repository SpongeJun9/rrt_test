# RViz 目标位姿 → RAG 单次规划 → 原控制器执行

本入口不使用比赛决策，也不调用 Nav2 的导航行为树。一次启动接收一个目标，调用一次 RAG 规划、发送一次 FollowPath，任务完成或失败后退出。执行中再次点击不替换目标、不重规划。再次试验需重新运行命令。

## 1. 在机器人 Linux / ROS 2 Humble 上准备

保留原雷达、定位、TF、地图、controller_server、速度平滑、速度坐标变换与串口运行。按原使用说明核对 site.yaml 中车体尺寸和接口，并完成低速配置；本次修改没有替你修改实车参数。

在本目录构建：

```bash
bash scripts/build.sh
bash scripts/run_trial.sh --check-only
```

只读检查应通过后再执行。如果原工程使用独立 underlay，按原说明设置 SENTRY_UNDERLAY_SETUP。

## 2. 打开专用 RViz

在本目录另一终端执行：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
rviz2 -d src/rag_rrt_real_validation/config/rag_single_plan.rviz
```

专用配置使用 rviz_default_plugins/SetGoal，向 /rag_rrt/goal_pose 发布 PoseStamped。原工程的 nav2_rviz_plugins/GoalTool 会直接发起原导航任务，不应使用它来做本试验；两种工具外观相近但调用方式不同。

## 3. 启动并点击目标

```bash
bash scripts/run_trial.sh --rviz-goal --method rag --seed 0 --execute
```

终端提示等待 RViz 目标后，在专用 RViz 点击 2D Goal Pose，按住鼠标拖动设定朝向并释放。绿色路径显示本次送入控制器的路径。默认等候目标 120 秒，可在 site.yaml 修改 click_timeout_sec。

--method rag 对应带 TopK-Fallback 的 rag_bpc_conn_init_rel_only_topk_fallback。若论文指定无 TopK 版本，使用 --method rag_bpc_conn_init_rel_only。当前跟踪配置关闭旋转，接收目标朝向不代表验证了到达朝向控制。

结果默认保存在 ~/rag_rrt_real_results，每次独立目录，新增 rviz_goal.json 保存原始目标。失败不重试、不切换算法；规划与执行异常沿用原验证程序的退出、取消及记录逻辑。

## 当前验证边界

本机验证了模拟 ROS 传输下的一次规划/一次执行、重复目标不替换、坐标系及四元数拒绝行为。未在本机编译 ROS，未验证 RViz 实际加载和实车运行。原 zip 保持原样；新增功能位于当前解压目录，部署时请使用更新后的目录或 RViz 单次规划版本压缩包。
