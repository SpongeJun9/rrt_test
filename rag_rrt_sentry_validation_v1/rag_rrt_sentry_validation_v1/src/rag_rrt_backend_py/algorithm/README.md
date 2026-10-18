# Stage 6 最终实验收尾包

本包用于把 Stage 5 之后完成的：

- 最终 Relative-only learned-v4；
- ROS2/Gazebo P3-B4～P3-B5；
- 最终实验冻结决定；

并入项目的论文证据系统。

## 安装

```bash
unzip -o rag_rrt_stage6_final_experiment_closure.zip
cd rag_rrt_stage6_final_experiment_closure
bash scripts/apply_stage6_final_experiment_closure.sh /home/yanzu/rg_rrt
```

安装过程不会删除或覆盖 Stage 5 文件，而是新增：

```text
results/00_audit_reports/stage6_final_experiment_closure/
docs/04_paper_writing/02_section_drafts/
docs/04_paper_writing/03_material_index/
```

下一步按照 `final_remaining_experiment_gaps_cn.md`，优先生成 F3 效率—质量权衡图。
