import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rag_rrt_backend_py'))
from rag_rrt_backend_py.occupancy_grid_map import MapGeometry,OccupancyGridMap
from rag_rrt_backend_py.rrt_star_adapter import RRTStarAdapter,RRTStarAdapterConfig

records=[]
for origin_yaw in [0.0,0.4]:
    mask=np.zeros((160,160),dtype=np.int16);mask[65:95,65:95]=100
    geometry=MapGeometry(width=160,height=160,resolution=0.05,origin_x=-4.,origin_y=-4.,origin_yaw=origin_yaw)
    grid=OccupancyGridMap(geometry=geometry,data=mask.ravel().tolist())
    for method in ['rrt_star','irrt_star','rag_bpc_conn_init_rel_only_topk_fallback']:
        adapter=RRTStarAdapter(RRTStarAdapterConfig(project_root=str(ROOT/'src/rag_rrt_backend_py/algorithm'),
                                                  clearance_cells=10,iter_max=5000,iter_after_initial=300))
        result=adapter.plan(grid,geometry.planner_to_world(25.,80.),geometry.planner_to_world(135.,80.),
                            seed=0,planner_name=method,collect_debug_tree=False)
        assert result.success and result.world_path is not None,result.message
        assert result.planner_result.metadata['final_path_validation_success']
        assert adapter._astar_prior_class.__name__=='AStarPriorGenerator'
        if method.startswith('rag_'):
            assert result.planner_result.metadata['use_topk_fallback']
        record=dict(method=method,map_origin_yaw=origin_yaw,seed=0,clearance_cells=10,
                    success=result.success,path_vertices=len(result.world_path))
        records.append(record)
        print(json.dumps(record),flush=True)
print('OFFLINE_ALGORITHM_CHECK_PASS: '+str(len(records))+' runs; not formal experiments')
