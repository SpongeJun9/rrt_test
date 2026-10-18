import math
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src/rag_rrt_real_validation'))
sys.path.insert(0,str(root/'src/rag_rrt_backend_py'))
from rag_rrt_real_validation.geometry import required_radius,footprint_covers,resample,distance_to_path
from rag_rrt_backend_py.final_path_validation import validate_polyline
from rag_rrt_backend_py.execution_grid_map import execution_map_class
import numpy as np

class GeometryTests(unittest.TestCase):
    def test_radius(self):
        self.assertAlmostEqual(required_radius(.6,.6,.05),math.sqrt(.35**2*2))
        with self.assertRaises(ValueError):required_radius(-1,.6,.05)

    def test_footprint(self):
        self.assertFalse(footprint_covers('',.2,.6,.6,.05))
        self.assertTrue(footprint_covers('[]',.5,.6,.6,.05))
        self.assertTrue(footprint_covers('[[.35,.35],[.35,-.35],[-.35,-.35],[-.35,.35]]',0,.6,.6,.05))
        self.assertFalse(footprint_covers('[[.2,.2],[.2,-.2],[-.2,-.2],[-.2,.2]]',.5,.6,.6,.05))
        self.assertFalse(footprint_covers('[[0,0],[1,0],[2,0]]',0,.6,.6,.05))

    def test_path_vertices_and_endpoint(self):
        path=resample([(0.,0.),(.23,0.),(.23,.31)],.05)
        self.assertIn((.23,0.),path);self.assertEqual(path[-1],(.23,.31))
        self.assertTrue(all(math.dist(a,b)<=.05+1e-9 for a,b in zip(path,path[1:])))
        with self.assertRaises(ValueError):resample([(0.,0.),(0.,0.)],.05)
        with self.assertRaises(ValueError):resample([(0.,0.),(1.,1.)],0)

    def test_path_distance(self):
        self.assertAlmostEqual(distance_to_path((1.,.5),[(0.,0.),(2.,0.)]),.5)
        self.assertAlmostEqual(distance_to_path((3.,0.),[(0.,0.),(2.,0.)]),1.)

    def test_conservative_collision(self):
        mask=np.ones((7,7));mask[3,3]=0
        self.assertFalse(validate_polyline(mask,[(1,2.5),(5,2.5)])[0])
        self.assertFalse(validate_polyline(mask,[(1,2),(5,2)],1)[0])
        self.assertTrue(validate_polyline(mask,[(1,2),(5,2)],0)[0])
        self.assertFalse(validate_polyline(mask,[(math.nan,1),(2,2)])[0])

    def test_planner_statistics(self):
        sys.path.insert(0,str(root/'src/rag_rrt_backend_py/algorithm'))
        from envs.grid_map_2d import GridMap2D
        cls=execution_map_class(GridMap2D);mask=np.ones((7,7));mask[3,3]=0
        grid=cls(mask,clearance=1)
        self.assertFalse(grid.edge_valid((1,2),(5,2)))
        self.assertEqual(grid.collision_checks,1)
        self.assertGreater(grid.collision_check_samples,0)
        self.assertGreaterEqual(grid.collision_check_time,0)

if __name__=='__main__':unittest.main()
