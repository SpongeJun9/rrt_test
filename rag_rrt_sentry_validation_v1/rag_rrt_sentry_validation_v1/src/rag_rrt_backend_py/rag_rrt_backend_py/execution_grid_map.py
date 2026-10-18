"""Use one conservative edge checker for all planners, without editing core code."""
import time
from rag_rrt_backend_py.final_path_validation import validate_polyline


def execution_map_class(base):
    class ExecutionGridMap(base):
        def edge_valid(self, start, end, resolution=1.0):
            started = time.perf_counter()
            self.collision_checks += 1
            try:
                ok, checks, _ = validate_polyline(self.binary_mask, [start, end], self.clearance)
                self.collision_check_samples += checks
                return ok
            finally:
                self.collision_check_time += time.perf_counter() - started
    return ExecutionGridMap
