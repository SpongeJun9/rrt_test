"""Check generated control limits and footprint against the actual navigation YAML."""
from pathlib import Path
import copy
import json
import math
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rag_rrt_real_validation'))
from rag_rrt_real_validation.stack_config import prepare_navigation, prepare_stack
from rag_rrt_real_validation.geometry import planning_radius


class StackConfigTests(unittest.TestCase):
    def setUp(self):
        self.site = yaml.safe_load((ROOT/'src/rag_rrt_real_validation/config/site.yaml').read_text())
        # An explicit CLI path also permits validating against a deployed robot config.
        nav_path = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else (
            ROOT.parents[1]/'src/pb2025_sentry_nav-main/pb2025_nav_bringup/config/reality/nav2_params.yaml')
        self.nav = yaml.safe_load(nav_path.read_text(encoding='utf-8'))

    def test_source_preserved_and_diagonal_speed_bounded(self):
        before = copy.deepcopy(self.nav)
        trial = prepare_navigation(self.nav, self.site)
        self.assertEqual(self.nav, before)
        p = trial['velocity_smoother']['ros__parameters']
        self.assertLessEqual(math.hypot(*p['max_velocity'][:2]), self.site['max_linear_speed_mps']+1e-10)
        self.assertEqual(p['max_velocity'][2], 0.)
        self.assertEqual(trial['fake_vel_transform']['ros__parameters']['init_spin_speed'], 0.)

    def test_footprint_contains_body_margin_and_rotation_disabled(self):
        trial = prepare_navigation(self.nav, self.site)
        for name in ('local_costmap', 'global_costmap'):
            polygon = json.loads(trial[name][name]['ros__parameters']['footprint'])
            self.assertAlmostEqual(max(p[0] for p in polygon), self.site['robot_length_m']/2+self.site['body_margin_m'])
            self.assertAlmostEqual(max(p[1] for p in polygon), self.site['robot_width_m']/2+self.site['body_margin_m'])
        controller = trial['controller_server']['ros__parameters']
        self.assertFalse(controller['FollowPath']['enable_rotation'])
        self.assertEqual(controller['min_y_velocity_threshold'], .001)

    def test_invalid_dimensions_rejected(self):
        for key, bad in [('max_linear_speed_mps', 0), ('robot_length_m', -1), ('body_margin_m', math.nan)]:
            site = dict(self.site); site[key] = bad
            with self.assertRaises(ValueError):
                prepare_navigation(self.nav, site)

    def test_original_profile_preserves_every_navigation_parameter(self):
        before, site_before = copy.deepcopy(self.nav), copy.deepcopy(self.site)
        nav, settings = prepare_stack(self.nav, self.site)
        self.assertEqual(nav, before)
        self.assertEqual(self.nav, before)
        self.assertEqual(self.site, site_before)
        self.assertEqual(settings['navigation_profile'], 'original')
        self.assertEqual(settings['base_frame'], 'gimbal_yaw_fake')
        self.assertAlmostEqual(planning_radius(settings), .2)
        self.assertAlmostEqual(settings['max_linear_speed_mps'], math.hypot(2.5, 2.5))
        self.assertAlmostEqual(settings['endpoint_tolerance_m'], .15)
        saved = json.loads(settings['expected_navigation_parameters'])
        self.assertEqual(saved['/controller_server']['FollowPath.v_linear_max'], 8.)
        self.assertEqual(nav['fake_vel_transform']['ros__parameters']['init_spin_speed'], 3.14)

    def test_original_polygon_radius_is_derived_without_changing_polygon(self):
        self.nav['global_costmap']['global_costmap']['ros__parameters'].update(
            footprint='[[0.3,0.4],[0.3,-0.4],[-0.3,-0.4],[-0.3,0.4]]', footprint_padding=.02)
        before = copy.deepcopy(self.nav)
        nav, settings = prepare_stack(self.nav, self.site, 'original')
        self.assertEqual(nav, before)
        self.assertAlmostEqual(planning_radius(settings), .52)

    def test_validation_profile_keeps_explicit_trial_limits(self):
        nav, settings = prepare_stack(self.nav, self.site, 'validation')
        self.assertEqual(nav, prepare_navigation(self.nav, self.site))
        self.assertAlmostEqual(planning_radius(settings), math.hypot(.35, .35))
        with self.assertRaisesRegex(ValueError, 'nav_profile'):
            prepare_stack(self.nav, self.site, 'unknown')


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
