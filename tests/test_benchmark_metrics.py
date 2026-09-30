import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from benchmarks.metrics import evaluate
from benchmarks.schema import Annotation, EvaluationConfig

class BenchmarkMetricTests(unittest.TestCase):
    def test_initial_baseline_with_resource_failure_cannot_pass(self):
        from benchmarks.runner import run
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (10, 10), 'white').save(root/'scene.png')
            annotation = {'scene_id': 'failed_inference', 'image': 'scene.png',
                          'roi': [0, 0, 10, 10], 'objects': [],
                          'annotation_status': 'reviewed', 'provenance': 'test fixture'}
            (root/'annotation.json').write_text(json.dumps(annotation))
            manifest = {'mock': False, 'objects': [], 'resource_failures': [{'model': 'sam', 'reason': 'oom'}]}
            (root/'scene.json').write_text(json.dumps(manifest))
            report = run(root/'annotation.json', root/'report', root/'unused.yaml',
                         root/'unused_models.yaml', manifest_path=root/'scene.json')
            self.assertEqual(report['gate']['status'], 'FAIL')
            self.assertEqual(report['gate']['checks']['execution']['status'], 'FAIL')
            self.assertTrue((root/'report'/'benchmark_report.json').exists())

    def test_similarity_does_not_define_semantic_metrics(self):
        from cv.completion_metrics import completion_metrics
        m=completion_metrics({'width':10,'height':10,'reconstruction_score':1.0,'objects':[], 'ownership':{'eligible_pixels':100,'residual_background':{'pixel_count':90},'unassigned_ratio':.9}})
        self.assertEqual(m['reconstruction']['similarity'],1.0); self.assertAlmostEqual(m['scene']['semantic_coverage'],.1)
