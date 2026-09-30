"""Real adapter contracts without downloads; neural inference is a separate smoke."""
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from pydantic import ValidationError

from app.models import ModelConfig, SAMConfig, load_models
from services.detection_postprocess import postprocess_detections, normalize_label
from services.grounding_service import GroundingService
from services.sam_service import SAMService
from services.model_support import ModelUnavailableError, torch_runtime


class DetectionTests(unittest.TestCase):
    def test_clipping_class_aware_nms_and_invalid_predictions(self) -> None:
        boxes = [[-2.2, 1.5, 12.1, 8.2], [0, 2, 12, 8], [0, 2, 12, 8], [5, 2, 4, 3], [0, 0, float('nan'), 3], [40, 0, 45, 2]]
        values = postprocess_detections(boxes, [.9,.8,.7,.95,.9,.8], ['tree','tree','rock','tree','tree','tree'], ['tree','rock'], 20, 10, .3, .5, 10)
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0]['bbox'], {'x':0,'y':1,'w':13,'h':8})
        self.assertEqual([obj['category'] for obj in values], ['tree', 'rock'])

    def test_labels_are_canonical_and_ambiguity_is_rejected(self) -> None:
        self.assertEqual(normalize_label('a green tree.', ['tree','rock']), 'tree')
        self.assertIsNone(normalize_label('tree rock', ['tree','rock']))
        self.assertIsNone(normalize_label('unknown', ['tree','rock']))
        self.assertEqual(normalize_label('pine tree', ['tree','pine tree']), 'pine tree')

    def test_threshold_limit_and_shape_validation(self) -> None:
        result = postprocess_detections([[0,0,2,2],[5,5,7,7]], [.2,.8], ['tree','tree'], ['tree'],10,10,.3,.5,1)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['confidence'],.8)
        with self.assertRaises(ValueError):
            postprocess_detections([], [.5], [], [],10,10,.3,.5,10)

    def test_real_grounding_calls_processor_with_original_size(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)/'image.png'
            Image.new('RGB',(20,10)).save(path)
            service = GroundingService(False)
            inputs = MagicMock()
            inputs.to.return_value = inputs
            inputs.__getitem__.return_value = 'input_ids'
            processor = MagicMock(return_value=inputs)
            boxes, scores = MagicMock(), MagicMock()
            boxes.detach.return_value.cpu.return_value.tolist.return_value = [[2,3,8,9]]
            scores.detach.return_value.cpu.return_value.tolist.return_value = [.8]
            processor.post_process_grounded_object_detection.return_value = [{'boxes':boxes,'scores':scores,'text_labels':['tree']}]
            service._model = MagicMock()
            service._processor = processor
            service._torch = SimpleNamespace(inference_mode=nullcontext)
            service._device = 'cpu'
            result = service.detect(str(path), ['tree'])
            self.assertEqual(result[0]['bbox'],{'x':2,'y':3,'w':6,'h':6})
            self.assertEqual(processor.call_args.kwargs['text'],'tree.')
            self.assertEqual(processor.post_process_grounded_object_detection.call_args.kwargs['target_sizes'],[(10,20)])


class SAMAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name)/'image.png'
        Image.new('RGB',(20,10)).save(self.path)
        self.record = {'id':'tree_001','category':'tree','confidence':.8,'bbox':{'x':2,'y':3,'w':6,'h':6}}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def service(self) -> SAMService:
        service = SAMService(False, ModelConfig(sam=SAMConfig(multimask_output=True)))
        service._torch = SimpleNamespace(inference_mode=nullcontext)
        service._predictor = MagicMock()
        return service

    def test_xyxy_best_mask_binary_alpha_and_identity(self) -> None:
        service = self.service()
        masks = np.zeros((3,10,20), dtype=np.float32)
        masks[1,4:8,3:6] = 1
        service._predictor.predict.return_value = (masks, np.array([.2,.9,.4]), None)
        results = service.segment(str(self.path),[self.record, {**self.record,'id':'tree_002'}])
        self.assertEqual([obj['id'] for obj in results], ['tree_001','tree_002'])
        self.assertEqual(results[0]['confidence'], .8)
        self.assertEqual(results[0]['mask'].dtype, np.uint8)
        self.assertEqual(int(results[0]['mask'].sum()),12*255)
        np.testing.assert_array_equal(service._predictor.predict.call_args.kwargs['box'],[2,3,8,9])
        service._predictor.set_image.assert_called_once()
        service._predictor.reset_predictor.assert_called_once()
        self.assertNotIn('mask', self.record)

    def test_invalid_mask_is_rejected_and_predictor_reset(self) -> None:
        service = self.service()
        service._predictor.predict.return_value = (np.zeros((1,5,5)), np.array([.9]), None)
        with self.assertRaises(ValueError):
            service.segment(str(self.path),[self.record])
        service._predictor.reset_predictor.assert_called_once()

    def test_empty_detections_do_not_load_models(self) -> None:
        service = SAMService(False)
        with patch.object(service,'load') as load:
            self.assertEqual(service.segment('unused.png',[]),[])
            load.assert_not_called()

    def test_box_bounds_fail_before_model_loading(self) -> None:
        service = SAMService(False)
        record = {**self.record,'bbox':{'x':19,'y':3,'w':6,'h':6}}
        with self.assertRaises(ValueError), patch.object(service,'load') as load:
            service.segment(str(self.path),[record])
        load.assert_not_called()


class ModelConfigTests(unittest.TestCase):
    def test_default_cache_is_relative_to_yaml(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'models.yaml'
            path.write_text('{}', encoding='utf-8')
            self.assertEqual(load_models(path).cache_dir, path.resolve().parent / '.cache/models')

    def test_relative_paths_and_local_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'model.pt').touch()
            path=root/'models.yaml'
            path.write_text('cache_dir: cache\nsam:\n  checkpoint: model.pt\ngrounding:\n  model_id: ./dino\n', encoding='utf-8')
            config=load_models(path)
            self.assertEqual(config.cache_dir,root.resolve()/'cache')
            self.assertEqual(config.sam.checkpoint,root.resolve()/'model.pt')
            self.assertEqual(config.grounding.model_id,str(root.resolve()/'dino'))

    def test_bad_settings_fail_early(self) -> None:
        for data in ({'device':'metal'}, {'categories':[]}, {'categories':['tree. rock']}, {'grounding':{'box_threshold':2}}, {'sam':{'checkpoint':'/nonexistent/model.pt'}}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ModelConfig.model_validate(data)

    def test_device_choice_does_not_silently_replace_requested_device(self) -> None:
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False)))
        with patch.dict('sys.modules', {'torch':fake}):
            self.assertEqual(torch_runtime('auto')[1],'cpu')
            with self.assertRaises(ModelUnavailableError):
                torch_runtime('cuda')
            with self.assertRaises(ModelUnavailableError):
                torch_runtime('mps')
