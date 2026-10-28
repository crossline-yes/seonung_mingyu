"""Regression checks for artifact integrity and mask coordinate safety."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / 'scripts'))

from PIL import Image
from mask_audit import validate_masks
from pipeline_contracts import (read_names, validate_scaled_models,
    fingerprints, verify_fingerprints, write_json)
from run_pipeline import Pipeline


def model(root, width, height, k, pose=None, model_id=1):
    root.mkdir()
    (root/'cameras.bin').write_bytes(struct.pack('<Qi i Q Q 4d',1,1,model_id,width,height,*k))
    pose = pose or [1.,0.,0.,0.,0.,0.,0.]
    data = struct.pack('<Q',3)
    for idx in range(3):
        data += struct.pack('<i7di',idx+1,*pose,1) + f'{idx}.png'.encode() + b'\0' + struct.pack('<Q',0)
    (root/'images.bin').write_bytes(data)


class ContractsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_scaled_intrinsics_and_pose_required(self):
        names = ['0.png','1.png','2.png']
        model(self.root/'source',100,80,[100.,110.,50.,40.])
        model(self.root/'good',50,40,[50.,55.,25.,20.])
        self.assertEqual(validate_scaled_models(self.root/'source',self.root/'good',names)['0.png']['dense'],(50,40))
        model(self.root/'bad_k',50,40,[50.,55.,24.,20.])
        with self.assertRaisesRegex(ValueError,'intrinsics'):
            validate_scaled_models(self.root/'source',self.root/'bad_k',names)
        model(self.root/'bad_pose',50,40,[50.,55.,25.,20.],[1.,0.,0.,0.,1.,0.,0.])
        with self.assertRaisesRegex(ValueError,'pose'):
            validate_scaled_models(self.root/'source',self.root/'bad_pose',names)

    def test_distorted_camera_cannot_be_resized_as_pinhole(self):
        model(self.root/'source',100,80,[100.,50.,40.,.1],model_id=2)
        model(self.root/'dense',50,40,[50.,50.,25.,20.])
        with self.assertRaisesRegex(ValueError,'PINHOLE'):
            validate_scaled_models(self.root/'source',self.root/'dense',['0.png','1.png','2.png'])

    def test_masks_require_complete_binary_nonempty_matching_images(self):
        images = self.root/'images'; images.mkdir()
        masks = self.root/'masks'; masks.mkdir()
        Image.new('RGB',(10,8)).save(images/'x.png')
        with self.assertRaisesRegex(ValueError,'Missing'):
            validate_masks(images,['x.png'],masks)
        for value in (0,128):
            Image.new('L',(10,8),value).save(masks/'x.png.png')
            with self.assertRaisesRegex(ValueError,'nonbinary or empty'):
                validate_masks(images,['x.png'],masks)
        Image.new('L',(10,8),255).save(masks/'x.png.png')
        self.assertEqual(validate_masks(images,['x.png'],masks)['count'],1)

    def test_selected_image_cannot_escape_root(self):
        images = self.root/'images'; images.mkdir()
        for idx in range(3):
            (images/f'{idx}.png').write_bytes(b'input')
        manifest = self.root/'images.txt'
        manifest.write_text('0.png\n1.png\n../images.txt\n')
        with self.assertRaisesRegex(ValueError,'unsafe'):
            read_names(manifest,images)

    def test_reused_file_change_is_rejected(self):
        path = self.root/'artifact'; path.write_bytes(b'before')
        frozen = fingerprints([path])
        path.write_bytes(b'after')
        with self.assertRaisesRegex(ValueError,'changed'):
            verify_fingerprints(frozen)

    def test_resume_preserves_failed_attempt_and_skips_completed_stage(self):
        cfg = {'name':'retry_test'}
        calls = []
        def imported(pipeline, out):
            calls.append('import')
            write_json(out/'inputs.json',{})
        def fail(pipeline,out):
            (out/'partial.txt').write_text('keep failed output')
            raise RuntimeError('simulated interruption')
        with patch('run_pipeline.WORKSPACE',self.root), patch.object(Pipeline,'code_hashes',return_value={}):
            with patch.object(Pipeline,'stage_import',imported), patch.object(Pipeline,'stage_fusion_masks',fail):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    Pipeline(cfg).run('fusion_masks')
            with patch.object(Pipeline,'stage_fusion_masks',lambda p,o: (o/'success.txt').write_text('ok')):
                Pipeline(cfg,resume=True).run('fusion_masks')
            out = self.root/'실행결과/retry_test'
            self.assertTrue((out/'fusion_masks_01/partial.txt').exists())
            self.assertTrue((out/'fusion_masks_02/success.txt').exists())
            self.assertEqual(calls,['import'])
            with self.assertRaisesRegex(ValueError,'Configuration changed'):
                Pipeline({**cfg,'modified':True},resume=True)


if __name__ == '__main__':
    unittest.main()
