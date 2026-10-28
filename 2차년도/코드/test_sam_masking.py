import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from sam_masking import prepare, validate_prompts
from mask_review import validate_review, ReviewRequired
from pipeline_contracts import read_json, write_json, sha256
from run_pipeline import Pipeline, config_from


class MaskingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.spec={'seed_image':'b.png','points_xy':[[2,3],[8,5]],'labels':[1,0],
            'box_xyxy':[1,1,7,7],'review_crop_xyxy':[0,0,10,8],
            'ordering':'explicit list','target':'test object'}

    def tearDown(self): self.temp.cleanup()

    def test_invalid_prompts_fail_before_inference(self):
        validate_prompts(self.spec,(10,8),['b.png','a.png','c.png'])
        for key,value in [('points_xy',[[10,3],[8,5]]),('labels',[1]),('box_xyxy',[7,1,2,7])]:
            spec=copy.deepcopy(self.spec); spec[key]=value
            with self.assertRaises(ValueError): validate_prompts(spec,(10,8),['b.png','a.png','c.png'])
        with self.assertRaisesRegex(ValueError,'seed'):
            validate_prompts(self.spec,(10,8),['a.png','b.png','c.png'])

    def test_preparation_preserves_declared_order_and_original_coordinates(self):
        images=self.root/'images'; images.mkdir()
        for name in ['a.png','b.png','c.png']: Image.new('RGB',(10,8),(50,100,150)).save(images/name)
        order=self.root/'images.txt'; order.write_text('b.png\na.png\nc.png\n')
        repo=self.root/'repo'; (repo/'sam2').mkdir(parents=True)
        (repo/'sam2/model.py').write_text('# fixture')
        checkpoint=self.root/'model.pt'; checkpoint.write_bytes(b'fixture')
        source=self.root/'source.json'
        write_json(source,{'repository_path':str(repo),'checkpoint':{'path':str(checkpoint),'sha256':sha256(checkpoint)}})
        cfg={'images':str(images),'image_list':str(order),'mask_generation':{**self.spec,'source_json':str(source)}}
        out=self.root/'prepared'; out.mkdir(); prepare(cfg,out)
        manifest=read_json(out/'manifest.json')
        self.assertEqual([r['image'] for r in manifest['records']],['b.png','a.png','c.png'])
        self.assertEqual(manifest['image_size'],[10,8])
        self.assertFalse(manifest['ground_truth_masks_used'])
        self.assertEqual(len(list((out/'frames').glob('*.jpg'))),3)
        self.assertEqual(sha256(images/'a.png'),read_json(out/'inputs.json')[str(images/'a.png')])

    def test_review_rejects_changed_artifacts_and_missing_pages(self):
        record={'outputs':{'mask':'hash'}}; pages={'one.jpg':['a'],'two.jpg':['b']}
        receipt={'kind':'masks','artifact_hashes':record['outputs'],'decision':'accept',
            'reviewer':'Codex','reviewer_type':'assistant','inspected_pages':list(pages),
            'scope':'overview','notes':'not pixel accuracy'}
        validate_review(receipt,record,pages,'masks')
        with self.assertRaisesRegex(ValueError,'version'):
            validate_review(receipt,{'outputs':{'mask':'changed'}},pages,'masks')
        with self.assertRaisesRegex(ValueError,'all generated'):
            validate_review({**receipt,'inspected_pages':['one.jpg']},record,pages,'masks')
        with self.assertRaises(ReviewRequired):
            validate_review({**receipt,'decision':'needs_changes'},record,pages,'masks')

    def test_pipeline_waits_for_explicit_review_then_reuses_seed(self):
        cfg={'name':'review_gate','mask_generation':{'enabled':True}}
        calls=[]
        def prepared(p,out): write_json(out/'inputs.json',{})
        def seeded(p,out):
            calls.append('seed'); (out/'seed_review.jpg').write_bytes(b'fixture')
            write_json(out/'seed_result.json',{'review_pages':{'seed_review.jpg':['a']}})
        with patch('run_pipeline.WORKSPACE',self.root), patch.object(Pipeline,'code_hashes',return_value={}), \
             patch.object(Pipeline,'stage_prepare_images',prepared), patch.object(Pipeline,'stage_seed',seeded):
            pipeline=Pipeline(cfg); pipeline.run('seed_review')
            self.assertEqual(pipeline.state['status'],'awaiting_review')
            receipt={'kind':'seed','artifact_hashes':pipeline.state['stages']['seed']['outputs'],
                'decision':'accept','reviewer':'Codex','reviewer_type':'assistant',
                'inspected_pages':['seed_review.jpg'],'scope':'fixture','notes':'fixture'}
            (pipeline.out/'reviews').mkdir(); write_json(pipeline.out/'reviews/seed.json',receipt)
            resumed=Pipeline(cfg,resume=True); resumed.run('seed_review')
            self.assertEqual(resumed.state['stages']['seed_review']['status'],'completed')
            self.assertEqual(calls,['seed'])
            self.assertTrue((pipeline.out/'seed_review_01/request.json').exists())

    def test_new_masks_cannot_silently_reuse_feature_masked_sfm(self):
        # Keep this contract test independent of historical runs and private PC paths.
        cfg={'schema_version':1,'name':'invalid_feature_masked_reuse',
            'workflow':'cached_dense','feature_masking':True,'threads':1,
            'fusion_min_num_pixels':1,'mesh_depth':8,'evaluation':None,
            'images':'fixture/images','image_list':'fixture/images.txt',
            'masks':'fixture/masks','mask_review':'fixture/review.json',
            'sparse_run':'fixture/sparse','sparse_model':'fixture/sparse/0',
            'dense_workspace':'fixture/dense','dense_provenance':'fixture/dense.json',
            'source_hashes':'fixture/hashes.json',
            'mask_generation':{'device':'cpu','threads':1,'random_seed':0,
                'ordering':'explicit fixture list','target':'synthetic test object'}}
        altered=self.root/'invalid.json'; write_json(altered,cfg)
        with self.assertRaisesRegex(ValueError,'feature-masked SfM'):
            config_from(altered)

    def test_fusion_adapter_consumes_generated_masks_not_stale_masks(self):
        generated=self.root/'propagate'; (generated/'masks').mkdir(parents=True)
        stale=self.root/'stale'; stale.mkdir()
        mask=Image.new('L',(10,8),0)
        mask.paste(255,(0,0,4,8)); mask.save(generated/'masks/a.png.png')
        Image.new('L',(10,8),255).save(stale/'a.png.png')
        dense=self.root/'dense'; (dense/'images').mkdir(parents=True)
        Image.new('RGB',(5,4)).save(dense/'images/a.png')
        imported=self.root/'import'; imported.mkdir()
        write_json(imported/'sizes.json',{'a.png':{'source':[10,8],'dense':[5,4]}})
        cfg={'name':'handoff','mask_generation':{'enabled':True},'masks':str(stale),'dense_workspace':str(dense)}
        with patch('run_pipeline.WORKSPACE',self.root), patch.object(Pipeline,'code_hashes',return_value={}):
            pipeline=Pipeline(cfg)
            pipeline.state['stages']={'propagate':{'directory':str(generated)},'import':{'directory':str(imported)}}
            out=self.root/'derived'; out.mkdir(); pipeline.stage_fusion_masks(out)
            with Image.open(out/'masks/a.png.png') as result:
                self.assertEqual(result.size,(5,4))
                self.assertEqual(result.getpixel((0,0)),255)
                self.assertEqual(result.getpixel((4,0)),0)


if __name__=='__main__': unittest.main()
