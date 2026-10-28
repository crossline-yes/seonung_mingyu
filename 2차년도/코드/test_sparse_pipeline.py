"""Check snapshot isolation, mask gates and upstream mutation on resume."""
from pathlib import Path
from contextlib import closing
import tempfile
import sqlite3
import unittest
from unittest.mock import patch
from pipeline_contracts import write_json, fingerprints
from sparse_pipeline import copy_database, validate_sparse_config, clear_geometry_cache
from run_pipeline import Pipeline


class FreshSparseTest(unittest.TestCase):
    def test_geometry_reset_preserves_raw_matches_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = root/'raw.db'; dst = root/'geometry.db'
            with closing(sqlite3.connect(src)) as db, db:
                db.execute('CREATE TABLE matches(pair_id INTEGER, rows INTEGER)')
                db.execute('CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER)')
                db.execute('INSERT INTO matches VALUES (1,9)')
                db.execute('INSERT INTO two_view_geometries VALUES (1,0)')
            copy_database(src,dst)
            self.assertEqual(clear_geometry_cache(dst),1)
            with closing(sqlite3.connect(dst)) as db:
                self.assertEqual(db.execute('SELECT * FROM matches').fetchall(),[(1,9)])
                self.assertEqual(db.execute('SELECT * FROM two_view_geometries').fetchall(),[])
            with closing(sqlite3.connect(src)) as db:
                self.assertEqual(db.execute('SELECT * FROM two_view_geometries').fetchall(),[(1,0)])

    def test_database_copy_isolated_and_active_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = root/'source.db'; dst = root/'copy.db'
            src.write_bytes(b'closed database')
            copy_database(src,dst)
            dst.write_bytes(b'changed')
            self.assertEqual(src.read_bytes(),b'closed database')
            (root/'source.db-wal').write_bytes(b'active')
            with self.assertRaisesRegex(ValueError,'closed snapshot'):
                copy_database(src,root/'next.db')

    def test_masked_config_cannot_silently_fall_back(self):
        cfg = {'images':'images','image_list':'list','feature_masking':True,
            'sparse_generation':{'camera_model':'PINHOLE','use_gpu':False,'single_camera':False,
                'max_image_size':1024,'max_features':8192,'random_seed':0,'max_ratio':0.8}}
        with self.assertRaisesRegex(ValueError,'require masks'):
            validate_sparse_config(cfg)
        validate_sparse_config({**cfg,'feature_masking':False})

    def test_sparse_resume_rejects_source_change_before_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'image'; source.write_bytes(b'original')
            def audited(p,out):
                write_json(out/'inputs.json',fingerprints([source]))
            cfg = {'name':'fresh','workflow':'fresh_sparse'}
            with patch('run_pipeline.WORKSPACE',root), patch.object(Pipeline,'code_hashes',return_value={}), patch.object(Pipeline,'stage_sparse_inputs',audited):
                Pipeline(cfg).run('sparse_inputs')
                source.write_bytes(b'changed')
                with patch.object(Pipeline,'stage_features') as next_stage:
                    with self.assertRaisesRegex(ValueError,'changed'):
                        Pipeline(cfg,resume=True).run('features')
                    next_stage.assert_not_called()

if __name__ == '__main__':
    unittest.main()
