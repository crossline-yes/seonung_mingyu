"""Fresh COLMAP stages, with immutable raw-match and reconstruction snapshots."""
from pathlib import Path
from contextlib import closing
import shutil
import sqlite3
import struct

from input_audit import audit_images
from mask_audit import validate_masks
from pipeline_contracts import (read_json, write_json, read_names, fingerprints,
                                verify_fingerprints, model_images, sha256)
from toolchain import COLMAP_EXE

SPARSE_STAGES = ('sparse_inputs','features','matching','geometry','sparse','evaluation','report')


def validate_sparse_config(cfg):
    if not cfg.get('images') or not cfg.get('image_list'):
        raise ValueError('Fresh sparse requires images and image_list')
    spec = cfg['sparse_generation']
    if spec['camera_model'] not in ('PINHOLE','SIMPLE_RADIAL','OPENCV'):
        raise ValueError('Unsupported camera model')
    for key in ('use_gpu','single_camera'):
        if type(spec[key]) is not bool:
            raise ValueError('Expected bool: '+key)
    for key in ('max_image_size','max_features'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError('Expected positive integer: '+key)
    if type(spec['random_seed']) is not int or not 0 < spec['max_ratio'] <= 1:
        raise ValueError('Invalid random seed or match ratio')
    if cfg['feature_masking'] and not cfg.get('mask_generation') and not (cfg.get('masks') and cfg.get('mask_review')):
        raise ValueError('Masked features require masks and an explicit review receipt')
    if cfg.get('mask_generation') and not cfg['feature_masking']:
        raise ValueError('Fresh sparse mask generation requires feature_masking=true')


def copy_database(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Database target already exists')
    for suffix in ('-wal','-journal'):
        sidecar = Path(str(source)+suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError('Database is not a closed snapshot')
    before = sha256(source)
    shutil.copyfile(source,target)
    if before != sha256(target) or before != sha256(source):
        raise ValueError('Database changed during copy')


def database_counts(path, names):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)) as db:
        rows = db.execute('SELECT i.name,k.rows,d.rows FROM images i LEFT JOIN keypoints k USING(image_id) LEFT JOIN descriptors d USING(image_id)').fetchall()
        if set(r[0] for r in rows) != set(names) or len(rows) != len(names):
            raise ValueError('Feature database selection differs from input')
        if any(k is None or d is None or k != d for _,k,d in rows):
            raise ValueError('Missing or inconsistent features')
        return {'images':len(rows),'features':sum(r[1] for r in rows),
                'zero_feature_images':[r[0] for r in rows if r[1] == 0],
                'raw_pair_rows':db.execute('SELECT count(*) FROM matches').fetchone()[0],
                'raw_matches':db.execute('SELECT coalesce(sum(rows),0) FROM matches').fetchone()[0],
                'verified_nonempty_pairs':db.execute('SELECT count(*) FROM two_view_geometries WHERE rows>0').fetchone()[0]}


def clear_geometry_cache(path):
    """Only on a new private copy: raw-only matching writes empty cache rows."""
    with closing(sqlite3.connect(path)) as db, db:
        count = db.execute('SELECT count(*) FROM two_view_geometries').fetchone()[0]
        db.execute('DELETE FROM two_view_geometries')
    return count


class SparseStages:
    def stage_sparse_inputs(self,out):
        c = self.cfg
        names = read_names(c['image_list'],c['images'])
        audit = audit_images(c['images'],names)
        write_json(out/'input_audit.json',audit)
        if audit['unreadable_images'] or any(r.get('exif_orientation') != 1 for r in audit['images']):
            raise ValueError('Unreadable images or unsupported EXIF orientation')
        if c['sparse_generation']['single_camera'] and len(audit['dimensions']) != 1:
            raise ValueError('Shared camera requires identical image dimensions')
        inputs = [Path(c['image_list'])] + [Path(c['images'])/n for n in names]
        if c['feature_masking']:
            masks = validate_masks(c['images'],names,self.mask_root)
            write_json(out/'mask_audit.json',masks)
            receipt = read_json(self.mask_review_path)
            if c.get('mask_generation'):
                if receipt.get('decision') != 'accept':
                    raise ValueError('Generated masks have not been accepted')
            elif (receipt.get('decision') != 'accept' or
                  receipt.get('mask_hashes') != {r['image']:r['sha256'] for r in masks['records']} or
                  not receipt.get('reviewer') or not receipt.get('scope')):
                raise ValueError('Imported mask review must accept these exact mask hashes and declare reviewer/scope')
            inputs += [self.mask_review_path]+[self.mask_root/(n+'.png') for n in names]
        if c['evaluation']:
            ev = c['evaluation']
            inputs += [Path(ev['script']),Path(ev['protocol']),Path(ev['gt'])/'cameras.txt',Path(ev['gt'])/'images.txt']
            protocol = read_json(ev['protocol'])
            pairs = protocol['pairs']
            if not pairs or any(len(p) != 2 or p[0] == p[1] or not set(p).issubset(names) for p in pairs):
                raise ValueError('Evaluation protocol contains invalid/out-of-selection pairs')
        frozen = fingerprints(inputs)
        for record in audit['images']:
            if frozen[str((Path(c['images'])/record['name']).resolve())] != record['sha256']:
                raise ValueError('Input changed during audit')
        write_json(out/'inputs.json',frozen)
        (out/'images.txt').write_text('\n'.join(names)+'\n',encoding='utf-8')

    def sparse_names(self):
        return read_names(self.artifact('sparse_inputs','images.txt'),self.cfg['images'])

    def stage_features(self,out):
        verify_fingerprints(read_json(self.artifact('sparse_inputs','inputs.json')))
        c,s = self.cfg,self.cfg['sparse_generation']
        cmd = [COLMAP_EXE,'feature_extractor','--database_path',out/'database.db',
            '--image_path',c['images'],'--image_list_path',self.artifact('sparse_inputs','images.txt'),
            '--ImageReader.camera_model',s['camera_model'],'--ImageReader.single_camera',int(s['single_camera']),
            '--FeatureExtraction.use_gpu',int(s['use_gpu']),'--FeatureExtraction.num_threads',c['threads'],
            '--FeatureExtraction.max_image_size',s['max_image_size'],'--SiftExtraction.max_num_features',s['max_features'],
            '--default_random_seed',s['random_seed']]
        if c['feature_masking']:
            cmd += ['--ImageReader.mask_path',self.mask_root]
        self.command(cmd,out)
        write_json(out/'diagnostics.json',database_counts(out/'database.db',self.sparse_names()))

    def match_command(self,out,raw_only):
        s = self.cfg['sparse_generation']
        self.command([COLMAP_EXE,'exhaustive_matcher','--database_path',out/'database.db',
            '--FeatureMatching.use_gpu',int(s['use_gpu']),'--FeatureMatching.num_threads',self.cfg['threads'],
            '--FeatureMatching.skip_geometric_verification',int(raw_only),'--FeatureMatching.guided_matching',0,
            '--SiftMatching.max_ratio',s['max_ratio'],'--SiftMatching.max_distance',0.7,'--SiftMatching.cross_check',1,
            '--TwoViewGeometry.min_num_inliers',0 if raw_only else 15,'--default_random_seed',s['random_seed']],out)

    def stage_matching(self,out):
        copy_database(self.artifact('features','database.db'),out/'database.db')
        self.match_command(out,True)
        counts = database_counts(out/'database.db',self.sparse_names())
        if counts['raw_pair_rows'] != counts['images']*(counts['images']-1)//2:
            raise ValueError('Exhaustive raw matching did not record every selected pair')
        write_json(out/'diagnostics.json',counts)

    def stage_geometry(self,out):
        copy_database(self.artifact('matching','database.db'),out/'database.db')
        cleared = clear_geometry_cache(out/'database.db')
        write_json(out/'cache_reset.json',{'removed_geometry_rows_in_private_copy':cleared,
            'reason':'Raw-only COLMAP matching writes empty geometry cache entries; force real verification',
            'original_raw_database_modified':False})
        self.match_command(out,False)
        counts = database_counts(out/'database.db',self.sparse_names())
        write_json(out/'diagnostics.json',counts)
        if not counts['verified_nonempty_pairs']:
            raise ValueError('No geometrically verified image pairs for reconstruction')

    def stage_sparse(self,out):
        # mapper may update a database in future versions; never share prior snapshots.
        copy_database(self.artifact('geometry','database.db'),out/'database.db')
        model_root = out/'models'; model_root.mkdir()
        self.command([COLMAP_EXE,'mapper','--database_path',out/'database.db','--image_path',self.cfg['images'],
            '--output_path',model_root,'--Mapper.num_threads',self.cfg['threads'],
            '--Mapper.random_seed',self.cfg['sparse_generation']['random_seed']],out)
        names = self.sparse_names()
        models = []
        for model in sorted(model_root.iterdir()):
            if not model.is_dir() or not (model/'images.bin').is_file():
                continue
            registered = sorted(model_images(model/'images.bin'))
            if not set(registered).issubset(names):
                raise ValueError('Model includes unselected images')
            with (model/'points3D.bin').open('rb') as stream:
                points, = struct.unpack('<Q',stream.read(8))
            models.append({'path':str(model),'registered_count':len(registered),'points3D':points,
                'registered_images':registered,'unregistered_images':sorted(set(names)-set(registered))})
        if not models:
            raise ValueError('Mapper did not produce a sparse model')
        best = max(models,key=lambda m:(m['registered_count'],m['points3D']))
        write_json(out/'diagnostics.json',{'input_count':len(names),'models':models,
            'selected_model':best['path'],'selection_rule':'most registered images, then most points; no GT used',
            'selected_registered_count':best['registered_count'],'selected_unregistered_images':best['unregistered_images'],
            'quality':'registration diagnostics, not independent geometry accuracy'})

    def sparse_report(self,out):
        verify_fingerprints(read_json(self.artifact('sparse_inputs','inputs.json')))
        # Detect changes to intermediate snapshots even in an uninterrupted run.
        for stage in SPARSE_STAGES[:-1]:
            verify_fingerprints(self.state['stages'][stage]['outputs'])
        diagnostics = read_json(self.artifact('sparse','diagnostics.json'))
        result = {'scene':self.cfg['scene'],'workflow':'fresh_sparse','source_images_unchanged':True,
            'feature_masking':self.cfg['feature_masking'],'freshly_computed':['features','raw_matches','two_view_geometry','sparse'],
            'reused_reconstruction':False,'dense_and_mesh':'not_executed','sparse':diagnostics,
            'evaluation':read_json(self.artifact('evaluation','result.json')),'official_pass':None,
            'feature_database':str(self.artifact('features','database.db')),
            'raw_match_database':str(self.artifact('matching','database.db')),
            'human_work_seconds':None,
            'current_stage_seconds':{k:v['seconds'] for k,v in self.state['stages'].items() if 'seconds' in v}}
        write_json(out/'summary.json',result)
        (out/'summary.md').write_text('\n'.join(['# 건축물 사진의 새 특징점·희소 복원 실행','',
            f"대상: {result['scene']}. 입력 {diagnostics['input_count']}장, 선택 모델 등록 {diagnostics['selected_registered_count']}장.",
            f"모델 수: {len(diagnostics['models'])}. 특징점 마스킹: {result['feature_masking']}.",
            '특징점·원시 매칭·기하 검증·카메라 복원을 새로 계산했다. 기존 복원 결과와 기준 포즈를 복원 입력으로 사용하지 않았다.',
            '원시 매칭 DB를 별도 보존하고 복원용 기하 검증 DB와 구분했다.',
            f"평가 상태: {result['evaluation']['status']}. 잠정 규약이며 공식 목표 달성 판정은 하지 않는다.",
            '밀집복원·메시 생성과 건축물 구조 보존 검증은 이 실행 범위에 포함되지 않는다.','']),encoding='utf-8')
