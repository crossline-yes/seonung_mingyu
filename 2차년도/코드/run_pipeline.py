"""Run fresh sparse reconstruction or connect reviewed masks to cached dense data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

# Shared, established utilities and datasets remain in the parent project.
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKSPACE.parent / 'scripts'))

from PIL import Image
from mask_audit import validate_masks
from pipeline_contracts import (read_json, write_json, sha256, fingerprints,
    verify_fingerprints, read_names, validate_scaled_models, ply_counts)
from toolchain import PROJECT, COLMAP_EXE, COLMAP_VERSION, colmap_environment
from mask_review import ReviewRequired, require_review
from sparse_pipeline import SparseStages, SPARSE_STAGES, validate_sparse_config

STAGES = ('import', 'fusion_masks', 'fusion', 'mesh', 'evaluation', 'report')
MASK_STAGES = ('prepare_images','seed','seed_review','propagate','mask_review','mask_export')
PATH_KEYS = ('images', 'image_list', 'masks', 'mask_review', 'sparse_run',
             'sparse_model', 'dense_workspace', 'dense_provenance', 'source_hashes')


def utc():
    return datetime.now(timezone.utc).isoformat()


def config_from(path):
    cfg = read_json(path)
    if cfg.get('schema_version') != 1 or not re.fullmatch(r'[A-Za-z0-9_-]+', cfg.get('name', '')):
        raise ValueError('Invalid schema or run name')
    fresh = cfg.get('workflow') == 'fresh_sparse'
    if cfg.get('workflow', 'cached_dense') not in ('cached_dense', 'fresh_sparse'):
        raise ValueError('Unsupported workflow')
    for key in PATH_KEYS:
        if cfg.get('mask_generation') and key in ('masks','mask_review'):
            cfg[key] = None
            continue
        if fresh and not cfg.get(key):
            cfg[key] = None
            continue
        p = Path(cfg[key])
        cfg[key] = str((PROJECT / p).resolve())
    for key in (('threads',) if fresh else ('threads', 'fusion_min_num_pixels', 'mesh_depth')):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Expected positive integer: {key}')
    if (not fresh and not 5 <= cfg['mesh_depth'] <= 13) or type(cfg['feature_masking']) is not bool:
        raise ValueError('Invalid mesh depth or feature masking setting')
    if cfg['evaluation'] is not None:
        for key in ('script', 'gt', 'protocol'):
            cfg['evaluation'][key] = str((PROJECT / cfg['evaluation'][key]).resolve())
    if cfg.get('mask_generation'):
        spec=cfg['mask_generation']
        for key in ('python','source_json','prompt_provenance'):
            if spec.get(key): spec[key]=str((PROJECT/spec[key]).resolve())
        if spec['device'] != 'cpu' or type(spec['threads']) is not int or spec['threads'] < 1:
            raise ValueError('This adapter uses the pinned CPU protocol with positive thread count')
        if type(spec['random_seed']) is not int or not spec.get('ordering') or not spec.get('target'):
            raise ValueError('Explicit random seed, image ordering and target description required')
        if cfg['feature_masking'] and not fresh:
            raise ValueError('New masks with imported feature-masked SfM are unsupported; connect fresh features first')
    if fresh:
        validate_sparse_config(cfg)
    return cfg


class Pipeline(SparseStages):
    def __init__(self, cfg, resume=False):
        self.cfg = cfg
        self.out = WORKSPACE / '실행결과' / cfg['name']
        if self.out.exists():
            if not resume:
                raise ValueError('Run exists; use --resume or choose a new name')
            if read_json(self.out / 'config.json') != cfg:
                raise ValueError('Configuration changed; choose a new run name')
            if read_json(self.out / 'code_sha256.json') != self.code_hashes():
                raise ValueError('Runner changed; choose a new run name')
            self.state = read_json(self.out / 'run.json')
        else:
            if resume:
                raise ValueError('Cannot resume a missing run')
            self.out.mkdir(parents=True)
            write_json(self.out / 'config.json', cfg)
            write_json(self.out / 'code_sha256.json', self.code_hashes())
            self.state = {'started': utc(), 'status': 'created', 'stages': {}}
            self.save()

    def code_hashes(self):
        files=[Path(__file__), Path(__file__).with_name('pipeline_contracts.py'),
               Path(__file__).with_name('sparse_pipeline.py'),
               Path(__file__).with_name('mask_review.py'), PROJECT/'scripts/mask_audit.py', PROJECT/'scripts/toolchain.py']
        if self.cfg.get('mask_generation'):
            files += [Path(__file__).with_name('sam_masking.py')]
        if self.cfg.get('mask_generation') or self.cfg.get('workflow') == 'fresh_sparse':
            files += [PROJECT/'scripts/input_audit.py']
        return fingerprints(files)

    @property
    def stages(self):
        main = SPARSE_STAGES if self.cfg.get('workflow') == 'fresh_sparse' else STAGES
        return MASK_STAGES + main if self.cfg.get('mask_generation') else main

    @property
    def mask_root(self):
        return self.artifact('propagate','masks') if self.cfg.get('mask_generation') else Path(self.cfg['masks'])

    @property
    def mask_review_path(self):
        return self.artifact('mask_review','review.json') if self.cfg.get('mask_generation') else Path(self.cfg['mask_review'])

    def save(self):
        write_json(self.out / 'run.json', self.state)

    def command(self, args, directory):
        cmd = [str(v) for v in args]
        write_json(directory / 'command.json', cmd)
        with (directory / 'command.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=colmap_environment())
        write_json(directory / 'exit.json', {'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError(f'Command failed ({result.returncode}); see {directory / "command.log"}')

    def artifact(self, stage, name):
        return Path(self.state['stages'][stage]['directory']) / name

    def run(self, stop_after):
        # Exclusive lock is automatically released after normal errors. A hard-kill
        # leaves evidence: inspect the PID before manually removing a stale lock.
        lock = self.out / 'active.lock'
        with lock.open('x') as stream:
            import os
            stream.write(str(os.getpid()))
        try:
            if stop_after not in self.stages:
                raise ValueError('Requested stage is not enabled by this configuration')
            for stage in self.stages:
                previous = self.state['stages'].get(stage, {})
                if previous.get('status') == 'completed':
                    verify_fingerprints(previous['outputs'])
                    if stage in ('import','prepare_images','sparse_inputs'):
                        verify_fingerprints(read_json(self.artifact(stage, 'inputs.json')))
                    print(f'Reuse completed stage: {stage}', flush=True)
                else:
                    attempt = previous.get('attempt', 0) + 1
                    directory = self.out / f'{stage}_{attempt:02d}'
                    directory.mkdir()
                    record = {'status': 'running', 'attempt': attempt, 'directory': str(directory), 'started': utc()}
                    self.state['stages'][stage] = record
                    self.state.update(status='running', active_stage=stage)
                    self.save()
                    started = time.monotonic()
                    print(f'Running stage: {stage}', flush=True)
                    try:
                        getattr(self, 'stage_' + stage)(directory)
                        record.update(status='completed', seconds=time.monotonic()-started,
                                      outputs=fingerprints(p for p in directory.rglob('*') if p.is_file()))
                    except ReviewRequired as exc:
                        record.update(status='awaiting_review',seconds=time.monotonic()-started,
                                      reason=str(exc),outputs=fingerprints(p for p in directory.rglob('*') if p.is_file()))
                        self.state.update(status='awaiting_review',active_stage=stage)
                        self.save()
                        print(str(exc),flush=True)
                        return
                    except Exception as exc:
                        record.update(status='failed', seconds=time.monotonic()-started, error=str(exc))
                        self.state.update(status='failed', active_stage=None)
                        self.save()
                        raise
                    self.save()
                if stage == stop_after:
                    break
            self.state.update(status='completed_pending_quality_review' if stop_after == 'report' else 'stopped_after_' + stop_after,
                              active_stage=None, finished=utc())
            self.save()
        finally:
            lock.unlink()

    def sam_command(self, action, out, *extra):
        self.command([self.cfg['mask_generation']['python'],'-X','utf8',Path(__file__).with_name('sam_masking.py'),
            action,'--config',self.out/'config.json','--out',out,*extra],out)

    def stage_prepare_images(self,out):
        self.sam_command('prepare',out)

    def stage_seed(self,out):
        self.sam_command('seed',out,'--prepared',self.artifact('prepare_images',''))

    def stage_seed_review(self,out):
        require_review(self.out,'seed',out)

    def stage_propagate(self,out):
        self.sam_command('propagate',out,'--prepared',self.artifact('prepare_images',''),
                         '--seed',self.artifact('seed',''))

    def stage_mask_review(self,out):
        require_review(self.out,'masks',out)

    def stage_mask_export(self,out):
        verify_fingerprints(read_json(self.artifact('prepare_images','inputs.json')))
        names=read_names(self.cfg['image_list'],self.cfg['images'])
        audit=validate_masks(self.cfg['images'],names,self.mask_root)
        write_json(out/'mask_audit.json',audit)
        review=read_json(self.mask_review_path)
        write_json(out/'mask_bundle.json',{'schema_version':1,'source':'new_sam_inference',
            'image_root':self.cfg['images'],'image_names':names,'mask_root':str(self.mask_root),
            'coordinate_space':'original_image_pixels','mask_filename':'<image_name>.png',
            'values':{'keep':255,'exclude':0},'mask_hashes':{r['image']:r['sha256'] for r in audit['records']},
            'input_manifest':str(self.artifact('prepare_images','manifest.json')),
            'review':{'path':str(self.mask_review_path),'sha256':sha256(self.mask_review_path),
                      'reviewer_type':review['reviewer_type'],'scope':review['scope'],'notes':review['notes']},
            'quality':'reviewed_for_declared_scope_not_pixel_accuracy','human_work_seconds':None})

    def stage_import(self, out):
        c = self.cfg
        names = read_names(c['image_list'], c['images'])
        sparse_run, model, dense = map(Path, (c['sparse_run'], c['sparse_model'], c['dense_workspace']))
        original = read_json(sparse_run / 'run.json')
        if not original['status'].startswith('reconstructed'):
            raise ValueError('Imported sparse reconstruction did not complete')
        if model not in [Path(p) for p in original['models']]:
            raise ValueError('Selected sparse model does not belong to the imported run')
        if Path(original['images']).resolve() != Path(c['images']):
            raise ValueError('Sparse run uses a different image root')
        if names != read_names(sparse_run/'images.txt', c['images']):
            raise ValueError('Sparse input selection differs')
        expected_mode = 'feature_extraction' if c['feature_masking'] else 'none'
        if original['masking'] != expected_mode:
            raise ValueError('Declared feature-mask application differs from imported run')
        if c['feature_masking'] and Path(original['mask_path']).resolve() != self.mask_root:
            raise ValueError('Sparse feature masks differ')
        database = sparse_run / 'database.db'
        for suffix in ('-wal', '-journal'):
            extra = Path(str(database) + suffix)
            if extra.exists() and extra.stat().st_size:
                raise ValueError('Imported database is not a closed snapshot')
        audit = read_json(sparse_run/'input_audit.json')
        source_records = {i['name']: i for i in audit['images']}
        if set(source_records) != set(names) or audit['unreadable_images']:
            raise ValueError('Input audit is incomplete')
        source_hashes = {str(Path(c['images'])/n): source_records[n]['sha256'] for n in names}
        verify_fingerprints(source_hashes)
        mask_audit = validate_masks(c['images'], names, self.mask_root)
        frozen = read_json(c['source_hashes'])
        required_frozen = [model / f for f in ('cameras.bin', 'images.bin', 'points3D.bin')]
        if not c.get('mask_generation'):
            required_frozen += [self.mask_root/(n+'.png') for n in names]
        frozen = {str(Path(p).resolve()): h for p, h in frozen.items()}
        for path in required_frozen:
            if str(path.resolve()) not in frozen:
                raise ValueError(f'Missing original dense input hash: {path}')
            verify_fingerprints({str(path): frozen[str(path.resolve())]})
        provenance = read_json(c['dense_provenance'])
        def matching_steps(command):
            return [s for s in provenance['steps'] if s.get('stage') == command and s.get('exit_code') == 0]
        def option(step, key):
            cmd = step['command']
            return Path(cmd[cmd.index(key)+1]).resolve() if key in cmd else None
        undistort = [s for s in matching_steps('image_undistorter') if option(s,'--output_path') == dense]
        if len(undistort) != 1 or option(undistort[0],'--input_path') != model or option(undistort[0],'--image_path') != Path(c['images']):
            raise ValueError('Dense workspace provenance does not match selected images/model')
        depth_steps = [s for s in matching_steps('patch_match_stereo') if option(s,'--workspace_path') == dense]
        if len(depth_steps) != 1:
            raise ValueError('Missing successful depth-map generation record')
        sizes = validate_scaled_models(model, dense/'sparse', names)
        for name in names:
            with Image.open(Path(c['images'])/name) as source, Image.open(dense/'images'/name) as derived:
                if source.size != tuple(sizes[name]['source']) or derived.size != tuple(sizes[name]['dense']):
                    raise ValueError(f'Image/camera size mismatch: {name}')
            for kind in ('depth_maps', 'normal_maps'):
                if not (dense/'stereo'/kind/(name+'.geometric.bin')).is_file():
                    raise ValueError(f'Missing cached {kind}: {name}')
        inputs = [Path(c[k]) for k in ('image_list','dense_provenance','source_hashes')]
        inputs += [self.mask_review_path] + [self.mask_root/(n+'.png') for n in names]
        inputs += [sparse_run/f for f in ('run.json','input_audit.json','images.txt','database.db')]
        inputs += required_frozen
        inputs += [p for p in dense.rglob('*') if p.is_file() and p.parts[-1] not in ('fused.ply','fused.ply.vis','fused_masked.ply','fused_masked.ply.vis')]
        if c['evaluation']:
            ev = c['evaluation']
            inputs += [Path(ev['script']), Path(ev['protocol']), Path(ev['gt'])/'cameras.txt', Path(ev['gt'])/'images.txt']
        record = fingerprints(inputs)
        record.update(source_hashes)
        write_json(out/'inputs.json', record)
        write_json(out/'mask_audit.json', mask_audit)
        write_json(out/'sizes.json', sizes)
        write_json(out/'reuse.json', {'images':len(names), 'mask_generation':'new_inference' if c.get('mask_generation') else 'reused', 'features_and_sfm':'reused',
            'undistortion_and_depth_maps':'reused', 'feature_masking':c['feature_masking'],
            'historical_compute_seconds_not_in_current_timing':sum(s['seconds'] for s in original['steps'] + undistort + depth_steps),
            'historical_timing_scope':'sparse run, undistortion and depth maps; masking cost excluded',
            'manual_intervention_seconds':None, 'quality_status':'not_measured_by_import_checks'})

    def stage_fusion_masks(self, out):
        sizes = read_json(self.artifact('import', 'sizes.json'))
        masks = out/'masks'; masks.mkdir()
        for name, size in sizes.items():
            target = masks/(name+'.png'); target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(self.mask_root/(name+'.png')) as im:
                im.resize(tuple(size['dense']), Image.Resampling.NEAREST).save(target)
        write_json(out/'audit.json', validate_masks(Path(self.cfg['dense_workspace'])/'images', list(sizes), masks))

    def stage_fusion(self, out):
        self.command([COLMAP_EXE, 'stereo_fusion', '--workspace_path', self.cfg['dense_workspace'],
            '--input_type','geometric','--output_path',out/'fused.ply',
            '--StereoFusion.mask_path',self.artifact('fusion_masks','masks'),
            '--StereoFusion.num_threads',self.cfg['threads'],'--StereoFusion.cache_size',4,
            '--StereoFusion.min_num_pixels',self.cfg['fusion_min_num_pixels']], out)
        counts = ply_counts(out/'fused.ply')
        if counts.get('vertex', 0) == 0:
            raise ValueError('Fusion produced no points')
        write_json(out/'diagnostics.json', counts)

    def stage_mesh(self, out):
        self.command([COLMAP_EXE,'poisson_mesher','--input_path',self.artifact('fusion','fused.ply'),
            '--output_path',out/'mesh.ply','--PoissonMeshing.depth',self.cfg['mesh_depth'],
            '--PoissonMeshing.trim',self.cfg['mesh_trim'],'--PoissonMeshing.num_threads',self.cfg['threads']],out)
        counts = ply_counts(out/'mesh.ply')
        if min(counts.get('vertex',0), counts.get('face',0)) <= 0:
            raise ValueError('Mesher produced an empty mesh')
        write_json(out/'diagnostics.json', {'header_counts':counts, 'geometry_quality':'pending_review',
            'note':'A nonempty file does not establish completeness, accuracy or preservation of thin structures'})

    def stage_evaluation(self, out):
        evaluation = self.cfg['evaluation']
        if evaluation is None:
            write_json(out/'result.json', {'status':'not_configured', 'score':None, 'official_pass':None,
                'reason':'Independent feature metric, GT and fixed pair protocol still require selection'})
            return
        database = self.artifact('matching','database.db') if self.cfg.get('workflow') == 'fresh_sparse' else Path(self.cfg['sparse_run'])/'database.db'
        self.command([sys.executable,evaluation['script'],'--database',database,
            '--gt',evaluation['gt'],'--protocol',evaluation['protocol'],'--output',out/'result.json',
            '--epsilon-px',evaluation.get('epsilon_px',5),
            '--gt-name-prefix',evaluation.get('gt_name_prefix','')],out)
        result = read_json(out/'result.json')
        if result.get('status') != 'evaluated' or result.get('approval_status') != 'provisional_not_official':
            raise ValueError('Unexpected provisional evaluator result')

    def stage_report(self, out):
        if self.cfg.get('workflow') == 'fresh_sparse':
            return self.sparse_report(out)
        verify_fingerprints(read_json(self.artifact('import','inputs.json')))
        reuse = read_json(self.artifact('import','reuse.json'))
        mesh = read_json(self.artifact('mesh','diagnostics.json'))
        evaluation = read_json(self.artifact('evaluation','result.json'))
        report = {'scene':self.cfg['scene'], 'source_images_unchanged':True,
            'mask_method':self.cfg['mask_method'], 'feature_masking':self.cfg['feature_masking'],
            'fusion_masking':True, 'reused':reuse, 'mesh':mesh,
            'mesh_path':str(self.artifact('mesh','mesh.ply')),
            'evaluation':evaluation, 'official_pass':None,
            'current_stage_seconds':{k:v['seconds'] for k,v in self.state['stages'].items() if 'seconds' in v}}
        write_json(out/'summary.json',report)
        feature = '적용' if self.cfg['feature_masking'] else '미적용'
        lines = ['# 모듈 연결 실행 결과', '', f"대상은 {self.cfg['scene']}이며 입력은 {reuse['images']}장이다.", '',
            ('SAM 마스크를 새로 계산하고 명시된 범위에서 검토했다. 특징점·카메라·깊이맵은 기존 결과를 재사용했다.' if self.cfg.get('mask_generation') else
             '기존 마스크·특징점·카메라·깊이맵을 재사용했다.') + ' 융합용 마스크 변환, 대상 점군 융합, 메시 생성을 연결했다.', '',
            f'특징점 마스크는 {feature}, 점군 융합 마스크는 적용했다. 이 경로의 결과로 특징점 마스킹에 따른 개선을 주장하지 않는다.', '',
            f"메시: {report['mesh_path']}", f"정점 {mesh['header_counts']['vertex']}개, 면 {mesh['header_counts']['face']}개. 파일 생성 확인이며 형상 품질 판정은 별도다.", '',
            f"독립 특징점 평가 상태: {evaluation['status']}. 사업계획서 목표 달성 판정은 보류한다.", '',
            '이번 단계별 시간은 summary.json에 기록했다. 재사용한 과거 연산 비용과 사람 작업시간을 0으로 해석하지 않는다.', '',
            '대형 건축물 적용, 임의 사진 순서의 마스크 자동 생성, 렌즈 왜곡이 있는 입력의 마스크 변환은 이 실행으로 검증하지 않았다.']
        (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--plan',action='store_true',help='Print stage modes without executing or creating a run')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',choices=tuple(dict.fromkeys(MASK_STAGES+STAGES+SPARSE_STAGES)),default='report')
    args = parser.parse_args()
    cfg = config_from(args.config)
    if args.plan:
        generated=bool(cfg.get('mask_generation'))
        if cfg.get('workflow') == 'fresh_sparse':
            print(json.dumps({'name':cfg['name'],'stages':(MASK_STAGES if generated else ())+SPARSE_STAGES,
                'feature_masking':cfg['feature_masking'],'reused_reconstruction':False,
                'dense_and_mesh':'not_in_this_workflow','evaluation':'provisional' if cfg['evaluation'] else 'pending'},ensure_ascii=False,indent=2))
            return
        print(json.dumps({'name':cfg['name'],'stages':MASK_STAGES+STAGES if generated else STAGES,
            'reused':(['saved prompt information'] if generated else ['mask generation']) + ['features/SfM','depth maps'],
            'executed':(['frame preparation','SAM seed','SAM propagation','explicit review gates','mask bundle'] if generated else []) + ['artifact validation','fusion mask scaling','fusion','mesh','report'],
            'evaluation':'provisional' if cfg['evaluation'] else 'pending'},ensure_ascii=False,indent=2))
        return
    version = subprocess.run([str(COLMAP_EXE),'version'],capture_output=True,text=True,
        encoding='utf-8',errors='replace',check=True,env=colmap_environment())
    if f'COLMAP {COLMAP_VERSION}' not in version.stdout+version.stderr:
        raise ValueError('Unexpected COLMAP version')
    Pipeline(cfg,args.resume).run(args.stop_after)


if __name__ == '__main__':
    main()
