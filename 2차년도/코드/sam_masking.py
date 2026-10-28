"""Configured SAM2 seed/propagation adapter; no existing mask is an inference input."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts'))
from input_audit import audit_images
from mask_audit import validate_masks
from pipeline_contracts import read_json, write_json, read_names, fingerprints, verify_fingerprints, sha256


def validate_prompts(spec, size, names):
    w, h = size
    if spec['seed_image'] != names[0]:
        raise ValueError('Forward propagation requires the declared seed to be first in the explicit image list')
    points = np.asarray(spec['points_xy'],dtype=float)
    labels = np.asarray(spec['labels'])
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or labels.shape != (len(points),):
        raise ValueError('Point/label dimensions mismatch')
    if not np.isfinite(points).all() or (points < 0).any() or (points >= [w,h]).any():
        raise ValueError('Prompt points outside original image coordinates')
    if not np.isin(labels,[0,1]).all() or not (labels == 1).any():
        raise ValueError('Binary labels and at least one foreground point required')
    for key in ('box_xyxy','review_crop_xyxy'):
        if spec.get(key) is None and key == 'review_crop_xyxy':
            continue
        box = np.asarray(spec[key],dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(f'Invalid {key}')
        x0,y0,x1,y1 = box
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f'Invalid {key} bounds')


def prepare(cfg, out):
    spec=cfg['mask_generation']; root=Path(cfg['images'])
    names=read_names(cfg['image_list'],root)
    if len({str((root/n).resolve()).casefold() for n in names}) != len(names):
        raise ValueError('Image list aliases the same source path')
    audit=audit_images(root,names)
    write_json(out/'input_audit.json',audit)
    if audit['unreadable_images'] or len(audit['dimensions']) != 1:
        raise ValueError('SAM video adapter requires readable, equally sized images')
    if any(i['exif_orientation'] != 1 for i in audit['images']):
        raise ValueError('Normalize EXIF orientation with recorded mask coordinates before SAM')
    size=(audit['images'][0]['width'],audit['images'][0]['height'])
    validate_prompts(spec,size,names)
    source=read_json(spec['source_json']); repo=Path(source['repository_path'])
    checkpoint=Path(source['checkpoint']['path'])
    if sha256(checkpoint) != source['checkpoint']['sha256']:
        raise ValueError('Checkpoint does not match pinned source descriptor')
    files=[Path(cfg['image_list']),Path(spec['source_json']),checkpoint]
    files += list((repo/'sam2').rglob('*.py')) + list((repo/'sam2').rglob('*.yaml'))
    if spec.get('prompt_provenance'): files.append(Path(spec['prompt_provenance']))
    source_hashes=fingerprints(files)
    source_hashes.update({str(root/item['name']):item['sha256'] for item in audit['images']})
    write_json(out/'inputs.json',source_hashes)
    frames=out/'frames'; frames.mkdir()
    records=[]
    for index,name in enumerate(names):
        dest=frames/f'{index:05d}.jpg'
        with Image.open(root/name) as im:
            im.convert('RGB').save(dest,quality=100,subsampling=0)
        records.append({'frame':index,'image':name,'jpeg':dest.name,'jpeg_sha256':sha256(dest)})
        print(f'Prepared {index+1}/{len(names)}',flush=True)
    verify_fingerprints(source_hashes)
    write_json(out/'manifest.json',{'image_root':str(root),'image_size':size,'records':records,
        'ordering':spec['ordering'],'target':spec['target'], 'source':source,
        'conversion':'RGB JPEG quality=100 subsampling=0 at original dimensions; no EXIF transform',
        'prompts':spec, 'ground_truth_masks_used':False})


def runtime(cfg, prepared):
    manifest=read_json(prepared/'manifest.json')
    verify_fingerprints(read_json(prepared/'inputs.json'))
    sys.path.insert(0,manifest['source']['repository_path'])
    import torch
    spec=cfg['mask_generation']
    torch.set_num_threads(spec['threads']); torch.manual_seed(spec['random_seed'])
    return torch,manifest


def seed(cfg, prepared, out):
    torch,manifest=runtime(cfg,prepared)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    spec=cfg['mask_generation']; size=tuple(manifest['image_size'])
    predictor=SAM2ImagePredictor(build_sam2(spec['model_config'],manifest['source']['checkpoint']['path'],
        device=spec['device'],apply_postprocessing=False))
    with Image.open(prepared/'frames'/manifest['records'][0]['jpeg']) as im:
        rgb=np.asarray(im.convert('RGB'))
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks,scores,_=predictor.predict(point_coords=np.asarray(spec['points_xy']),
            point_labels=np.asarray(spec['labels']),box=np.asarray(spec['box_xyxy']),multimask_output=True)
    if not np.isfinite(scores).all() or masks.shape[1:] != size[::-1]:
        raise ValueError('Invalid SAM seed prediction')
    chosen=int(np.argmax(scores)); candidates=out/'candidates'; candidates.mkdir()
    page=Image.new('RGB',(480*(len(masks)+1),420),'white'); draw=ImageDraw.Draw(page)
    prompt=Image.fromarray(rgb); marks=ImageDraw.Draw(prompt)
    marks.rectangle(spec['box_xyxy'],outline='blue',width=max(2,size[0]//550))
    radius=max(3,size[0]//240)
    for (x,y),label in zip(spec['points_xy'],spec['labels']):
        marks.ellipse((x-radius,y-radius,x+radius,y+radius),fill='lime' if label else 'red')
    crop=spec.get('review_crop_xyxy') or [0,0,*size]
    prompt=prompt.crop(crop); prompt.thumbnail((480,370)); page.paste(prompt,(0,35))
    draw.text((5,5),'Saved seed prompts (display crop)',fill='black')
    for i,mask in enumerate(masks):
        mi=Image.fromarray(mask.astype('uint8')*255); mi.save(candidates/f'{i}.png')
        cut=Image.composite(Image.fromarray(rgb),Image.new('RGB',size,(225,225,225)),mi).crop(crop)
        cut.thumbnail((480,370)); page.paste(cut,((i+1)*480,35))
        draw.text(((i+1)*480+5,5),f'Candidate {i}: score={scores[i]:.4f}; selected={i==chosen}',fill='black')
    page.save(out/'seed_review.jpg',quality=95)
    if not masks[chosen].any(): raise ValueError('Empty selected seed')
    Image.fromarray(masks[chosen].astype('uint8')*255).save(out/'seed_mask.png')
    names=[manifest['records'][0]['image']]
    review_page(names,Path(cfg['images']),None,out/'seed_full_review.jpg',spec.get('review_crop_xyxy'),out/'seed_mask.png')
    write_json(out/'seed_result.json',{'selected':chosen,'scores_not_measured_accuracy':scores.tolist(),
        'torch':torch.__version__,'device':spec['device'],'postprocessing':False,
        'review_pages':{'seed_review.jpg':names,'seed_full_review.jpg':names},
        'status':'generated_pending_visual_review','seed_mask_sha256':sha256(out/'seed_mask.png')})


def review_page(names, root, masks, target, crop=None, single_mask=None):
    canvas=Image.new('RGB',(1920,len(names)*300),'white'); draw=ImageDraw.Draw(canvas)
    for row,name in enumerate(names):
        with Image.open(root/name) as original: rgb=original.convert('RGB')
        with Image.open(single_mask or (masks/(name+'.png'))) as source: mask=source.copy()
        box=crop or [0,0,*rgb.size]
        retained=Image.composite(rgb,Image.new('RGB',rgb.size,(225,225,225)),mask)
        scene=rgb.copy(); ImageDraw.Draw(scene).rectangle(box,outline='cyan',width=max(2,rgb.width//600))
        for col,im in enumerate([scene,retained,rgb.crop(box),retained.crop(box)]):
            im.thumbnail((480,270)); canvas.paste(im,(col*480,row*300+24))
        draw.text((5,row*300+4),f'{name} | full source / full retained / fixed crop / retained crop',fill='black')
    canvas.save(target,quality=95)


def propagate(cfg, prepared, seed_dir, out):
    torch,manifest=runtime(cfg,prepared)
    from sam2.build_sam import build_sam2_video_predictor
    spec=cfg['mask_generation']; size=tuple(manifest['image_size'])
    names=[r['image'] for r in manifest['records']]
    masks=out/'masks'; masks.mkdir()
    result={'status':'running','records':[],'started':datetime.now(timezone.utc).isoformat(),
        'source':'new_inference','torch':torch.__version__,'device':spec['device'],
        'postprocessing':False,'mask_threshold':'logit > 0','additional_prompts':0}
    write_json(out/'propagation.json',result); start=time.monotonic()
    try:
        predictor=build_sam2_video_predictor(spec['model_config'],manifest['source']['checkpoint']['path'],
            device=spec['device'],apply_postprocessing=False)
        with torch.inference_mode():
            state=predictor.init_state(str(prepared/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            with Image.open(seed_dir/'seed_mask.png') as im: initial=np.asarray(im)>0
            predictor.add_new_mask(state,frame_idx=0,obj_id=1,mask=initial)
            for index,ids,logits in predictor.propagate_in_video(state):
                if ids != [1] or index != len(result['records']):
                    raise ValueError('Unexpected frame order or object IDs')
                mask=(logits[0,0]>0).cpu().numpy()
                if mask.shape != size[::-1] or not mask.any(): raise ValueError('Invalid propagated mask')
                path=masks/(names[index]+'.png'); path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(mask.astype('uint8')*255).save(path)
                result['records'].append({'frame':index,'image':names[index],'sha256':sha256(path),'area_fraction':float(mask.mean())})
                write_json(out/'propagation.json',result)
                print(f'Propagated {index+1}/{len(names)} area={mask.mean():.5f}',flush=True)
        if len(result['records']) != len(names): raise ValueError('Incomplete propagation')
        write_json(out/'mask_audit.json',validate_masks(Path(cfg['images']),names,masks))
        pages={}
        for start_index in range(0,len(names),6):
            subset=names[start_index:start_index+6]; page=f'review_{start_index:03d}.jpg'
            review_page(subset,Path(cfg['images']),masks,out/page,spec.get('review_crop_xyxy'))
            pages[page]=subset
        write_json(out/'review_pages.json',pages)
        verify_fingerprints(read_json(prepared/'inputs.json'))
        result['status']='generated_pending_visual_review'
    except Exception as exc:
        result.update(status='failed',error=str(exc)); raise
    finally:
        result['seconds']=time.monotonic()-start
        write_json(out/'propagation.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','seed','propagate'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--seed',type=Path)
    args=parser.parse_args(); cfg=read_json(args.config)
    if args.stage=='prepare': prepare(cfg,args.out)
    elif args.stage=='seed': seed(cfg,args.prepared,args.out)
    else: propagate(cfg,args.prepared,args.seed,args.out)


if __name__=='__main__': main()
