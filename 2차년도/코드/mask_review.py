"""Bind an explicit visual review to the precise generated artifacts it inspected."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from pipeline_contracts import read_json, write_json, verify_fingerprints, sha256


class ReviewRequired(Exception):
    pass


def review_binding(run, kind):
    state=read_json(Path(run)/'run.json')
    stage='seed' if kind=='seed' else 'propagate'
    record=state['stages'][stage]
    if record['status'] != 'completed': raise ValueError('Review source stage is not complete')
    verify_fingerprints(record['outputs'])
    directory=Path(record['directory'])
    pages=read_json(directory/'seed_result.json')['review_pages'] if kind=='seed' else read_json(directory/'review_pages.json')
    return record,pages


def validate_review(receipt, record, pages, kind):
    if receipt.get('kind') != kind or receipt.get('artifact_hashes') != record['outputs']:
        raise ValueError('Review does not match current artifact version')
    if receipt.get('decision') != 'accept':
        raise ReviewRequired('Review requires changes; revise prompts/settings and use a new run')
    if receipt.get('reviewer_type') not in ('assistant','human') or not receipt.get('reviewer'):
        raise ValueError('Explicit reviewer identity and type required')
    if set(receipt.get('inspected_pages',[])) != set(pages):
        raise ValueError('Review must cover all generated review pages')
    if not receipt.get('scope') or not receipt.get('notes'):
        raise ValueError('Review scope and limitations required')


def require_review(run,kind,out):
    record,pages=review_binding(run,kind)
    path=Path(run)/'reviews'/f'{kind}.json'
    write_json(out/'request.json',{'kind':kind,'receipt_path':str(path),'pages':pages,
        'directory':record['directory'],'status':'review_required',
        'note':'Integrity checks do not approve visual quality. Human or assistant review must be explicit.'})
    if not path.is_file(): raise ReviewRequired(f'Visual review required: {path}')
    receipt=read_json(path); validate_review(receipt,record,pages,kind)
    write_json(out/'review.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--kind',choices=['seed','masks'],required=True)
    parser.add_argument('--reviewer',required=True)
    parser.add_argument('--reviewer-type',choices=['assistant','human'],required=True)
    parser.add_argument('--decision',choices=['accept','needs_changes'],required=True)
    parser.add_argument('--scope',required=True)
    parser.add_argument('--notes',required=True)
    parser.add_argument('--inspected',nargs='+',required=True,help='Basenames of every review page actually inspected')
    args=parser.parse_args(); record,pages=review_binding(args.run,args.kind)
    if len(args.inspected) != len(set(args.inspected)) or set(args.inspected) != set(pages):
        parser.error('List every reviewed page exactly once')
    receipt={'kind':args.kind,'decision':args.decision,'reviewer':args.reviewer,'reviewer_type':args.reviewer_type,
        'scope':args.scope,'notes':args.notes,'inspected_pages':args.inspected,
        'image_names':list(dict.fromkeys(n for p in args.inspected for n in pages[p])),
        'artifact_hashes':record['outputs'],'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'human_work_seconds':None,'pixel_accuracy_measured':False}
    folder=args.run/'reviews'; folder.mkdir(exist_ok=True)
    path=folder/f'{args.kind}.json'
    if path.exists(): parser.error('Review already exists; keep immutable receipts and use a new run for changed review')
    write_json(path,receipt); print(path)


if __name__=='__main__': main()
