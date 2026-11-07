# run002: 10문제 활성값(.pt) + 메타데이터를 내 Google Drive에 백업
# 문제별 ZIP은 원래 .pt와 메타데이터를 그대로 보존합니다. 원본은 삭제하지 않습니다.
# 재실행하면 SHA-256이 일치하는 파일은 건너뛰며, 충돌 파일은 덮어쓰지 않습니다.
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil
from google.colab import drive

SOURCE = Path('/content/run002_release')
DRIVE_DIR = Path('/content/drive/MyDrive/mllm-line-tracing/experiments/exp001_qwen3vl4b_trace10/run002_artifacts')
MANIFEST_SHA256 = '7d86f02d1d52ab50a40a1abe0028334e1773f5a4b854bb645efe7f7ddca35e5e'

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def check_file(path, size, expected_hash):
    if not path.is_file() or path.stat().st_size != size:
        raise RuntimeError(f'파일 크기/존재 확인 실패: {path}')
    if file_sha256(path) != expected_hash:
        raise RuntimeError(f'SHA-256 불일치: {path}')

manifest_path = SOURCE / 'artifact_manifest.json'
assert file_sha256(manifest_path) == MANIFEST_SHA256, '고정된 run002 manifest와 다릅니다.'
manifest = json.loads(manifest_path.read_text())
assert manifest['status'] == 'complete' and manifest['run_name'] == 'run002'
assert manifest['case_count'] == 10 and manifest['tensor_file_count'] == 2060
archives = manifest['archives']
assert len(archives) == 11 and sum(a['kind'] == 'case' for a in archives) == 10
assert sum(a['kind'] == 'metadata' for a in archives) == 1
assert len({a['file'] for a in archives}) == 11
for entry in archives:
    assert Path(entry['file']).name == entry['file'], '잘못된 아카이브 경로'
    check_file(SOURCE / entry['file'], entry['bytes'], entry['sha256'])
print('원본 확인 완료: 10문제, 텐서 2,060개, 메타데이터 포함.', flush=True)
print('Drive 연결 후 백업을 시작합니다.', flush=True)
drive.mount('/content/drive')
assert Path('/content/drive/MyDrive').is_dir(), 'Google Drive 연결이 필요합니다.'
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

artifacts = archives + [{
    'file': 'artifact_manifest.json',
    'bytes': manifest_path.stat().st_size,
    'sha256': MANIFEST_SHA256,
    'kind': 'manifest',
}]
verified = []
for index, entry in enumerate(artifacts, 1):
    source = SOURCE / entry['file']
    target = DRIVE_DIR / entry['file']
    if target.exists():
        check_file(target, entry['bytes'], entry['sha256'])
        action = '이미 저장됨 / 해시 일치'
    else:
        partial = DRIVE_DIR / (entry['file'] + '.partial')
        with source.open('rb') as reader, partial.open('wb') as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        check_file(partial, entry['bytes'], entry['sha256'])
        partial.replace(target)
        check_file(target, entry['bytes'], entry['sha256'])
        action = '복사 완료 / 해시 일치'
    verified.append({'file': entry['file'], 'bytes': entry['bytes'], 'sha256': entry['sha256']})
    print(f'[{index:02d}/{len(artifacts)}] {entry["file"]}: {action}', flush=True)

receipt = {
    'status': 'complete',
    'verified_at_utc': datetime.now(timezone.utc).isoformat(),
    'destination': str(DRIVE_DIR),
    'run_name': 'run002',
    'case_count': 10,
    'tensor_file_count': 2060,
    'tensor_files_uncompressed_bytes': manifest['source_tensor_bytes'],
    'archive_count': 11,
    'archive_bytes': manifest['total_archive_bytes'],
    'manifest_sha256': MANIFEST_SHA256,
    'verification': 'All 12 destination files reread and SHA-256 matched.',
    'files': verified,
    'restore': manifest['restore'],
}
receipt_path = DRIVE_DIR / 'drive_backup_receipt.json'
receipt_partial = DRIVE_DIR / 'drive_backup_receipt.json.partial'
with receipt_partial.open('w') as stream:
    json.dump(receipt, stream, indent=2, ensure_ascii=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
receipt_partial.replace(receipt_path)
assert json.loads(receipt_path.read_text()) == receipt
print('\nDRIVE BACKUP COMPLETE', flush=True)
print(f'10문제 / 원시 .pt 2,060개 / ZIP 11개 + manifest + 백업 확인서', flush=True)
print(f'압축 백업 크기: {manifest["total_archive_bytes"] / 2**30:.3f} GiB', flush=True)
print(f'저장 위치: {DRIVE_DIR}', flush=True)
print(f'확인서: {receipt_path}', flush=True)
