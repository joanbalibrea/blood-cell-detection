"""Build and upload a private Linux wheel/weights dataset for offline Kaggle training."""
import hashlib
import json
import subprocess
import sys
import requests
from manage import ROOT, api, save

PACKAGES = ['ultralytics==8.4.32', 'ultralytics-thop==2.2.4', 'polars==1.38.1',
            'polars-runtime-32==1.38.1', 'opencv-python==4.11.0.86']

if __name__ == '__main__':
    folder = ROOT / 'staging/offline'
    packages = folder / 'packages'
    packages.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'download', '--dest', str(packages),
                           '--only-binary=:all:', '--platform', 'manylinux2014_x86_64',
                           '--python-version', '313', '--no-deps', *PACKAGES])
    # Remove superseded wheels from this generated staging directory only.
    for wheel in packages.glob('*.whl'):
        if not any(wheel.name.startswith(p.split('==')[0].replace('-', '_')+'-'+p.split('==')[1]+'-') for p in PACKAGES):
            wheel.unlink()
    for name in ('yolo26s.pt', 'yolo26n.pt'):
        weights = folder / name
        if not weights.exists():
            url = f'https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}'
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            weights.write_bytes(response.content)
    manifest = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ('.whl', '.pt')}
    save(folder / 'checksums.json', manifest)
    save(ROOT / 'reports/offline-assets.json', manifest)
    client = api()
    ref = client.get_config_value('username') + '/yolo26-offline-training-assets'
    save(folder / 'dataset-metadata.json', {'id': ref, 'title': 'YOLO26 Offline Training Assets',
         'licenses': [{'name': 'other'}], 'description': 'Private research bundle. Original package and model licenses apply.'})
    if '--update' in sys.argv:
        print(client.dataset_create_version(str(folder), 'Refresh offline wheels and AMP check weights',
                                            quiet=True, convert_to_csv=False, dir_mode='zip'))
    else:
        print(client.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode='zip'))
    metadata_path = ROOT / 'kaggle/kernel-metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['dataset_sources'] = [r for r in metadata['dataset_sources'] if r != ref] + [ref]
    metadata['enable_internet'] = False
    save(metadata_path, metadata)
    print('Offline assets attached in local kernel metadata. Run manage.py launch to submit.')
