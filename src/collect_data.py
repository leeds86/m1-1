"""Download official observations without logging credentials."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
SERIES = {
    'base_rate': ('722Y001', 'M', '201512', '202512', '0101000'),
    'usdkrw': ('731Y004', 'M', '201512', '202512', '0000001/0000100'),
    'kospi': ('901Y014', 'M', '201512', '202512', '1070000'),
    'cpi': ('901Y009', 'M', '201412', '202512', '0'),
    'unemployment': ('901Y027', 'M', '201512', '202512', 'I61BC/I28B'),
}
GOLD_URL = 'https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx'


def read_key():
    key = os.environ.get('ECOS_API_KEY', '')
    path = ROOT / '.env'
    if not key and path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            name, sep, value = line.partition('=')
            if sep and name.strip() == 'ECOS_API_KEY':
                key = value.strip().strip('\"').strip("'")
    if not key or key == 'sample':
        raise ValueError('Set your personal ECOS_API_KEY in .env')
    return key


def download(url):
    try:
        with urlopen(url, timeout=45) as response:
            return response.read()
    except Exception:
        raise RuntimeError('Source request failed; URL hidden to protect credentials') from None


def collect():
    key = read_key()
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {'retrieved_at': datetime.now(timezone.utc).isoformat(), 'sources': {}}
    for name, (table, cycle, start, end, item) in SERIES.items():
        tail = f'{table}/{cycle}/{start}/{end}/{item}'
        payload = download(f'https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/1000/{tail}')
        result = json.loads(payload)
        if 'StatisticSearch' not in result:
            code = result.get('RESULT', {}).get('CODE', 'UNKNOWN')
            raise RuntimeError(f'{name}: ECOS response {code}')
        block = result['StatisticSearch']
        if block['list_total_count'] != len(block['row']):
            raise ValueError(f'{name}: incomplete response')
        (RAW / f'{name}.json').write_bytes(payload)
        manifest['sources'][name] = {'table': table, 'item': item, 'cycle': cycle,
            'start': start, 'end': end, 'rows': len(block['row']),
            'source_url': 'https://ecos.bok.or.kr/', 'sha256': hashlib.sha256(payload).hexdigest()}
        print(f'{name}: {len(block["row"])} observations')
    payload = download(GOLD_URL)
    if not payload.startswith(b'PK'):
        raise ValueError('Gold source is not an XLSX archive')
    (RAW / 'worldbank_monthly.xlsx').write_bytes(payload)
    manifest['sources']['gold_usd'] = {'source_url': GOLD_URL, 'sha256': hashlib.sha256(payload).hexdigest()}
    (RAW / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Official raw files saved; credentials omitted')


if __name__ == '__main__':
    collect()
