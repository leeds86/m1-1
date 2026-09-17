"""Monthly alignment and explicit percentage/percentage-point transforms."""
import numpy as np
import pandas as pd
import json
from pathlib import Path

PRICE_COLUMNS = {'usdkrw', 'kospi', 'gold_usd'}


def validate_monthly(series, start, end):
    result = series.copy()
    result.index = pd.PeriodIndex(result.index, freq='M')
    if result.index.has_duplicates:
        raise ValueError('Duplicate month')
    expected = pd.period_range(pd.Period(start, freq='M'), pd.Period(end, freq='M'), freq='M')
    result = result.reindex(expected)
    if not np.isfinite(pd.to_numeric(result, errors='coerce')).all():
        raise ValueError('Missing month or invalid numeric observation')
    return result.astype(float)


def changes(frame):
    return pd.DataFrame({name: (values.pct_change(fill_method=None) * 100 if name in PRICE_COLUMNS
                               else values.diff()) for name, values in frame.items()})


def build():
    root = Path(__file__).resolve().parents[1]
    raw = root / 'data/raw'
    series = {}
    for name in ['base_rate', 'usdkrw', 'kospi', 'cpi', 'unemployment']:
        rows = json.loads((raw / f'{name}.json').read_text(encoding='utf-8'))['StatisticSearch']['row']
        values = pd.Series([float(r['DATA_VALUE']) for r in rows], index=[r['TIME'] for r in rows])
        series[name] = validate_monthly(values, '201412' if name == 'cpi' else '201512', '202512')
    series['inflation_yoy'] = series.pop('cpi').pct_change(12, fill_method=None) * 100
    gold = pd.read_excel(raw / 'worldbank_monthly.xlsx', sheet_name='Monthly Prices', header=4)
    gold = gold[gold.iloc[:, 0].astype(str).str.fullmatch(r'\d{4}M\d{2}')].copy()
    gold_index = gold.iloc[:, 0].str.replace('M', '', regex=False)
    series['gold_usd'] = validate_monthly(pd.Series(gold['Gold'].values, index=gold_index), '201512', '202512')
    wide = pd.DataFrame(series).loc['2015-12':'2025-12']
    for name in wide:
        validate_monthly(wide[name], '201512', '202512')
    out = root / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    wide.index.name = 'date'
    wide.to_csv(out / 'monthly_with_baseline.csv', encoding='utf-8-sig')
    study = wide.loc['2016-01':'2025-12'].copy()
    changes(wide).loc[study.index].to_csv(out / 'monthly_changes.csv', encoding='utf-8-sig')
    long = study.reset_index().melt(id_vars='date', var_name='indicator', value_name='value')
    units = {'base_rate':'%', 'usdkrw':'KRW/USD', 'kospi':'1980-01-04=100',
             'gold_usd':'USD/troy oz', 'inflation_yoy':'% YoY', 'unemployment':'% SA'}
    long['unit'] = long.indicator.map(units)
    long['memo'] = ''
    long.loc[(long.indicator == 'gold_usd') & (long.date >= pd.Period('2025-06', freq='M')), 'memo'] = 'Gold pricing definition changed June 2025; see World Bank Description'
    long['source_id'] = long.indicator.map(lambda x: 'worldbank_pink_sheet' if x == 'gold_usd' else 'bok_ecos')
    long.sort_values(['date', 'indicator']).to_csv(out / 'economic_indicators_monthly.csv', index=False, encoding='utf-8-sig')
    print(f'Validated {len(study)} months, {len(study.columns)} indicators, {len(long)} observations')
    print('Missing values:', int(study.isna().sum().sum()))


if __name__ == '__main__':
    build()
