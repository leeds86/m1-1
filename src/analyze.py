"""Exploratory associations, never causal estimates."""
import numpy as np
import pandas as pd
from .clean_data import changes

EVENTS = [
    {'date': '2020-03', 'label': 'WHO 팬데믹 선언', 'url': 'https://www.who.int/news/item/29-06-2020-covidtimeline'},
    {'date': '2022-02', 'label': '러시아의 우크라이나 전면 침공', 'url': 'https://ukraine.un.org/en/173250-statement-secretary-general-%E2%80%93-ukraine'},
]


def pair(x, y):
    values = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(values)
    r = None
    if n >= 3 and (values.nunique() > 1).all():
        r = float(values.iloc[:, 0].corr(values.iloc[:, 1]))
    return {'r': r, 'n': n}


def lag_pairs(x, y, lag):
    if lag < 0:
        raise ValueError('lag must be nonnegative')
    return pair(x, y.shift(-lag))


def analyze(wide):
    delta = changes(wide).loc['2016-01':'2025-12']
    levels = wide.loc[delta.index]
    matrix = {a: {b: pair(delta[a], delta[b]) for b in delta} for a in delta}
    regimes = pd.Series(np.where(delta.base_rate > 0, '인상', np.where(delta.base_rate < 0, '인하', '동결')), index=delta.index)
    groups = {}
    for name in ['인상', '동결', '인하']:
        subset = delta.loc[regimes == name, 'kospi']
        groups[name] = {'n': len(subset), 'mean': float(subset.mean()), 'median': float(subset.median()),
                        'positive': int((subset > 0).sum()), 'values': subset.tolist()}
    events = []
    for event in EVENTS:
        month = pd.Period(event['date'], freq='M')
        periods = {}
        for label, begin, end in [('before', month - 12, month - 1), ('after', month + 1, month + 12)]:
            section = delta.loc[begin:end]
            periods[label] = {'n': len(section), 'mean_changes': section.mean().to_dict(),
                             'fx_kospi': pair(section.usdkrw, section.kospi)}
        events.append({**event, **periods, 'note': '사건 달 제외. 다른 충격과 정책 변화가 겹쳐 인과효과를 뜻하지 않습니다.'})
    return {'correlation': matrix, 'level_correlation': levels.corr().to_dict(), 'regimes': groups,
            'lags': {name: [dict(lag=k, **lag_pairs(delta.base_rate, delta[name], k)) for k in range(4)]
                     for name in ['usdkrw', 'kospi', 'gold_usd']},
            'events': events,
            'gold_sensitivity': pair(delta.loc[:'2025-05', 'gold_usd'], delta.loc[:'2025-05', 'kospi']),
            'rolling_fx_kospi': delta.usdkrw.rolling(24).corr(delta.kospi).replace([np.inf, -np.inf], np.nan).to_dict()}
