"""Historical forecast experiment, using expanding training windows."""
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL


def predict(values, horizon, model):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite training data required')
    if model == 'naive':
        return np.repeat(values[-1], horizon)
    if model == 'ses':
        return np.asarray(SimpleExpSmoothing(values, initialization_method='estimated').fit().forecast(horizon))
    raise ValueError('Unknown model')


def backtest(series, initial=60, horizons=(1, 3, 6), models=('naive', 'ses')):
    values = np.asarray(series, dtype=float)
    rows = []
    for stop in range(initial, len(values)):
        for model in models:
            predicted = predict(values[:stop], max(horizons), model)
            for h in horizons:
                target = stop + h - 1
                if target < len(values):
                    rows.append({'model': model, 'origin': stop - 1, 'horizon': h,
                                 'prediction': float(predicted[h - 1]), 'actual': float(values[target]),
                                 'error': float(values[target] - predicted[h - 1])})
    return rows


def forecast(series):
    rows = backtest(series)
    scores = []
    for model in ['naive', 'ses']:
        for h in [1, 3, 6]:
            errors = [r['error'] for r in rows if r['model'] == model and r['horizon'] == h]
            scores.append({'model': model, 'horizon': h, 'n': len(errors), 'mae': float(np.mean(np.abs(errors)))})
    # Preserve both models: model selection and evaluation on the same sample is optimistic.
    predicted = {m: predict(series, 6, m).tolist() for m in ['naive', 'ses']}
    bands = {}
    for h in [1, 3, 6]:
        errors = [r['error'] for r in rows if r['model'] == 'naive' and r['horizon'] == h]
        bands[str(h)] = np.quantile(errors, [.1, .9]).tolist()
    result = STL(np.asarray(series), period=12, robust=True).fit()
    return {'cutoff': str(series.index[-1]), 'predictions': predicted, 'scores': scores,
            'baseline_error_quantiles_10_90': bands, 'validation': rows,
            'decomposition': {'trend': result.trend.tolist(), 'seasonal': result.seasonal.tolist(), 'residual': result.resid.tolist()},
            'note': '2025-12 기준 역사적 실험. 오차 띠는 과거 오차의 10~90백분위이며 보장된 신뢰구간이 아닙니다. 개정된 자료를 이용해 당시 정보집합을 완전히 재현하지 못합니다.'}
