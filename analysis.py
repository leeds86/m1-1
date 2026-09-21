"""미션 m1-1 전체 분석을 한 파일에서 재현한다.

실행: python analysis.py
입력: data/economic_indicators_monthly.csv
출력: images/*.png, web/data/analysis.json, web/data/economic_indicators_monthly.csv
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LABELS = {
    "base_rate": "기준금리", "usdkrw": "원/달러 환율", "kospi": "KOSPI",
    "unemployment": "실업률", "inflation_yoy": "물가 상승률", "gold_usd": "금 가격",
}
UNITS = {
    "base_rate": "%", "usdkrw": "원/USD", "kospi": "포인트",
    "unemployment": "% (계절조정)", "inflation_yoy": "% (전년 동월 대비)",
    "gold_usd": "USD/트로이온스",
}
PRICE_COLUMNS = {"usdkrw", "kospi", "gold_usd"}
EVENTS = [
    {"date": "2020-03", "label": "WHO 팬데믹 선언",
     "url": "https://www.who.int/news/item/29-06-2020-covidtimeline"},
    {"date": "2022-02", "label": "러시아의 우크라이나 전면 침공",
     "url": "https://ukraine.un.org/en/173250-statement-secretary-general-%E2%80%93-ukraine"},
]


# 1. 데이터 읽기와 월간 변화 계산
# ------------------------------------------------------------


def load_data():
    """공식 자료를 합쳐 둔 월별 CSV를 읽고 기본 오류를 확인한다."""
    frame = pd.read_csv(ROOT / "data/economic_indicators_monthly.csv", index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    expected = pd.period_range("2015-12", "2025-12", freq="M")
    if frame.index.has_duplicates or not frame.index.equals(expected):
        raise ValueError("날짜가 중복되었거나 빠진 달이 있습니다.")
    if list(frame.columns) != ["base_rate", "usdkrw", "kospi", "unemployment", "inflation_yoy", "gold_usd"]:
        raise ValueError("데이터 열이 예상한 6개 지표와 다릅니다.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("빈 값이나 숫자가 아닌 값이 있습니다.")
    return frame.astype(float)


def monthly_changes(frame):
    """가격은 변화율(%), 금리·물가·실업률은 차이(%p)를 계산한다."""
    return pd.DataFrame({
        name: values.pct_change(fill_method=None).mul(100) if name in PRICE_COLUMNS else values.diff()
        for name, values in frame.items()
    })


def correlation(x, y):
    paired = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    value = None if len(paired) < 3 or not (paired.nunique() > 1).all() else float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
    return {"r": value, "n": len(paired)}


# 2. 상관관계·금리 구간·사건 전후 분석
# ------------------------------------------------------------


def analyze(frame):
    changes = monthly_changes(frame).loc["2016-01":"2025-12"]
    matrix = {a: {b: correlation(changes[a], changes[b]) for b in changes} for a in changes}
    rate_group = pd.Series(np.where(changes.base_rate > 0, "인상", np.where(changes.base_rate < 0, "인하", "동결")), index=changes.index)
    regimes = {}
    for name in ["인상", "동결", "인하"]:
        values = changes.loc[rate_group == name, "kospi"]
        regimes[name] = {"n": len(values), "mean": float(values.mean()), "median": float(values.median()),
                         "positive": int((values > 0).sum()), "values": values.tolist()}
    lags = {name: [dict(lag=lag, **correlation(changes.base_rate, changes[name].shift(-lag))) for lag in range(4)]
            for name in ["usdkrw", "kospi", "gold_usd"]}
    events = []
    for event in EVENTS:
        month = pd.Period(event["date"], freq="M")
        parts = {}
        for label, start, end in [("before", month-12, month-1), ("after", month+1, month+12)]:
            block = changes.loc[start:end]
            parts[label] = {"n": len(block), "mean_changes": block.mean().to_dict(),
                            "fx_kospi": correlation(block.usdkrw, block.kospi)}
        events.append({**event, **parts, "note": "사건 달은 제외했습니다. 함께 일어난 다른 변화도 있어 원인을 단정할 수 없습니다."})
    return {
        "correlation": matrix,
        "level_correlation": frame.loc[changes.index].corr().to_dict(),
        "regimes": regimes,
        "lags": lags,
        "events": events,
        "gold_sensitivity": correlation(changes.loc[:"2025-05", "gold_usd"], changes.loc[:"2025-05", "kospi"]),
        "rolling_fx_kospi": changes.usdkrw.rolling(24).corr(changes.kospi).replace([np.inf, -np.inf], np.nan).to_dict(),
    }


# 3. 환율 분해와 간단한 과거 예측 실험
# ------------------------------------------------------------


def predict(values, horizon, model):
    values = np.asarray(values, dtype=float)
    if model == "naive":
        return np.repeat(values[-1], horizon)
    if model == "ses":
        return np.asarray(SimpleExpSmoothing(values, initialization_method="estimated").fit().forecast(horizon))
    raise ValueError("알 수 없는 예측 모델입니다.")


def forecast(series):
    """처음 60개월부터 학습 범위를 늘리며 1·3·6개월 뒤를 시험한다."""
    values = np.asarray(series, dtype=float)
    rows = []
    for stop in range(60, len(values)):
        for model in ["naive", "ses"]:
            guesses = predict(values[:stop], 6, model)
            for horizon in [1, 3, 6]:
                target = stop + horizon - 1
                if target < len(values):
                    rows.append({"model": model, "origin": stop-1, "horizon": horizon,
                                 "prediction": float(guesses[horizon-1]), "actual": float(values[target]),
                                 "error": float(values[target] - guesses[horizon-1])})
    scores = []
    for model in ["naive", "ses"]:
        for horizon in [1, 3, 6]:
            errors = [row["error"] for row in rows if row["model"] == model and row["horizon"] == horizon]
            scores.append({"model": model, "horizon": horizon, "n": len(errors),
                           "mae": float(np.mean(np.abs(errors)))})
    bands = {}
    for horizon in [1, 3, 6]:
        errors = [row["error"] for row in rows if row["model"] == "naive" and row["horizon"] == horizon]
        bands[str(horizon)] = np.quantile(errors, [.1, .9]).tolist()
    parts = STL(values, period=12, robust=True).fit()
    return {
        "cutoff": str(series.index[-1]),
        "predictions": {model: predict(values, 6, model).tolist() for model in ["naive", "ses"]},
        "scores": scores, "baseline_error_quantiles_10_90": bands, "validation": rows,
        "decomposition": {"trend": parts.trend.tolist(), "seasonal": parts.seasonal.tolist(), "residual": parts.resid.tolist()},
        "note": "2025-12 기준 과거 실험입니다. 오차 범위는 미래를 보장하지 않습니다.",
    }


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def asset_data(frame):
    prices = frame[["kospi", "usdkrw", "gold_usd"]].copy()
    prices["gold_krw"] = prices.gold_usd * prices.usdkrw
    returns = prices.pct_change(fill_method=None).mul(100).loc["2016-01":"2025-12"]
    drawdown = prices.div(prices.cummax()).sub(1).mul(100).loc[returns.index]
    names = {"kospi": "KOSPI", "usdkrw": "달러(원화 가격)", "gold_usd": "금(달러 가격)", "gold_krw": "금(원화 환산 추정)"}
    return prices.loc[returns.index], returns, drawdown, names


# 4. REPORT에서 사용하는 그래프 만들기
# ------------------------------------------------------------


def make_charts(frame, stats, prediction):
    """REPORT.md에서 사용하는 13개 그래프를 다시 만든다."""
    image_dir = ROOT / "images"
    image_dir.mkdir(exist_ok=True)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    if "Malgun Gothic" in fonts:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    levels = frame.loc["2016-01":"2025-12"]
    changes = monthly_changes(frame).loc[levels.index]
    dates = levels.index.to_timestamp()

    def save(name):
        plt.tight_layout(); plt.savefig(image_dir / f"{name}.png", bbox_inches="tight"); plt.close()

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for ax, key in zip(axes.flat, levels):
        ax.plot(dates, levels[key], color="#126e82", label="월별 값")
        ax.plot(dates, levels[key].rolling(6).mean(), color="#dc8525", label="6개월 이동평균")
        ax.set_title(LABELS[key]); ax.set_ylabel(UNITS[key]); ax.grid(alpha=.2); ax.legend(fontsize=8)
    save("01_trends")

    core = ["usdkrw", "kospi", "gold_usd"]
    fig, ax = plt.subplots(figsize=(11, 4))
    for key in core: ax.plot(dates, levels[key] / levels[key].iloc[0] * 100, label=LABELS[key])
    ax.set_title("가격지표 비교 (2016-01=100, 배당 미포함)"); ax.legend(); ax.grid(alpha=.2); save("02_indexed")
    fig, ax = plt.subplots(figsize=(11, 4))
    for key in core: ax.plot(dates, changes[key], label=LABELS[key], alpha=.7)
    ax.set_ylabel("전월 대비 %"); ax.legend(); ax.grid(alpha=.2); save("03_changes")

    corr = changes.corr(); fig, ax = plt.subplots(figsize=(8, 7)); image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(6), [LABELS[key] for key in changes], rotation=35, ha="right")
    ax.set_yticks(range(6), [LABELS[key] for key in changes])
    for i in range(6):
        for j in range(6): ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("월간 변화 간 Pearson 상관계수 · n=120"); fig.colorbar(image, ax=ax); save("04_correlation")

    fig, ax = plt.subplots(figsize=(9, 4))
    for key in core: ax.plot(range(4), [row["r"] for row in stats["lags"][key]], marker="o", label=LABELS[key])
    ax.set_xlabel("금리 변화 후 개월 (0~3), n=120~117"); ax.set_ylabel("상관계수"); ax.legend(); ax.grid(alpha=.2); save("05_lags")
    groups = stats["regimes"]; fig, ax = plt.subplots(figsize=(8, 4)); names = list(groups)
    ax.boxplot([groups[name]["values"] for name in names], tick_labels=[f'{name} (n={groups[name]["n"]})' for name in names])
    ax.set_ylabel("KOSPI 월간 변화율 (%)"); ax.set_title("금리 변화가 있었던 달 기준"); save("06_regimes")
    fig, ax = plt.subplots(figsize=(11, 4)); ax.plot(dates, levels.kospi, label="KOSPI")
    for event in EVENTS: ax.axvline(pd.Timestamp(event["date"]), ls="--", alpha=.6, label=event["label"])
    ax.legend(); ax.set_title("사건 시점은 원인 입증이 아닌 배경 정보"); save("07_events")

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, (label, values) in zip(axes, [("원본", levels.usdkrw), *prediction["decomposition"].items()]):
        ax.plot(dates, values); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].set_title("환율 STL 분해 · 12개월 주기 가정"); save("08_decomposition")
    future = pd.period_range("2026-01", periods=6, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(dates[-24:], levels.usdkrw.iloc[-24:], label="관측")
    for model, values in prediction["predictions"].items(): ax.plot(future, values, "--o", label=model)
    for horizon in [1, 3, 6]:
        point = prediction["predictions"]["naive"][horizon-1]; low, high = prediction["baseline_error_quantiles_10_90"][str(horizon)]
        ax.vlines(future[horizon-1], point+low, point+high, color="#9b6635", lw=5, alpha=.35)
    ax.set_title("2025-12 기준 과거 예측 실험"); ax.set_ylabel("원/USD"); ax.legend(); save("09_forecast")

    prices, returns, drawdown, asset_names = asset_data(frame); asset_dates = prices.index.to_timestamp()
    standardized = (levels - levels.mean()) / levels.std(ddof=1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for key in levels: axes[0].plot(dates, standardized[key], label=LABELS[key], alpha=.8)
    axes[0].set_title("질문 1 · 6개 지표의 상대적 위치 겹쳐 보기"); axes[0].set_ylabel("평균에서 떨어진 거리"); axes[0].legend(ncol=3, fontsize=9); axes[0].grid(alpha=.2)
    for key in prices: axes[1].plot(asset_dates, prices[key] / prices[key].iloc[0] * 100, label=asset_names[key])
    axes[1].set_title("자산 가격 비교 · 2016-01=100"); axes[1].set_ylabel("기준점 대비 가격지수"); axes[1].legend(fontsize=9); axes[1].grid(alpha=.2); save("10_overlay")

    down = returns.loc[returns.kospi < 0]; fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    down.rename(columns=asset_names).boxplot(ax=axes[0], rot=20); axes[0].axhline(0, color="gray", lw=1)
    axes[0].set_title(f"질문 3 · 주식 하락 {len(down)}개월의 가격 변화"); axes[0].set_ylabel("월간 변화 (%)")
    for key in ["usdkrw", "gold_krw"]: axes[1].scatter(down.kospi, down[key], label=asset_names[key], alpha=.7)
    axes[1].axhline(0, color="gray", lw=1); axes[1].set_xlabel("KOSPI 월간 변화 (%)"); axes[1].set_ylabel("비교 자산 월간 변화 (%)"); axes[1].legend(fontsize=9); save("11_down_months")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].bar([asset_names[key] for key in prices], [returns[key].std(ddof=1) for key in prices], color="#327e72")
    axes[0].set_title("질문 4 · 월별 변화의 흔들림"); axes[0].set_ylabel("월간 변화율 표준편차 (%p)")
    for key in prices: axes[1].plot(asset_dates, drawdown[key], label=asset_names[key])
    axes[1].set_title("이전 최고 가격 대비 하락"); axes[1].set_ylabel("고점 대비 하락 (%)"); axes[1].legend(fontsize=9); axes[1].grid(alpha=.2); save("12_risk")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True); keys = ["usdkrw", "gold_usd", "gold_krw"]
    for ax, event in zip(axes, EVENTS):
        month = pd.Period(event["date"], freq="M")
        for offset, start, end, label in [(-.18, month-12, month-1, "이전 12개월"), (.18, month+1, month+12, "이후 12개월")]:
            block = returns.loc[start:end]
            values = [correlation(block.kospi, block[key])["r"] for key in keys]
            bars = ax.bar([i+offset for i in range(3)], values, width=.36, label=label); ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xticks(range(3), [asset_names[key] for key in keys], rotation=15); ax.set_title(event["date"] + " · " + event["label"], fontsize=10); ax.set_ylim(-1, 1); ax.axhline(0, color="gray", lw=1); ax.legend(fontsize=8)
    axes[0].set_ylabel("KOSPI 월간 변화와의 상관계수"); fig.suptitle("질문 5 · 사건 전후 관계 비교"); save("13_crisis_relationships")


# 5. 보너스 웹 대시보드용 자료 만들기
# ------------------------------------------------------------


def export_web(frame, stats, prediction):
    levels = frame.loc["2016-01":"2025-12"]
    changes = monthly_changes(frame).loc[levels.index]
    payload = {
        "schema_version": 1, "dates": levels.index.astype(str).tolist(), "labels": LABELS, "units": UNITS,
        "levels": levels.to_dict("list"), "changes": changes.to_dict("list"), "analysis": stats, "forecast": prediction,
        "summary": {key: {"count": len(levels), "unit": UNITS[key], "mean": float(levels[key].mean()),
                          "min": float(levels[key].min()), "max": float(levels[key].max()), "last": float(levels[key].iloc[-1])}
                    for key in levels},
    }
    target = ROOT / "web/data"; target.mkdir(parents=True, exist_ok=True)
    (target / "analysis.json").write_text(json.dumps(safe(payload), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    long = levels.reset_index(names="date").melt(id_vars="date", var_name="indicator", value_name="value")
    long["date"] = long.date.astype(str); long["unit"] = long.indicator.map(UNITS)
    long.to_csv(target / "economic_indicators_monthly.csv", index=False, encoding="utf-8-sig")


def main():
    frame = load_data()
    stats = analyze(frame)
    prediction = forecast(frame.loc["2016-01":"2025-12", "usdkrw"])
    make_charts(frame, stats, prediction)
    export_web(frame, stats, prediction)
    print("완료: 120개월 × 6개 지표, 그래프 13개, 웹 데이터 생성")
    print(f"환율과 KOSPI 월간 변화 상관계수: {stats['correlation']['usdkrw']['kospi']['r']:.3f}")


if __name__ == "__main__":
    main()
