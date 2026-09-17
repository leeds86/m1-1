"""Rebuild charts, report, notebook and public data from checked local inputs."""
import json
import shutil
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'tmp/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import nbformat
from .analyze import analyze, EVENTS
from .forecast import forecast
from .clean_data import changes

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'base_rate':'기준금리', 'usdkrw':'원/달러 환율', 'kospi':'KOSPI', 'unemployment':'실업률', 'inflation_yoy':'물가 상승률', 'gold_usd':'금 가격'}
UNITS = {'base_rate':'%', 'usdkrw':'원/USD', 'kospi':'포인트', 'unemployment':'% (계절조정)', 'inflation_yoy':'% (전년 동월 대비)', 'gold_usd':'USD/트로이온스'}


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


def run():
    wide = pd.read_csv(ROOT/'data/processed/monthly_with_baseline.csv', index_col=0)
    wide.index = pd.PeriodIndex(wide.index, freq='M')
    levels = wide.loc['2016-01':'2025-12']
    delta = changes(wide).loc[levels.index]
    stats = analyze(wide)
    prediction = forecast(levels.usdkrw)
    dates = levels.index.astype(str).tolist()
    payload = {'schema_version':1, 'dates':dates, 'labels':LABELS, 'units':UNITS,
        'levels':levels.to_dict('list'), 'changes':delta.to_dict('list'), 'analysis':stats, 'forecast':prediction,
        'summary':{c:{'count':len(levels), 'unit':UNITS[c], 'mean':float(levels[c].mean()),
                      'min':float(levels[c].min()), 'max':float(levels[c].max()), 'last':float(levels[c].iloc[-1])} for c in levels}}
    target = ROOT/'web/data'
    target.mkdir(parents=True, exist_ok=True)
    (target/'analysis.json').write_text(json.dumps(safe(payload), ensure_ascii=False, allow_nan=False), encoding='utf-8')
    shutil.copyfile(ROOT/'data/processed/economic_indicators_monthly.csv', target/'economic_indicators_monthly.csv')
    image_dir = ROOT/'images'
    image_dir.mkdir(exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    if 'Malgun Gothic' in fonts:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 140
    t = levels.index.to_timestamp()

    def save(name):
        plt.tight_layout()
        plt.savefig(image_dir/f'{name}.png', bbox_inches='tight')
        plt.close()

    fig, axes = plt.subplots(3,2,figsize=(12,9))
    for ax,c in zip(axes.flat, levels):
        ax.plot(t, levels[c], color='#126e82', label='월별 값')
        ax.plot(t, levels[c].rolling(6).mean(), color='#dc8525', label='6개월 이동평균')
        ax.set_title(LABELS[c]); ax.set_ylabel(UNITS[c]); ax.grid(alpha=.2); ax.legend(fontsize=8)
    save('01_trends')
    core = ['usdkrw','kospi','gold_usd']
    fig,ax = plt.subplots(figsize=(11,4))
    for c in core:
        ax.plot(t, levels[c]/levels[c].iloc[0]*100, label=LABELS[c])
    ax.set_title('가격지표 비교 (2016-01=100, 배당 미포함)'); ax.legend(); ax.grid(alpha=.2)
    save('02_indexed')
    fig,ax = plt.subplots(figsize=(11,4))
    for c in core: ax.plot(t, delta[c], label=LABELS[c], alpha=.7)
    ax.set_ylabel('전월 대비 %'); ax.legend(); ax.grid(alpha=.2); save('03_changes')
    fig,ax = plt.subplots(figsize=(8,7))
    corr = delta.corr(); im = ax.imshow(corr, vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(6),[LABELS[c] for c in delta],rotation=35,ha='right')
    ax.set_yticks(range(6),[LABELS[c] for c in delta])
    for i in range(6):
        for j in range(6): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center')
    ax.set_title('월간 변화 간 Pearson 상관계수 · n=120'); fig.colorbar(im,ax=ax); save('04_correlation')
    fig,ax = plt.subplots(figsize=(9,4))
    for c in core: ax.plot(range(4),[r['r'] for r in stats['lags'][c]],marker='o',label=LABELS[c])
    ax.set_xlabel('금리 변화 후 개월 (0~3), n=120~117'); ax.set_ylabel('상관계수'); ax.legend(); ax.grid(alpha=.2); save('05_lags')
    fig,ax = plt.subplots(figsize=(8,4))
    groups=stats['regimes']; names=list(groups)
    ax.boxplot([groups[n]['values'] for n in names],tick_labels=[f'{n} (n={groups[n]["n"]})' for n in names])
    ax.set_ylabel('KOSPI 월간 변화율 (%)'); ax.set_title('금리 변화가 있었던 달 기준'); save('06_regimes')
    fig,ax = plt.subplots(figsize=(11,4)); ax.plot(t,levels.kospi,label='KOSPI')
    for e in EVENTS:
        ax.axvline(pd.Timestamp(e['date']),ls='--',alpha=.6,label=e['label'])
    ax.legend(); ax.set_title('사건 시점은 원인 입증이 아닌 배경 정보'); save('07_events')
    fig,axes = plt.subplots(4,1,figsize=(11,9),sharex=True)
    for ax,(label,vals) in zip(axes,[('원본',levels.usdkrw), *prediction['decomposition'].items()]):
        ax.plot(t,vals); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].set_title('환율 STL 분해 · 12개월 주기 가정 · 계절성 존재를 보장하지 않음'); save('08_decomposition')
    future = pd.period_range('2026-01',periods=6,freq='M').to_timestamp()
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(t[-24:],levels.usdkrw.iloc[-24:],label='관측')
    for m,vals in prediction['predictions'].items(): ax.plot(future,vals,'--o',label=m)
    for h in [1,3,6]:
        point=prediction['predictions']['naive'][h-1]; lo,hi=prediction['baseline_error_quantiles_10_90'][str(h)]
        ax.vlines(future[h-1],point+lo,point+hi,color='#9b6635',lw=5,alpha=.35)
    ax.set_title('2025-12 기준 과거 예측 실험 · 막대: 과거 오차 10~90백분위'); ax.set_ylabel('원/USD'); ax.legend(); save('09_forecast')
    write_report(payload)
    create_notebook()
    print(json.dumps({'fx_kospi':stats['correlation']['usdkrw']['kospi'], 'regimes':{k:{a:b for a,b in v.items() if a!='values'} for k,v in groups.items()},'forecast_scores':prediction['scores']},ensure_ascii=True,indent=2))


def write_report(p):
    a=p['analysis']; f=p['forecast']; c=a['correlation']; groups=a['regimes']
    text=f'''# 초보 투자자를 위한 거시경제 지표와 KOSPI 관계 분석

## 주제와 질문
금리·환율·KOSPI·국제 금·물가·실업률의 관계를 이해하는 교육용 분석이다. 금리가 변한 달의 주가는 어떻게 움직였는가? 환율과 주가는 함께 움직였는가? 1~3개월 뒤 관계는 다른가? 사건 전후 관계가 달라지는가? 과거 패턴으로 환율을 얼마나 예측할 수 있는가?

## 데이터와 정제
2016-01~2025-12 월별 120개월, 6지표 720행. 한국은행 ECOS와 세계은행 Pink Sheet를 사용했다. 2015-12 선행값을 이용해 첫 달 변화율도 계산했다. CPI는 전년 동월 대비로 계산했다. 중복·누락·비유한값은 오류로 처리하고 임의 보간하지 않았다. 실제 통합 결과 누락은 0개다. 극단값은 삭제하지 않았다. [출처와 항목 코드](docs/DATA_SOURCES.md).

기준금리와 KOSPI는 월말 기준, 환율과 금은 월평균이다. 실업률은 계절조정, 물가는 전년 동월 대비이다. 금리·물가·실업률의 변화는 %p, 가격 변화는 %다. 금 가격 기준이 2025-06에 바뀌었다. 월평균 가격 변화율은 매매 수익률과 다르다.

## 지표 읽는 법
기준금리는 정책금리다. 원/달러 환율 상승은 1달러를 사는 데 더 많은 원화가 필요하다는 뜻이다. KOSPI는 국내 주식시장 지수이며 배당을 포함한 총수익이 아니다. 달러 표시 금 가격은 원화로 투자한 금의 수익과 다르다. 물가 상승률 둔화는 가격 수준 하락을 뜻하지 않는다. 실업률은 노동시장의 일부를 설명한다.

## 분석 결과와 시각화
원본·6개월 이동평균, 가격지수화, 변화율, Pearson 상관, 시차, 국면별 분포를 사용했다. 표본은 120개월이며 시차 k 분석은 120-k개다. 금리 인상/동결/인하는 전월 대비 부호로 구분한다.

![장기 추세와 이동평균](images/01_trends.png)
서로 다른 단위는 별도 축에서 읽는다. 이동평균은 최근 6개 관측의 평균이다.

![가격지표 기준점 비교](images/02_indexed.png)
2016-01을 100으로 맞췄다. 초기 기준점과 달러 표시 여부에 따라 비교 의미가 달라진다.

![월간 변화](images/03_changes.png)
원본 가격 수준과 월간 움직임은 서로 다른 질문에 답한다.

![상관관계](images/04_correlation.png)
0에 가까운 값은 이 표본에서 선형 관계가 약하다는 뜻이며 모든 관계가 없다는 뜻은 아니다.

![시차](images/05_lags.png)
이번 달 금리 변화와 다음 k개월 지표 변화의 관계다. 여러 시차를 탐색했으므로 가장 큰 값도 인과효과나 예측력으로 단정하지 않는다.

![금리 변화별 주가](images/06_regimes.png)
같은 금리 방향에서도 주가 변화의 분포가 넓다. 각 집단 표본 수가 다르다.

![사건 표시](images/07_events.png)
WHO 2020-03-11 팬데믹 선언과 UN 2022-02-24 성명을 근거로 표시했다. 전후 각각 12개월, 사건 달 제외 비교 결과는 대시보드 자료에 포함된다.

## 근거가 있는 인사이트
1. **관찰:** 환율과 KOSPI 월간 변화의 상관은 {c['usdkrw']['kospi']['r']:.3f}(n=120)다. **해석:** 함께 움직이는 방향을 이해하는 단서지만 국제 자금 흐름 등의 원인을 이 분석만으로 입증할 수 없다. **활용:** 원화 약세만으로 개별 기업의 주가를 단정하지 않는다.
2. **관찰:** 금리 인상 달은 {groups['인상']['n']}개월이며 KOSPI 상승 달은 {groups['인상']['positive']}개월, 평균 변화율은 {groups['인상']['mean']:.2f}%다. 인하 달은 {groups['인하']['n']}개월, 평균 {groups['인하']['mean']:.2f}%다. **해석:** 금리 방향 하나가 주가 방향을 결정하지 않는다. 경기와 정책 반응이 겹칠 수 있다. **활용:** 평균뿐 아니라 분포와 표본 수를 확인한다.
3. **관찰:** 금과 KOSPI 변화 상관은 {c['gold_usd']['kospi']['r']:.3f}, 금 기준 변경 전 구간에서는 {a['gold_sensitivity']['r']:.3f}(n={a['gold_sensitivity']['n']})다. **해석:** 기간과 자료 정의에 따라 관계가 달라질 수 있다. **활용:** 금이 항상 주식과 반대로 움직인다고 가정하지 않는다.

## 보너스: 분해와 예측
![STL 분해](images/08_decomposition.png)
12개월 주기를 가정한 분해다. 추세와 계절성은 전체 구간을 사용하므로 실시간 매매에 사용할 수 있는 정보가 아니다. 이를 예측 모델의 학습 입력으로 사용하지 않았다.

![예측](images/09_forecast.png)
초기 60개월에서 시작해 학습 구간을 확장했다. 매번 이후 1·3·6개월을 예측했으며 미래 관측은 학습에 넣지 않았다. 마지막 값 유지(naive)와 단순 지수평활(ses)을 동일한 기준으로 비교한다. 아래 MAE는 원/USD 단위다.

| 모델 | 예측 개월 | 검증 표본 | MAE |
|---|---:|---:|---:|
'''
    text += '\n'.join(f"| {s['model']} | {s['horizon']} | {s['n']} | {s['mae']:.2f} |" for s in f['scores'])
    text += '''

2026-01~06 예측은 2025-12 기준의 역사적 실험이며 현재 전망이 아니다. 막대는 과거 오차의 10~90백분위 범위로 정확한 확률 보장을 하지 않는다. 개정된 자료를 사용해 당시 공개 자료를 완벽히 재현하지 못한다.

## 입문자 체크리스트와 한계
지표의 단위와 기간, 월말/평균, 원화/달러, 표본 수를 확인한다. 함께 변했다는 사실과 원인 가설을 구분한다. 전체 기간과 짧은 기간을 함께 본다. 단일 지표로 매수·매도를 결정하지 않는다.

관측 자료의 상관은 인과관계가 아니다. 자기상관·중첩된 물가 전년비·여러 시차 탐색·작은 국면별 표본 때문에 유의성이나 일반화를 과장하면 안 된다. 미국 금리, 기업 실적, 정책 기대 등 통제하지 않은 변수가 있다. 사건 전후 비교도 다른 충격이 겹친다. 교육용 자료이며 개인별 투자 자문이나 수익 보장이 아니다.

## AI 사용 로그
| 사용 작업 | 이유 | 검증 방법 |
|---|---|---|
| 수집·정제 코드 | 반복 작업 자동화 | 공식 항목명·기간 확인, 누락·중복·변화 계산 테스트 |
| 통계·분해·예측 코드 | 기법 구현과 비교 | 손계산 수열·시차 방향 테스트, 순차 검증 |
| 보고서 초안·화면 | 쉬운 설명과 결과 공유 | 저장된 계산값으로 문장 생성, 그래프·실행 검증 |

## 제출 상태
분석 파일은 로컬 생성 결과다. 공개 배포 URL 및 최종 브라우저 검증 상태는 README와 docs/acceptance.md에 별도로 기록한다.
'''
    (ROOT/'REPORT.md').write_text(text, encoding='utf-8')


def create_notebook():
    nb=nbformat.v4.new_notebook()
    nb.cells=[nbformat.v4.new_markdown_cell('# 경제지표 분석 재현\n프로젝트 가상환경 커널에서 위에서 아래로 실행합니다. 원본 다운로드는 별도 `python -m src.collect_data` 명령으로 수행합니다.'),
              nbformat.v4.new_code_cell("from pathlib import Path\nimport os, sys\nroot = Path.cwd()\nif root.name == 'notebooks': root = root.parent\nos.chdir(root)\nsys.path.insert(0, str(root))\nfrom src.clean_data import build\nbuild()"),
              nbformat.v4.new_code_cell('from src.export import run\nrun()'),
              nbformat.v4.new_code_cell("from IPython.display import display, Image\nfor path in sorted(Path('images').glob('*.png')): display(Image(filename=str(path)))")]
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    (ROOT/'notebooks').mkdir(exist_ok=True)
    nbformat.write(nb,ROOT/'notebooks/analysis.ipynb')


if __name__ == '__main__':
    run()
