# 경제를 읽는 첫걸음

초보 투자자를 위한 2016~2025년 경제지표·KOSPI 관계 분석. Python 분석과 순수 HTML/CSS/JavaScript 대시보드로 구성한다.

- 분석 리포트: [REPORT.md](REPORT.md)
- 재현 노트북: [notebooks/analysis.ipynb](notebooks/analysis.ipynb)
- 출처: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)
- 검증/제출 상태: [docs/acceptance.md](docs/acceptance.md)
- 보너스 제출용 화면과 시연 시나리오: [docs/DASHBOARD_DEMO.md](docs/DASHBOARD_DEMO.md)
- 공개 배포 URL: 아직 없음. 로컬 대시보드 구현 상태.
- 후속 AI 서비스 저장소: https://github.com/leeds86/m1-2

## 실행

Python 3.10 이상을 설치한 뒤 프로젝트 루트에서 실행한다. 아래는 Windows PowerShell 예시다.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m http.server 8765 --bind 127.0.0.1 --directory web
```

브라우저에서 http://127.0.0.1:8765 를 연다. 웹 화면에 API 키는 필요하지 않다.

## 원본 수집 및 분석 재생성

`.env.example`을 `.env`로 복사하고 한국은행 ECOS 키를 로컬에 입력한다. 키는 커밋하거나 프론트엔드에 넣지 않는다.

```powershell
.venv/Scripts/python.exe -m src.collect_data
.venv/Scripts/python.exe -m src.clean_data
.venv/Scripts/python.exe -m src.export
```

수집 원본과 SHA256은 data/raw에 남는다. 통합 CSV는 data/processed, 공개 화면용 결과는 web/data, 보고서 그래프는 images에 생성된다. export는 REPORT.md와 실행 전 노트북도 재생성하므로 수작업 편집은 생성 코드에 반영해야 한다. 노트북은 VS Code의 Python 커널을 `.venv`로 선택해 위에서 아래로 실행할 수 있다.

## Vercel 설정

새 프로젝트에 이 GitHub 저장소를 연결한다. Framework Preset은 Other, Root Directory는 저장소 루트, Output Directory는 web이다. Python 분석을 Vercel에서 실행하지 않는다. 분석 결과를 로컬에서 생성한 뒤 배포한다. 공개 폴더에는 HTML/CSS/JS와 정제 결과만 포함한다. ECOS_API_KEY를 Vercel 환경 변수로 등록할 필요가 없다.

## 화면에서 확인할 것

기간·지표·원본/변화율·금리 변동 달 필터를 변경하면 통계와 상관행렬이 바뀐다. 제외된 달은 그래프에서 연결하지 않는다. 분해·예측·시차·사건 분석은 전체 기간 고정이며 화면에 명시한다. CSV 다운로드는 선택된 기간과 금리 조건에 맞는 6개 지표의 원본 값을 제공한다.

## 해석 범위

상관은 인과나 매매 신호가 아니다. 가격 지표는 변화율, 금리·물가상승률·실업률은 퍼센트포인트 차이를 쓴다. 금 가격은 2025-06 정의가 바뀌었다. 예측은 2025-12 기준 과거 실험이다. 공개 원본의 개정과 월평균/월말 집계 차이가 있다.

## 데이터 이용 범위

무료·비상업적 교육용 프로젝트다. ECOS 제공 사실과 World Bank 금 자료의 CC BY 4.0 라이선스, 수집일 및 계산/가공 내용을 표시한다. 원본 제공기관은 이 분석을 보증하지 않는다. 유료화·광고 등 상업적 전환 시 ECOS 타 기관 자료의 승인을 별도로 확인해야 한다. 자세한 근거는 DATA_SOURCES.md에 기록했다.
