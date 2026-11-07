# Exp002: patch PCA, 단어별 attention, 선의 연속성

사용자가 노트북을 내려받아 Colab에서 셀별로 직접 실행하는 분석 작업입니다. exp001의 저장된 run002 활성값을 CPU에서 읽습니다. 모델 가중치 다운로드·새 추론·GPU는 필요하지 않습니다.

- [현재 노트북을 Colab에서 열기](https://colab.research.google.com/github/seungjoolee24/mllm-line-tracing/blob/main/experiments/exp002_patch_analysis/patch_analysis_cases00_04.ipynb)
- [현재 노트북 파일](patch_analysis_cases00_04.ipynb)
- [현재 노트북 다운로드](https://github.com/seungjoolee24/mllm-line-tracing/raw/refs/heads/main/experiments/exp002_patch_analysis/patch_analysis_cases00_04.ipynb)

현재 버전은 Drive 서버에서 확인한 **case00~04 다섯 문제**를 사용합니다. 예전 세 문제용 `patch_analysis.ipynb`는 과거 버전으로 보관하며 현재 실행 대상이 아닙니다. 큰 ZIP 재수집은 하지 않습니다.

## 대상과 표현

| 문제 | 결과 | 원 개수 |
|---|---|---:|
| `case_00_curve` | 성공 | 6 |
| `case_01_straight` | 성공 | 4 |
| `case_02_elbow` | 성공 | 8 |
| `case_03_swirl` | 실패 | 8 |
| `case_04_curve` | 성공 | 8 |

| 분석 | 입력 표현 | 공간 단위 |
|---|---|---|
| PCA 두 조건 | `merger/final/output`, `[400,2560]` | 20×20, 32px |
| 단어별 decoder attention | `[32 heads, selected query rows,400 image keys]`, 36층 | 20×20, 32px |
| 연속성 정량 비교 | `merger/final/input`, `[1600,1024]` | 40×40, 16px |

PCA는 LLM 첫 attention 직전의 표현을 사용합니다. 연속성은 사용자가 선택한 더 세밀한 **patch merge 전 최종 ViT 표현**을 사용합니다. 두 tap은 각각 첫 decoder의 이미지 입력과 최종 ViT block 출력에 대조합니다. ViT의 원래 순서는 2×2 block-major이므로 이미지 대응에는 저장된 token bbox를 사용합니다.

## 실행 순서

1. 노트북을 Colab에서 열고 Python 3 / CPU 런타임을 선택합니다.
2. 셀을 위에서부터 실행해 분석 소스를 준비하고 Drive 접근을 직접 승인합니다.
3. 입력·출력 경로와 고정 분석 설정을 확인합니다.
4. 작은 manifest가 없으면 고정 GitHub 원본을 SHA-256 검증한 뒤 복구합니다. 다섯 case ZIP과 metadata ZIP의 존재·크기를 확인한 뒤 분석 객체를 만듭니다. 기존 manifest가 다르면 덮어쓰지 않습니다.
5. 다섯 문제 복원과 해시 검증 → PCA → attention 추출·선택 보기·전체 저장 → 연속성 분석을 순서대로 실행합니다.
6. 마지막 동기화 셀에서 `flush_and_unmount()` → Drive 재연결 → 결과 해시 재검증을 수행하고 `ANALYSIS COMPLETE / DRIVE SYNC FLUSHED AND RECHECKED`를 확인합니다.

기본 입력:

```text
/content/drive/MyDrive/mllm-line-tracing/experiments/exp001_qwen3vl4b_trace10/run002_artifacts
```

기본 출력:

```text
/content/drive/MyDrive/mllm-line-tracing/experiments/exp002_patch_analysis/run002_cases00_04
```

cache는 `/content/exp002_patch_analysis_cache`에 두며, 선택한 다섯 문제만 복원합니다. 압축 파일·복원본·임시 산출물을 위해 약5GB 여유 공간을 확보합니다. 원래 Drive ZIP은 변경하지 않으며 누락된 작은 manifest만 검증 후 추가합니다. 계산 코드도 결과 폴더에 사본으로 보관합니다.

마운트 경로에서 파일을 다시 읽거나 `fsync`를 호출한 것만으로 서버 업로드 완료를 확정하지 않습니다. 마지막 셀은 [Colab 공식 Drive 코드의 flush 기능](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)을 사용해 대기 중인 쓰기를 마친 뒤 재연결·재검증합니다. 동기화 또는 재검증이 실패하면 런타임을 종료하지 않고 해당 출력을 확인합니다.

노트북은 각 Python 소스를 Git blob SHA로 가져온 뒤 SHA-256을 확인합니다. 고정 데이터셋의 commit과 원시 artifact manifest도 별도로 고정합니다. GitHub 토큰은 필요하지 않습니다. 환경·코드·설정·입력이 같고 결과 파일 해시가 일치할 때만 완료 단계를 재사용합니다. 환경 또는 설정을 바꾸면 새 출력 폴더를 사용합니다.

## 분석 정의

PCA는 **각 이미지와 각 조건에서 독립적으로 fit**합니다. `all`은400개 전체이고 `line`은 원본 path mask와 한 픽셀이라도 겹치는 patch입니다. 이 마스크에는 색 원과 S/E 아래의 경로도 포함됩니다. 각 성분의 fit-token 최솟값·최댓값으로 RGB를0~1 변환하고 선택 수·설명 분산·축·평균·원본 PCA 점수를 저장합니다. 서로 다른 fit의 RGB는 직접 대응하지 않습니다.

Attention은 실제 생성 색 단어의 첫 BPE를 예측한 위치를 사용합니다. 단어 자신의 입력 위치는 다음 단어 예측 위치이므로 사용하지 않습니다. 기본 그림은32개 head 평균입니다. 원래 full-sequence softmax의 이미지 열과, head 평균 후 이미지 mass로 나눈 조건부 분포를 구분합니다. 선택한 다섯 사례·단어·층에 공통 색상 범위를 적용합니다. 한 단어의36층 그림을6×6으로 배치하며 각 그림은20×20 patch 지도입니다. 개별 head는 화면에서 선택해서 볼 수 있습니다.

연속성은 **원본1024차원**을 사용합니다. 색 원·S/E 원판에2px 여유를 더해 조금이라도 겹치는16px cell, 한 cell에 분리된 여러 arc 구간이 들어가는 경우, 선 비율1% 미만을 제외합니다. 단일 arc 구간의 중점을 patch의 선상 위치로 사용합니다. 원시 cosine과 해당 이미지1,600개 토큰 평균을 뺀 cosine을 함께 저장합니다. 나선의 geometry matching 기준은 다음과 같습니다.

| 기준 | 값 |
|---|---:|
| 선상 가까운 후보 | `0 < 거리 ≤ 64px` |
| 선상 먼 후보 | `거리 ≥ 128px` |
| 두 후보의 화면상 거리 차이 | `≤ 8px` |
| 두 후보의 선 비율 차이 | `≤ 0.03` |
| 집계에 필요한 서로 다른 anchor | `≥ 5` |

한 anchor당 geometry만으로 최선의 near/far 쌍을 선택합니다. 활성값·유사도는 매칭에 사용하지 않습니다. 대응된 `cos(anchor,near)−cos(anchor,far)`의 개별 값과 중앙값을 기록합니다. 부족한 수·제외 사유·token 재사용을 함께 기록하며 기준을 자동으로 완화하지 않습니다. 직선은 선상 거리와 화면상 거리를 분리할 수 없는 구조적 대조 사례로 표시합니다.

## 결과와 해석 범위

문제별 `pca/`, `attention/`, `continuity/` 아래에 PNG·NPZ·JSON·CSV를 저장합니다. 기본 결과는 PCA 10개, head 평균 attention36층 패널68개입니다. root의 `input_manifest.json`, `analysis_config.json`, `environment.json`, `summary.json`, `output_manifest.json`에 입력·코드·환경·설정·결과 파일의 검증 정보를 보관합니다.

이 작업은 다섯 사례의 기술적 분석입니다. 성공한 나선은 이번 범위에 없으며, 성공4개·실패1개는 난도 일치 대조군이 아닙니다. 유효 patch나 매칭이 부족하면 상태를 그대로 보고합니다. Attention 자체로 value·output projection·residual·MLP를 통한 최종 hidden state의 인과적 기여를 분해하지 않습니다. p값이나 일반화·실패 원인 판정을 생성하지 않습니다.

실패 나선의 입력 기하 점검에서는 merge 전 clean patch가7개였고 고정한 거리 조건의 매칭 anchor가0개였습니다. 이 경우 조건부 중앙값은 `null`로 보고하며, 유사도 행렬과 거리 그래프는 생성합니다. 노트북은 모든 선택 사례의 수와 제외 사유를 다시 계산합니다.

## 코드 검증

```bash
python -m pip install -r experiments/exp002_patch_analysis/requirements-dev.txt
python -m unittest discover -s experiments/exp002_patch_analysis -p 'test_*.py' -v
```

검사는 합성 수치·기하 입력과 임시 파일로 수행합니다. 실제 다섯 사례의 활성값 분석과 Colab 실행은 사용자가 노트북에서 수행합니다.
