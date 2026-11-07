# Exp002: patch PCA, 단어별 attention, 선의 연속성

사용자가 노트북을 내려받아 Colab에서 셀별로 직접 실행하는 분석 작업입니다. exp001의 저장된 run002 활성값을 CPU에서 읽습니다. 모델 가중치 다운로드·새 추론·GPU는 필요하지 않습니다.

- [Colab에서 열기](https://colab.research.google.com/github/seungjoolee24/mllm-line-tracing/blob/main/experiments/exp002_patch_analysis/patch_analysis.ipynb)
- [노트북 파일](patch_analysis.ipynb)
- [노트북 다운로드](https://github.com/seungjoolee24/mllm-line-tracing/raw/refs/heads/main/experiments/exp002_patch_analysis/patch_analysis.ipynb)

## 대상과 표현

| 문제 | 결과 | 원 개수 |
|---|---|---:|
| `case_03_swirl` | 실패 | 8 |
| `case_07_swirl` | 성공 | 6 |
| `case_05_straight` | 성공 | 8 |

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
4. 세 문제 복원과 입력 검증 → PCA → attention 추출·선택 보기·전체 저장 → 연속성 분석을 순서대로 실행합니다.
5. 마지막 셀에서 `ANALYSIS COMPLETE / DRIVE VERIFIED`를 확인합니다.

기본 입력:

```text
/content/drive/MyDrive/mllm-line-tracing/experiments/exp001_qwen3vl4b_trace10/run002_artifacts
```

기본 출력:

```text
/content/drive/MyDrive/mllm-line-tracing/experiments/exp002_patch_analysis/run001
```

cache는 `/content/exp002_patch_analysis_cache`에 두며, 선택한 세 문제만 복원합니다. 압축 파일과 복원본을 합쳐 약2.5GB가 필요합니다. 원래 Drive 백업은 변경하지 않습니다. 계산 코드도 결과 폴더에 사본으로 보관합니다.

노트북은 각 Python 소스를 Git blob SHA로 가져온 뒤 SHA-256을 확인합니다. 고정 데이터셋의 commit과 원시 artifact manifest도 별도로 고정합니다. GitHub 토큰은 필요하지 않습니다. 환경·코드·설정·입력이 같고 결과 파일 해시가 일치할 때만 완료 단계를 재사용합니다. 환경 또는 설정을 바꾸면 새 출력 폴더를 사용합니다.

## 분석 정의

PCA는 **각 이미지와 각 조건에서 독립적으로 fit**합니다. `all`은400개 전체이고 `line`은 원본 path mask와 한 픽셀이라도 겹치는 patch입니다. 이 마스크에는 색 원과 S/E 아래의 경로도 포함됩니다. 각 성분의 fit-token 최솟값·최댓값으로 RGB를0~1 변환하고 선택 수·설명 분산·축·평균·원본 PCA 점수를 저장합니다. 서로 다른 fit의 RGB는 직접 대응하지 않습니다.

Attention은 실제 생성 색 단어의 첫 BPE를 예측한 위치를 사용합니다. 단어 자신의 입력 위치는 다음 단어 예측 위치이므로 사용하지 않습니다. 기본 그림은32개 head 평균입니다. 원래 full-sequence softmax의 이미지 열과, head 평균 후 이미지 mass로 나눈 조건부 분포를 구분합니다. 선택한 세 사례·단어·층에 공통 색상 범위를 적용합니다. 한 단어의36층 그림을6×6으로 배치하며 각 그림은20×20 patch 지도입니다. 개별 head는 화면에서 선택해서 볼 수 있습니다.

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

문제별 `pca/`, `attention/`, `continuity/` 아래에 PNG·NPZ·JSON·CSV를 저장합니다. 기본 결과는 PCA 6개, head 평균 attention36층 패널44개입니다. root의 `input_manifest.json`, `analysis_config.json`, `environment.json`, `summary.json`, `output_manifest.json`에 입력·코드·환경·설정·결과 파일의 검증 정보를 보관합니다.

이 작업은 세 사례의 기술적 분석입니다. 두 나선은 원 개수와 형태가 달라 난도 일치 대조군이 아닙니다. 유효 patch나 매칭이 부족하면 상태를 그대로 보고합니다. Attention 자체로 value·output projection·residual·MLP를 통한 최종 hidden state의 인과적 기여를 분해하지 않습니다. p값이나 일반화·실패 원인 판정을 생성하지 않습니다.

입력 기하만 점검한 결과, merge 전 clean patch는 실패 나선7개·성공 나선13개·직선4개입니다. 이들로 유사도 행렬과 거리 그래프를 만듭니다. 고정한 near/far·화면 거리·선 비율 조건의 매칭 anchor는0개이므로 두 나선의 조건부 중앙값은 `null`로 보고됩니다. 이는 모델 활성값의 결과가 아니라 geometry 기준의 실행 가능성 점검이며, 노트북에서도 실제 수와 제외 사유를 다시 계산합니다.

## 코드 검증

```bash
python -m pip install -r experiments/exp002_patch_analysis/requirements-dev.txt
python -m unittest discover -s experiments/exp002_patch_analysis -p 'test_*.py' -v
```

검사는 합성 수치·기하 입력과 임시 파일로 수행합니다. 실제 세 사례의 활성값 분석과 Colab 실행은 사용자가 노트북에서 수행합니다.
