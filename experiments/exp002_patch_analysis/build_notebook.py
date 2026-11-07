# -*- coding: utf-8 -*-
"""Build the reviewable user-run notebook after all source files are final."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "experiments/exp002_patch_analysis"
paths = [
    "experiments/exp002_patch_analysis/pipeline.py",
    "experiments/exp002_patch_analysis/views.py",
    "experiments/exp002_patch_analysis/continuity.py",
    "experiments/exp001_qwen3vl4b_trace10/analysis.py",
]
sources = {}
for relative in paths:
    data = (REPO / relative).read_bytes()
    sources[relative] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }

cells = []
def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip().splitlines(keepends=True)})

md("""# Exp002: case00~04 이미지 patch 표현·attention·연속성 분석

**이 노트북은 사용자가 위에서 아래로 셀별로 직접 실행합니다.** 저장된 활성값을 CPU에서 읽으며, 모델 가중치와 GPU가 필요하지 않습니다. Colab 런타임은 **Python 3 / CPU**로 선택해도 됩니다.

| 문제 | 결과 | 색 원 |
|---|---|---:|
| case_00_curve | 성공 | 6 |
| case_01_straight | 성공 | 4 |
| case_02_elbow | 성공 | 8 |
| case_03_swirl | 실패 | 8 |
| case_04_curve | 성공 | 8 |

분석은 승인한 세 가지입니다.

1. 이미지별 독립 PCA→RGB: 전체400 patch / 선 포함 patch 두 조건.
2. 실제 생성 색 단어의 첫 BPE **예측 위치**→이미지400 patch attention, decoder36층.
3. 비전 인코더 최종 출력·**patch merge 이전의1,600개 × 1,024차원**에서 선 연속성과 표현 유사도 비교.

입력은 exp001 run002의 Drive 백업 중 서버에서 확인한case00~04입니다. 원시 ZIP은 읽기만 합니다. 누락된 작은manifest만 GitHub 원본을 해시 검증해 복구하고, 결과는 새 exp002 Drive 폴더에 저장합니다. 실제 답을 causal replay하며 수집한 활성값을 분석합니다.
""")

md("""## 1. CPU 환경과 분석 소스 준비

가벼운 시각화 의존성을 확인·설치합니다. Colab에 이미 설치된 PyTorch는 CPU 파일 읽기에 사용합니다. 아래 소스 목록은 **내용 주소인 Git blob SHA와 SHA-256으로 고정**되어, GitHub main이 바뀌어도 이 노트북의 코드는 바뀌지 않습니다. GitHub 토큰은 필요하지 않습니다.
""")
code("""
import sys, subprocess, importlib.util
if importlib.util.find_spec('torch') is None:
    raise RuntimeError('PyTorch가 필요합니다. Colab 표준 Python 런타임을 사용하세요.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'numpy>=1.26,<3', 'matplotlib>=3.8,<4',
                'Pillow>=10,<13', 'ipywidgets>=8,<9'], check=True)
import numpy as np
import matplotlib
import torch
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__, '/ Matplotlib:', matplotlib.__version__, '/ Torch:', torch.__version__)
print('계산 장치: CPU / 모델 추론 없음')
""")
bootstrap = """
from pathlib import Path
import base64, hashlib, json, urllib.request

SOURCE_FILES = __SOURCE_FILES__
bundle_id = hashlib.sha256(json.dumps(SOURCE_FILES, sort_keys=True).encode()).hexdigest()[:12]
SOURCE_ROOT = Path('/content') / ('exp002_source_' + bundle_id)
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

for relative, record in SOURCE_FILES.items():
    target = SOURCE_ROOT / relative
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError('기존 분석 소스가 다릅니다: ' + str(target))
    else:
        url = 'https://api.github.com/repos/seungjoolee24/mllm-line-tracing/git/blobs/' + record['git_blob_sha']
        request = urllib.request.Request(url, headers={'User-Agent': 'mllm-exp002-notebook'})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        assert payload['encoding'] == 'base64' and payload['sha'] == record['git_blob_sha']
        data = base64.b64decode(payload['content'])
        assert hashlib.sha256(data).hexdigest() == record['sha256'], '소스 SHA-256 불일치'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print('SOURCE VERIFIED:', relative)

TASK = SOURCE_ROOT / 'experiments/exp002_patch_analysis'
stale_modules = []
for module_name in ('pipeline', 'views', 'continuity'):
    loaded = sys.modules.get(module_name)
    if loaded is not None and (not getattr(loaded, '__file__', None)
                               or Path(loaded.__file__).resolve().parent != TASK.resolve()):
        if not getattr(loaded, '__file__', None) or Path(loaded.__file__).resolve().parent.name != 'exp002_patch_analysis':
            raise RuntimeError('다른 프로젝트 모듈과 이름이 겹칩니다. 런타임을 다시 시작하세요: ' + module_name)
        stale_modules.append(module_name)
if stale_modules:
    for module_name in ('pipeline', 'views', 'continuity'):
        sys.modules.pop(module_name, None)
    print('이전 exp002 소스의 import cache를 해제했습니다. 아래 설정 셀에서 새 run을 만듭니다.')
sys.path.insert(0, str(TASK))
import importlib
importlib.invalidate_caches()
from pipeline import AnalysisRun, DEFAULT_CONFIG, CASES, input_preflight, verify_output_bundle
print('분석 소스 준비 완료:', TASK)
""".replace("__SOURCE_FILES__", repr(sources))
code(bootstrap)

md("""## 2. Google Drive 연결과 경로·분석 설정

Google 계정·Drive 접근 승인은 직접 완료하세요. `ARTIFACT_DIR`에는 메타데이터 ZIP과case00~04 ZIP이 있어야 합니다. 작은 `artifact_manifest.json`이 없으면 고정한GitHub 원본을 SHA-256으로 검증해 복구합니다. 이미 있는 manifest의 해시가 다르면 덮어쓰지 않고 중단합니다. case05~09는 요구하지 않습니다.

이 셀은 선택한 ZIP들의 존재·크기를 먼저 확인한 뒤 `AnalysisRun`을 만듭니다. 원시 ZIP의 SHA-256은 다음 복원 단계에서 검사합니다. `OUTPUT_DIR`은 이전 버전과 충돌하지 않는 새 폴더입니다.

같은 코드·설정·입력·계산환경의 완료 단계는 파일 해시를 확인하고 재사용합니다. 설정·코드·패키지 버전 등이 다르면 기존 결과에 섞지 않으므로, 그 경우 `run002` 등 **새 출력 폴더**를 지정하세요. `/content`의 cache는 임시 공간입니다.
""")
code("""
from google.colab import drive
drive.mount('/content/drive')

ARTIFACT_DIR = Path('/content/drive/MyDrive/mllm-line-tracing/experiments/exp001_qwen3vl4b_trace10/run002_artifacts')
OUTPUT_DIR = Path('/content/drive/MyDrive/mllm-line-tracing/experiments/exp002_patch_analysis/run002_cases00_04')
CACHE_DIR = Path('/content/exp002_patch_analysis_cache')

CONFIG = json.loads(json.dumps(DEFAULT_CONFIG))
# 승인한 연속성 비교 기준입니다. 변경 시 OUTPUT_DIR도 새로 지정하세요.
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
print('입력 Drive:', ARTIFACT_DIR)
print('출력 Drive:', OUTPUT_DIR)
print('임시 cache:', CACHE_DIR)
preflight = input_preflight(ARTIFACT_DIR, recover_manifest=True)
for entry in preflight['required_archives']:
    print('PREFLIGHT:', entry['file'], entry['bytes'], 'bytes')
run = AnalysisRun(SOURCE_ROOT, ARTIFACT_DIR, OUTPUT_DIR, CACHE_DIR, CONFIG)
""")

md("""## 3. 다섯 문제 복원과 입력 검증

문제5개의 ZIP과 메타데이터 ZIP을 복원합니다. 원래 ZIP·읽는 텐서·이미지·마스크의 해시와 shape/dtype을 확인합니다. 압축 파일과 복원본·임시 산출물을 위해 로컬cache에 약5GB의 여유 공간을 확보하세요.

각 문제에서 PCA용 `merger/final/output [400,2560]`과 첫 decoder 이미지 입력의 일치를 확인합니다. 연속성용 `merger/final/input [1600,1024]`은 최종 ViT block 출력과 대조합니다. 아래 표가 나오기 전에는 다음 셀로 진행하지 마세요.
""")
code("""
from IPython.display import display, Markdown, Image as DisplayImage
overview = run.prepare()
for row in overview:
    print(row)
display(Markdown('**INPUT VERIFIED: 다섯 문제와 표현 확인 완료**'))
""")

md("""## 4. 이미지별 PCA→RGB 두 조건

각 문제에서 두 PCA를 **별도로 fit**합니다. 첫 조건은400개 전체, 두 번째는 원본 path mask와 한 픽셀이라도 겹치는 patch만 사용합니다. path mask는 색 원과 S/E 아래의 경로도 포함합니다. 배경만 있는 patch는 두 번째 PCA 계산에서 제외됩니다.

세 그림은 원본 / PCA RGB / PCA에 사용한 patch입니다. RGB는 각 fit 안에서 성분별 min–max로0~1 변환합니다. **색과 축은 이미지·조건 사이에서 직접 대응하지 않습니다.** fit token 수와 설명 분산 비율도 함께 기록합니다.
""")
code("""
pca_results = run.pca()
for case_id in CASES:
    display(Markdown('### ' + case_id))
    for condition, item in pca_results[case_id]['conditions'].items():
        print(condition, item)
        if item.get('png'):
            display(DisplayImage(filename=str(item['png']), width=1150))
""")

md("""## 5. 생성 색 단어의 predictor→이미지 attention 추출

`red` 자체가 입력된 위치가 아니라 **red의 첫 BPE를 예측한 위치**를 선택합니다. 실제 생성된 색 단어를 사용하고 `end`·종료 토큰은 제외합니다. 실패 사례의 실제 출력과 정답을 함께 표시합니다.

각 층에서32개 head를 평균한 원래 attention, 이미지 총 attention mass, head 평균 후 이미지 내부로 정규화한 분포를 저장합니다. 이미지 mass가0이면 조건부 분포는 정의되지 않음으로 표시합니다.
""")
code("""
attention_results = run.attention()
for case_id, item in attention_results['cases'].items():
    print(case_id, 'words:', len(run.word_options(case_id)),
          'raw max:', item['raw_vmax'], 'conditional max:', item['conditional_vmax'])
    print('선택 가능한 단어:', run.word_options(case_id))
print('36층·다섯 사례 공통 색상 범위:', attention_results['raw_vmax'], attention_results['conditional_vmax'])
""")

md("""## 6. 선택한 단어의36층 지도 확인

아래 설정을 바꾸고 이 셀만 다시 실행하세요. `WORD_INDEX`는 출력 순서 기준0부터, `HEAD=None`은32개 head 평균입니다. 개별 head는0~31을 지정합니다.

**6×6은36개 층별 그림의 배치**입니다. 그림 하나 안에는 이미지400 patch의20×20 heatmap이 있습니다. head 평균 그림은 다섯 문제·모든 단어·36층에 동일한 색상 범위를 사용합니다. 개별 head는 해당 단어의36층 범위를 따로 표시합니다. 화면 확인은 Drive 결과 파일을 추가하지 않습니다.
""")
code("""
import matplotlib.pyplot as plt
SELECTED_CASE = 'case_03_swirl'
WORD_INDEX = 0
MODE = 'raw'                   # 'raw' 또는 'conditional'
HEAD = None                    # None 또는 0~31

print(run.word_options(SELECTED_CASE))
fig = run.show_attention(SELECTED_CASE, WORD_INDEX, MODE, HEAD)
display(fig)
plt.close(fig)
""")

md("""선택 사항: 다음 셀은 같은 지도를 dropdown과 표시 버튼으로 선택합니다. 화면에 widget이 표시되지 않아도 바로 위의 설정 셀을 사용할 수 있습니다.
""")
code("""
import ipywidgets as widgets
case_choice = widgets.Dropdown(options=list(CASES), description='문제')
word_choice = widgets.Dropdown(options=run.word_options(CASES[0]), description='단어')
mode_choice = widgets.Dropdown(options=[('원래 값', 'raw'), ('이미지 내부 분포', 'conditional')], description='표시')
head_choice = widgets.Dropdown(options=[('32 head 평균', None)] + [(str(i), i) for i in range(32)], description='head')
draw_button = widgets.Button(description='36층 표시')
viewer = widgets.Output()

def update_words(change):
    word_choice.options = run.word_options(change['new'])
case_choice.observe(update_words, names='value')

def draw_selected(_):
    with viewer:
        viewer.clear_output(wait=True)
        fig = run.show_attention(case_choice.value, word_choice.value, mode_choice.value, head_choice.value)
        display(fig)
        plt.close(fig)
draw_button.on_click(draw_selected)
display(widgets.VBox([case_choice, word_choice, mode_choice, head_choice, draw_button, viewer]))
""")

md("""## 7. 모든 생성 색 단어의 attention 지도 저장

34개 색 단어 × 원래 값/조건부 분포2종 = **36층 패널 PNG68개**를 Drive에 저장합니다. 개별 head 지도는 기본 일괄 저장 대상에 포함하지 않습니다. 이 셀은 그림 저장 때문에 앞의 추출 셀보다 오래 걸릴 수 있으며, 단계별 저장·검증 상태를 출력합니다.
""")
code("""
saved_attention = run.render_all_attention()
print('SAVED:', sum(len(items) for items in saved_attention.values()), 'attention grids')
""")

md("""## 8. Merge 전 원본1024차원 표현과 선의 연속성

이 분석은 최종 ViT 출력과 같은 `merger/final/input [1600,1024]`을 사용합니다. 각 토큰이16×16픽셀을 담당합니다. PCA와 decoder attention은 기존400개 토큰을 그대로 사용합니다.

PCA projection을 사용하지 않습니다. 16×16 선 포함 patch에서 색 원·S/E와 겹친 patch(2px 여유), 떨어진 여러 선 구간이 섞인 patch, 선 비율1% 미만 patch를 제외합니다. 원래ViT 토큰 순서는2×2 block-major이므로 단순40×40 reshape를 사용하지 않고 저장된 bbox로 위치를 연결합니다. 남은 patch를 선의 S→E 호 길이 순서로 정렬해 유사도 행렬과 거리 그래프를 만듭니다.

원시 cosine과, 해당 이미지1,600토큰 평균을 뺀 뒤 계산한 cosine을 함께 기록합니다. 나선에서는 선상 가까운 후보≤64px / 먼 후보≥128px를 화면상 거리 차이≤8px·선 비율 차이≤3%p로 매칭합니다. 매칭에는 활성값을 사용하지 않습니다. 최소5anchor 미달이면 집계 불가를 표시하고 기준을 완화하지 않습니다.

직선에서는 선상 거리와 화면상 거리가 같아 독립 효과를 분리할 수 없습니다. 이번 다섯 문제에는 성공한 나선이 없으며, 성공4개와 실패1개는 난도 일치 대조군이 아닙니다. 결과는 사례의 기술적 관찰이며 p값·인과적 실패 원인·모집단 일반화 결론을 만들지 않습니다.
""")
code("""
continuity_results = run.continuity()
for case_id, result in continuity_results.items():
    display(Markdown('### ' + case_id))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    for plot in sorted((OUTPUT_DIR / case_id / 'continuity').glob('*.png')):
        display(DisplayImage(filename=str(plot), width=1000))
""")

md("""## 9. Drive 결과 전체 검증과 완료 확인

모든 단계 receipt의 코드·설정·파일 해시를 재확인한 뒤 summary와 output manifest를 저장합니다. source 사본, 입력 주석·이미지·마스크, 계산 배열, 그림, CSV, 환경 버전을 함께 보관합니다.

이 셀은 마운트된 파일의 계산·저장·해시 검증까지만 수행합니다. **서버 업로드 완료와는 구분합니다. 마지막10번 동기화 셀까지 실행한 뒤 런타임을 종료하세요.** 같은 실행을 재개하려면 위에서부터 같은 설정으로 실행하면 됩니다.

Attention 지도는 각 층에서 이미지로 향하는 attention 분포입니다. value·출력 projection·residual·MLP를 포함한 최종 단어 hidden state의 전체 인과적 기여 분해는 이 실험 범위에 포함하지 않습니다.
""")
code("""
completion = run.finalize()
print(json.dumps(completion, indent=2, ensure_ascii=False))
print('결과 manifest:', OUTPUT_DIR / 'output_manifest.json')
print('요약:', OUTPUT_DIR / 'summary.json')
""")

md("""## 10. Drive 업로드 flush와 재연결 검증

`fsync` 또는 마운트 경로에서의 읽기만으로 Google Drive 서버 반영을 확정하지 않습니다. 이 셀은 Colab의 `flush_and_unmount()`로 대기 중인 Drive 업로드를 마친 뒤 다시 연결하고, 결과 파일을 다시 읽어 크기·해시를 확인합니다. 파일이 많으면 기다려야 합니다. 재연결 과정에서 계정 승인이 나오면 직접 완료하세요.

**`ANALYSIS COMPLETE / DRIVE SYNC FLUSHED AND RECHECKED`를 확인한 뒤 런타임을 종료하세요.** flush나 재연결·검증 오류가 발생하면 런타임을 유지하고 그 출력을 확인하세요.
""")
code("""
print('Drive 업로드 동기화를 기다립니다. 완료 전에는 런타임을 종료하지 마세요.', flush=True)
import os
assert os.path.ismount('/content/drive'), 'Drive가 실제로 마운트되어 있는지 확인하세요.'
drive.flush_and_unmount(timeout_ms=600_000)
print('DRIVE SYNC FLUSH COMPLETE', flush=True)
drive.mount('/content/drive')
synced = verify_output_bundle(OUTPUT_DIR)
print(json.dumps(synced, indent=2, ensure_ascii=False))
print('ANALYSIS COMPLETE / DRIVE SYNC FLUSHED AND RECHECKED')
""")

for i, cell in enumerate(cells):
    cell["id"] = f"exp002-{i:02d}"
notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "exp002_patch_analysis_cases00_04.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
(TASK / "patch_analysis_cases00_04.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Built {len(cells)} cells, {len(sources)} content-addressed source files")
