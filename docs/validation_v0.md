# 생성기 v0 검증 기록

2026년 11월 7일 Python 3.12.14, Pillow 11.3.0 환경에서 검증했다.

## 자동 검증

`python -m unittest discover -s tests -v`의 14개 테스트가 통과했다. 기하·정답 순서, 방향 대조쌍, 명시적 길이·호 길이 간격, jitter 범위·재현성, 끝 구간 여백 상한, 불가능한 설정의 실패, 이미지 색과 마스크 연결성, 저장 파일 손상 탐지, 기존 출력 경로 보호를 검사했다.

master seed 20261107로 아래 두 데이터셋을 생성하고 저장 PNG와 JSONL을 다시 읽어 audit했다. 각 데이터셋은 네 family별 25개로 구성된다.

| 조건 | 개수 | 길이 | 점 개수 | 점 간격 | 위치 jitter | 결과 |
| --- | --- | --- | --- | --- | --- | --- |
| 기본 등간격 | 100 | 480px | 8 | 자동 등분 | 0px | 통과 |
| 불규칙 간격 | 100 | 480px | 8 | 56px | ±4px | 통과 |

총 200개 기하, 600개 PNG의 해시·모드·크기, 1,600개 오브젝트 중심의 색·마스크 ID, 200개 경로 마스크의 단일 연결 성분을 확인했다. 실측 경로 길이는 부동소수점 오차 범위에서 480px였고, 두 조건 모두 끝 구간 여백 제한을 만족했다.

저장소의 기본 예제 4개와 jitter ±8px 예제 4개도 audit을 통과했다. 네 family의 contact sheet에서 S/E, 점 겹침, 선의 연속성, 정답 캡션을 직접 확인했다.

Python 3.9 문법 검사와 Colab notebook 코드 cell의 Python 문법 검사도 통과했다. 실제 Colab 런타임 실행이나 VLM 정확도 평가는 이 검증에 포함하지 않았다.

## 재검증

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m line_tracing.audit examples/easy_v0
python -m line_tracing.audit examples/irregular_v0
```
