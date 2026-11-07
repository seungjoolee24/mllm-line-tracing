# 단일 경로 오브젝트 추적 생성기 v0

이 생성기는 MLLM이 S에서 출발해 E까지 같은 선을 따라갈 때 만나는 색 원의 순서를 복원하는 기초 데이터를 만든다. 후속 연구의 초점은 내부 표현 분석이며 과제 자체의 신규성을 전제하지 않는다.

## 입력과 정답

모델에는 깨끗한 PNG 한 장과 `prompt`만 전달한다. 답은 공백으로 구분한 소문자 색 이름 뒤에 `end`를 붙인다. 기본 색은 red, orange, yellow, green, blue, purple, pink, brown이고 한 이미지 안에서 중복되지 않는다. 오브젝트 수는 2–8개로 설정한다. S와 E는 시작·끝 표지이며 색 순서에 포함하지 않는다.

초기 기본값은 640×640, 선 두께 4px, 원 반지름 12px, 경로 길이 480px다. 경로 길이를 생략하면 이미지 한 변의 75%로 결정된다. 전체 회전과 추적 방향은 무작위화하며 명시적으로 고정할 수도 있다. 색의 공간 배치는 추적 방향을 뒤집어도 유지되어, 동일 seed의 방향 대조쌍을 만들 수 있다.

`straight`는 직선, `curve`는 완만한 곡선, `elbow`는 한 번 꺾이는 선, `swirl`은 느슨한 나선이다. `mixed`는 이 네 family를 순서대로 순환한다. 나선은 기본 0.75회전이다. family 간 같은 경로 길이를 사용해도 전체 복잡도가 같다는 뜻은 아니다. 나선 밀도 등의 효과를 분석할 때는 실제 측정값으로 조건을 맞춰야 한다.

## 생성과 검증

기하를 먼저 생성하고 선을 촘촘한 polyline으로 저장한다. 원은 호 길이를 기준으로 등간격 배치한다. `--object-spacing`을 지정하면 점 중심 사이의 경로 위 거리를 해당 px로 고정하고 남은 길이는 양 끝에 동일하게 둔다. 생략하면 양 끝 여백을 제외한 구간에 원을 고르게 분포시킨다. 양끝의 빈 경로는 기본 점 간격의 1.25배 이내로 제한한다. 이 한계를 넘는 길이·간격 조합은 점 개수·간격 또는 길이를 바꾸도록 오류를 낸다. `--position-jitter`의 기본값은 0이며, 지정 시 각 원의 호 길이 위치에 최대 해당 px의 seeded 변위를 준다. 경로 위 간격은 곡선에서의 유클리드 직선거리와 다르므로 두 거리 모두 주석으로 확인한다.

프레임 밖의 경로, 선의 교차·자기 접촉, 원의 겹침, 원과 S/E 표지의 간섭, 원이 다른 구간을 덮는 모호한 배치를 거부한다. 요청한 길이·간격·원 크기가 들어가지 않으면 조건을 조용히 바꾸지 않고 명시적으로 실패한다.

4배 해상도로 그린 후 축소해 안티앨리어싱한다. 폰트는 Pillow에 내장된 기본 폰트를 사용하고 Pillow 버전을 고정한다. Python·Pillow 버전, 생성 config와 seed를 manifest에 남긴다. 동일 버전·config·seed에서 동일 주석과 PNG가 재현된다. sample seed는 master seed와 index의 SHA256에서 파생되어 총 생성 개수와 독립적이다.

`validate_sample`은 벡터 기하와 정답·순서 정합성을 검사한다. `python -m line_tracing.audit DATASET`은 저장된 파일을 다시 읽어 기하, 파일 해시, 이미지 모드·크기, 원 중심 픽셀의 색, 마스크 ID와 경로의 단일 연결 성분을 검사한다. unit test는 방향 대조쌍, seed 재현성, 부적합 옵션, 출력 충돌과 실패 시 정리를 검사한다.

## 주석 규약

- 좌표 원점은 좌상단이며 x는 오른쪽, y는 아래쪽으로 증가한다. 단위는 최종 이미지의 px다.
- `center_normalized`는 좌표를 이미지 크기로 나눈 값이다. 오브젝트의 bbox는 연속 좌표의 원 외접 사각형이다.
- `sequence_index`는 S에서 E 방향의 0 기반 순서다. `objects` 배열 저장 순서는 섞여 있으므로 순서를 읽을 때 이 필드를 사용한다.
- 접선은 S에서 E 방향의 단위 벡터다. `arc_length_px`는 S로부터의 경로 위 거리이며 `arc_fraction`은 이를 전체 길이로 나눈 값이다.
- `path.points_px`와 `path.cumulative_length_px`는 같은 길이의 배열이다. 경로 위치를 통해 모델 전처리 좌표로 다시 매핑할 수 있다.
- path mask는 원과 S/E를 그리기 전 선 전체의 0/255 이진 마스크다. 원 뒤에 가려진 선도 포함한다.
- object mask는 원만 그린 uint8 인스턴스 마스크다. 배경은 0, 원은 `mask_value=sequence_index+1`이다. S/E는 포함하지 않는다. 마스크에는 안티앨리어싱된 혼합 ID를 넣지 않는다.
- `files`는 데이터셋 루트 기준 상대 경로이며 `sha256`은 각 저장 PNG의 해시다.

모델의 patch index는 resize, padding, crop, token merging 규칙에 따라 달라진다. 생성기에서는 원본 위치를 저장하고 후속 분석에서 실제 모델 입력 좌표로 변환한다.

## 원전과의 관계

[VLMs Trace Without Tracking 본문 v2](https://arxiv.org/html/2605.15672v2)의 Appendix C.1/D.1을 참고했다. 원 Circuit Connections는 여러 wire 중 지정된 port에서 출발해 색 점 순서와 연결된 부품을 답한다. 원 Swirl은 단일 나선, S/E, 8개 고유 색 점을 사용하며 인접 회전 간격을 조절한다. 이 구현은 쉬운 단일선 조건으로 시작하며 원 Circuit의 부품·port나 Swirl의 density 조건을 그대로 재현하지 않는다. 선 두께와 원 반지름은 이 구현의 선택이다.

직선·완만한 곡선은 좌표 정렬 등으로도 해결할 수 있다. 기본 조건에서 정답을 맞힌 것만으로 내부 tracing 알고리즘이 작동했다고 단정하지 않는다. 이후 내부 표현 분석과 통제 조건을 결합한다.

## 재현 명령

```bash
python -m pip install -r requirements.txt
python -m line_tracing --output data/easy_v0 --count 100 --family curve --seed 42 --preview
python -m line_tracing.audit data/easy_v0
```

```bash
python -m line_tracing --output data/spaced_v0 --count 100 --family curve --path-length 480 --num-objects 8 --object-spacing 56 --position-jitter 0 --seed 42
```

이 설정은 점 8개 사이의 7개 구간이 각각 56px로 총 392px를 차지하고, 양 끝에는 각각 44px의 경로가 남는다. `--path-length`만 늘리면 이 간격과 원 개수는 유지되고 양 끝 여백이 늘어난다.

```bash
python -m line_tracing --output data/fixed_forward --count 8 --family elbow --seed 7 --direction forward --rotation 30
python -m line_tracing --output data/fixed_reverse --count 8 --family elbow --seed 7 --direction reverse --rotation 30
```

기존 출력 경로는 덮어쓰지 않는다. 생성 실패 시 완성되지 않은 데이터셋을 최종 경로에 남기지 않는다. QA preview는 최대 16개 샘플이며 캡션에 정답을 포함하므로 모델 입력으로 사용하지 않는다.
