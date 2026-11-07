> **이전 프로토콜 결과**: run001은 모델의 sampling 기본값이 상속된 오류가 확인되어 greedy baseline에서 제외한다. 원시 파일 무결성과 수집 범위 검증은 통과했으며, 저장 용량과 계측 확인 기록으로 보존한다. 수정된 프로토콜 1.1은 run002에서 다시 실행한다.

# 실험 결과 저장 상태

`pilot_run001/`은 Colab의 Tesla T4에서 실제 실행한 **첫 1문항의 고정 스냅샷**이다. 전체 계획은 10문항이며, 이 스냅샷에는 `case_00_curve`만 포함된다. 이후 실행 결과로 이 폴더를 덮어쓰지 않는다.

## 첫 파일럿 결과

- 모델: `Qwen/Qwen3-VL-4B-Instruct`, revision `70c41a926abfc92501c4a4da369a8d6312915b61`.
- 답: `brown yellow green orange blue purple end`. 정답과 일치했다.
- 이미지 토큰 400개, 질문 토큰 48개, 생성 토큰 8개를 기록했다.
- 활성값·attention 등 텐서 파일은 206개, 합계 423,801,468 bytes(404.17 MiB)다.
- 최대 GPU allocated memory는 9,267,371,520 bytes, reserved memory는 9,506,390,016 bytes였다.
- 생성 답의 causal replay는 모든 생성 단계에서 같은 argmax를 재현했다. 원본 생성과 replay의 최대 logit 차이는 0.125였다.
- 로컬에서 206개 텐서의 파일 해시와 shape·dtype을 검증했다.

## GitHub에 저장하는 파일

[고정 스냅샷](pilot_run001/)에는 실행 config, 환경, 노트북 provenance, 모델·생성 config, 실행 요약·크기 보고서, 첫 문항의 metadata와 token geometry를 둔다. [artifact manifest](pilot_run001/artifact_manifest.json)에 각 파일의 크기와 SHA-256이 있다.

**이 GitHub 폴더에는 원시 텐서 `.pt`나 ZIP archive가 포함되지 않는다.** 원시 결과는 로컬에 저장·검증됐으며 외부 업로드는 아직 완료되지 않았다. 업로드 완료 전에는 다운로드 링크를 제공하지 않는다.

원본 archive `run001.zip`은 369,354,665 bytes이고 SHA-256은 다음과 같다.

```text
e27bc7310799feeab901d12bbe240f9990bcdcaff65fec96e55dd8d6492c4616
```

## 전체 실험 상태

첫 문항 수집만 완료했다. 추가 9문항 실행은 실패 원인을 진단 중이며, 10문항 전체 실험이 완료된 상태가 아니다. 이 스냅샷의 `run_summary.json`도 `status: partial`, `completed_cases: 1`, `intended_cases: 10`으로 보존한다. 후속 결과는 별도 스냅샷으로 관리한다.
