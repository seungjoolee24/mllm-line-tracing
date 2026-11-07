# Exp001 수집 결과

정식 결과는 **run002, 프로토콜 1.1**이다. 10문제·2,060개 tensor를 모두 수집했고, Colab에서 각 파일의 SHA256·shape·dtype와 생성/replay 정렬을 검증했다. 정답은 9/10이며 오답은 case_03_swirl이다.

[실행 요약](run002/run_summary.json) · [용량 보고서](run002/size_report.json) · [파일 manifest](run002_release/artifact_manifest.json) · [분석 노트북](../analysis_run002.ipynb)

원시 파일은 총 4,253,107,416 bytes이며 GitHub 릴리스에 올릴 ZIP은 문제별 10개와 metadata 1개, 합계 3,706,256,058 bytes다. 원시 ZIP의 GitHub 업로드는 취소했다. 원시는 현재 수집 Colab 런타임에 있으며 같은 노트북에서 분석을 계속한다. 장기 보관은 사용자의 Drive를 사용할 계획이며 아직 복사·검증 전이다. 보관이 확인되기 전에는 원시 데이터가 있는 런타임을 종료하지 않는다.

초기 run001은 모델의 sampling 기본값이 상속된 오류 때문에 greedy baseline에서 제외한다. [첫 계측 기록](pilot_run001/artifact_manifest.json)과 [초기 원시 ZIP 릴리스](https://github.com/seungjoolee24/mllm-line-tracing/releases/tag/exp001-pilot-run001)는 저장 용량과 수집 무결성 확인용으로 보존한다.
