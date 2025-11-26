# 생명체형 메타러닝 집적회로 (Bio-Meta-IC)

이 프로젝트는 생명체 유사 구조(노드=세포/소자, 엣지=시냅스/배선)를 그래프 형태의 집적회로로 모델링하고, 메타러닝(MAML) 기반으로 사용자가 지정한 대상 패턴이나 행동을 빠르게 모방하는 프레임워크입니다. 파이썬과 C를 결합하여 PyCharm에서 개발 및 실행을 지원합니다.

## 빠른 시작
1. Python 3.10+ 설치
2. `pip install -r requirements.txt`
3. C 백엔드 빌드: `make -C csrc`
4. 예제 회로 구성: `python examples/build_bio_ic.py`
5. 메타 학습 실행: `python examples/quick_train.py --task config/tasks/target_mimic.yaml`
6. 시각화: `python src/viz/visualize.py --graph out/circuit.graphml --mode grid`

## 시각화 추천
- 네트워크 구조: Graphviz, Gephi
- 동적 신호/상태: PyQtGraph, matplotlib
- 3D 유사-생체 형태: Blender (Geometry Nodes), Unity/Unreal로 내보내기

## 구조 개요
- `src/circuit`: 생체형 회로 모델(노드/엣지/동역학)
- `src/models`: 미분 가능한 IC 모델, 손실함수
- `src/meta_core`: MAML 스타일 메타 트레이너
- `csrc`: C로 작성된 빠른 상태 업데이트 커널
- `src/viz`: 그래프 및 활동 시각화/내보내기
