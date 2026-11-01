# Toy2Box BC 성공률 향상 연구 — C축: EE 기준 상대 좌표로 관측하면 성능이 개선되는가?

- **과업**: `Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0` (Agibot 오른팔, RMPFlow 상대 EE 제어)
- **연구 축**: C (관측·행동 표현). 슬라이드 예시 질문 "물체 위치를 손끝 기준 상대 좌표로 주면?"을 한 주제로 깊게 다룸
- **방법**: 사람 성공 시범에서 잡는 순간과 놓는 순간에 **EE가 본 toy와 box**의 평균 상대 pose를 구해 **잡기 목표**와 **놓기 목표**로 삼았다. 매 step EE에서 본 두 목표(`grasp_err_ee`, `place_err_ee`)를 관측에 넣었다. 비교 대상은 기본 관측 23D만 쓰는 baseline이다.
- **요약** (평가 seed 42, 학습 seed 2개 × rollout 100회):
  - 성공률은 baseline **0.480 ± 0.050**에서 **0.545 ± 0.045**로 **+6.5%p** 올랐다. 다만 통계적으로 유의하지 않다(z = 1.30, p = 0.19).
  - **잡기 단계는 개선됐다.** 트럭을 잡아 들어 올린 rollout이 67%에서 **77%로 늘었다**(p = 0.03). 성공률 개선 6.5%p는 모두 접근 실패와 파지 실패가 줄어든 데서 나왔다.
  - **놓기 단계는 개선되지 않았다.** 놓는 순간 EE가 놓기 목표 xy 10 cm 안에 있던 비율은 48%에서 43%로 오히려 줄었다. 박스 밖 놓기도 16.0%에서 18.5%로 줄지 않았다.

---

## 1. 문제 정의

### 1.1 연구 질문

사람 키보드 시범 20개를 Isaac Lab Mimic으로 500개로 늘려 BC 정책을 학습한다. baseline 정책의 관측은 EE, 트럭, 박스의 **절대 pose**(base 기준) 23D다. 어디로 가서 어떻게 잡고 어디에 놓을지는 정책이 직접 계산해야 한다.

> 성공 시범에서 얻은 **잡기 목표와 놓기 목표까지의 상대 좌표를 EE 기준으로** 관측에 넣으면, baseline보다 두 목표에 잘 도달하고 성공률이 오르는가?

### 1.2 Baseline의 실패 진단

baseline(학습 seed 101, 111)을 검증 seed 42에서 rollout 100회씩 평가했다.

| 실패 단계 | 비율 (2 run 평균) |
|---|---|
| 접근 실패 (잡기 목표 3 cm 안에 한 번도 못 감) | **28.5%** |
| 파지 실패 | 6.5% |
| 운반 실패 | 15.0% |
| 배치 실패 | 2.0% |
| **성공** | **48.0%** |

- 실패의 절반 이상이 **잡기 단계(접근 + 파지, 35.0%)**에서 일어났다.
- 다음으로 큰 원인은 **박스 밖 놓기**(16.0%)다. 트럭을 들어 올렸지만 박스에 도착하기 전에 그리퍼를 열었다.

### 1.3 두 단계에서 확인할 것

| 단계 | 관측에 넣는 상대 좌표 | 확인할 것 |
|---|---|---|
| **잡기** | EE에서 본 파지 목표 $G^*$까지의 위치·자세 오차 `grasp_err_ee` (6D) | baseline보다 많이 잡는가. 잡는 순간 EE가 $G^*$에 얼마나 가까운가 |
| **놓기** | EE에서 본 놓기 목표 $P$까지의 위치 오차 `place_err_ee` (3D) | 놓는 순간 EE가 baseline보다 $P$에 가까운가. 박스 밖 놓기가 줄었는가 |

---

## 2. 가설

> **H_G (잡기).** `grasp_err_ee`를 받으면 baseline보다 잡기 목표 근처까지 가서 트럭을 잡는 경우가 늘고, 잡는 순간의 오차가 작아진다.

> **H_P (놓기).** `place_err_ee`를 받으면 baseline보다 놓기 목표 $P$에 가까운 곳에서 놓게 되어, 박스 밖 놓기가 줄어든다.

**판정 방법**:
- baseline과 같은 지표로 비교한다.
- 목표 도달 오차는 두 기준과도 비교한다.
  - **사람 시범 20개**: 목표를 정의한 데이터
  - **Mimic 데이터 500개**: 두 정책이 실제로 학습한 데이터

---

## 3. 개선 방법

### 3.1 표기

${}^{A}p_{X}$와 ${}^{A}R_{X}$는 frame $X$의 위치와 자세를 frame $A$ 기준으로 나타낸다. quaternion은 Isaac Lab 순서 $(w,x,y,z)$를 쓴다.

| frame | 의미 |
|---|---|
| $B$ | robot base. 모든 관측의 기준 |
| $E$ | end-effector (`right_gripper_center`). $z$축이 그리퍼 접근축 |
| $T$ | toy truck |
| $G$, $G'$ | 파지 목표 frame ($T$에 고정)과, 이를 접근축 기준으로 180° 돌린 대칭 frame |
| $G^*$ | $G$와 $G'$ 중 현재 EE 자세에 더 가까운 쪽 |
| $P$ | 놓기 목표 (박스 중심 위 높이 $\bar h$) |

### 3.2 EE에서 본 toy와 box

모든 상대 좌표는 **EE를 원점과 축으로 삼아** toy와 box를 바라본 값이다. 매 step 관측(base 기준 pose)에서 다음을 계산한다.

$${}^{E}p_{T}=({}^{B}R_{E})^{\top}\big({}^{B}p_{T}-{}^{B}p_{E}\big),\qquad {}^{E}R_{T}=({}^{B}R_{E})^{\top}\,{}^{B}R_{T}\qquad\text{(EE에서 본 toy)}$$

$${}^{E}p_{\text{box}}=({}^{B}R_{E})^{\top}\big({}^{B}p_{\text{box}}-{}^{B}p_{E}\big)\qquad\text{(EE에서 본 box)}$$

- ${}^{E}p_{T}$는 "손끝에서 toy가 어디에 있는가", ${}^{E}R_{T}$는 "손끝에서 toy가 어떤 자세로 보이는가"다.
- 예를 들어 ${}^{E}p_{T}=(0,0,+d)$이면 toy가 그리퍼 접근축($z$) 방향으로 $d$만큼 앞에 있다.

### 3.3 잡기 목표: 성공 시범에서 잡는 순간 EE가 본 toy

**목표 추정.** 사람 시범 20개(모두 성공)에서 잡는 순간 $t_g$에 EE가 본 toy pose $\big({}^{E}p_{T,i},\ {}^{E}R_{T,i}\big)$를 기록했다. "이렇게 보이면 잡을 수 있다"는 정답 상대 pose다.

- toy는 매 에피소드 다른 곳에 놓이므로, 이 정답을 **toy에 붙어 다니는 목표점 $G$**로 바꿔 저장한다. EE에서 본 toy pose의 역변환이다.
  $${}^{T}p_{G,i}=-\big({}^{E}R_{T,i}\big)^{\top}\,{}^{E}p_{T,i},\qquad {}^{T}R_{G,i}=\big({}^{E}R_{T,i}\big)^{\top}\qquad (t=t_g)$$
  - 의미: "toy가 이 자리에 있을 때, 손끝은 toy 기준 ${}^{T}p_{G,i}$에 있었다."
- 평행 그리퍼는 접근축(EE $z$축, $a=2$) 기준으로 180° 돌려도 같은 파지가 된다. 그래서 시범의 회전이 두 집단(8개, 12개)으로 갈린다.
- 각 시범의 회전을 ${}^{T}R_{G,i}$와 ${}^{T}R_{G,i}R_a(\pi)$ 중 기준에 가까운 쪽으로 접은 뒤 평균을 구했다. 위치는 산술 평균, 회전은 SVD chordal mean이다.
  $${}^{T}\bar p_{G}=\frac1N\sum_i {}^{T}p_{G,i},\qquad {}^{T}\bar R_{G}=U\,\mathrm{diag}\big(1,1,\det UV^{\top}\big)V^{\top},\quad U\Sigma V^{\top}=\mathrm{svd}\Big(\sum_i {}^{T}R_{G,i}^{\text{folded}}\Big)$$
  - 결과: ${}^{T}\bar p_{G}=(-0.20,\ -0.43,\ -1.53)$ cm. 접은 뒤 각 시범의 회전은 평균에서 모두 14.2° 이내다.

**매 step 목표.** 현재 EE에서 본 toy pose에 이 목표점을 붙여, **EE에서 본 잡기 목표**를 얻는다.

$${}^{E}p_{G}={}^{E}p_{T}+{}^{E}R_{T}\,{}^{T}\bar p_{G},\qquad {}^{E}R_{G}={}^{E}R_{T}\,{}^{T}\bar R_{G},\qquad {}^{E}R_{G'}={}^{E}R_{G}\,R_{a}(\pi)$$

$${}^{E}R_{G^*}=\arg\min_{X\in\{{}^{E}R_{G},\ {}^{E}R_{G'}\}}\ \angle\big(X,\ I\big),\qquad \angle(X,I)=\arccos\frac{\mathrm{tr}(X)-1}{2}$$

- 대칭 목표 중에서는 손끝에서 봤을 때 덜 돌려도 되는 쪽($I$에 가까운 쪽)을 고른다.
- 잡는 순간의 정답과 같은 상대 pose가 되면 ${}^{E}p_{G}=0$, ${}^{E}R_{G^*}=I$가 된다.

### 3.4 설계 수정: 관측을 base 기준에서 EE 기준으로

처음 설계에서는 같은 목표까지의 오차를 **base 기준**으로 계산했다.

$$e_{G,p}^{B}={}^{B}p_{G}-{}^{B}p_{E},\qquad e_{G,R}^{B}=\log\!\big({}^{B}R_{G^*}\,({}^{B}R_{E})^{\top}\big)^{\vee},\qquad e_{P}^{B}={}^{B}p_{P}-{}^{B}p_{E}$$

- 벡터 방향은 EE에서 목표로 향하지만, **좌표축은 base**다. 로봇 밖에 고정된 관찰자가 EE와 toy·box를 함께 바라본 값이다.
- 그래서 같은 목표라도 EE가 어느 쪽을 향하고 있는지가 오차에 드러나지 않는다. 예를 들어 toy가 그리퍼 바로 앞에 있어도, 그리퍼 자세에 따라 base 기준 오차는 $(+d,0,0)$이 되기도 하고 $(0,0,-d)$가 되기도 한다.

"손끝에서 물체를 바라본다"는 연구 질문에 맞추려면 3.2절처럼 **EE에서 toy와 box를 바라봐야** 한다. 그래서 EE 기준으로 바꿨다. 두 표현은 $e^{E}=({}^{B}R_{E})^{\top}e^{B}$ 관계다.

| | 처음 설계 (base 기준) | 수정 (EE 기준) |
|---|---|---|
| 의미 | 외부(base)에서 본 EE→목표 오차 | **손끝에서 본 toy·box와 그 목표** |
| 잡기 위치 | ${}^{B}p_{G}-{}^{B}p_{E}$ | ${}^{E}p_{G}={}^{E}p_{T}+{}^{E}R_{T}\,{}^{T}\bar p_{G}$ |
| 잡기 자세 | $\log\big({}^{B}R_{G^*}({}^{B}R_{E})^{\top}\big)$ (base 축에서 합성) | $\log\big({}^{E}R_{G^*}\big)$ (EE 자기 축) |
| 놓기 위치 | ${}^{B}p_{P}-{}^{B}p_{E}$ | ${}^{E}p_{P}={}^{E}p_{\text{box}}+\bar h\,{}^{E}\hat z_B$ |
| "toy가 그리퍼 바로 앞 $d$" | EE 자세에 따라 값이 달라짐 | 항상 $(0,0,+d)$ (EE $z$축 = 접근축) |
| 관측 키 | `grasp_err` | `grasp_err_ee`, `place_err_ee` |
| 코드 (`grasp_feat.py`) | `frame="base"` | `frame="ee"` |

- 담긴 정보량은 같고, 바라보는 기준만 다르다.
- 이 보고서의 실험 모델(`gepl`)은 **수정된 EE 기준 관측**을 쓴다.
- 위의 EE 기준 식이 코드의 `frame="ee"` 출력과 같은 값임을 rollout 데이터에서 확인했다(최대 차이 약 $10^{-15}$).

### 3.5 `grasp_err_ee`: EE에서 본 잡기 목표 (6D)

$$\texttt{grasp\_err\_ee}=\Big[\ {}^{E}p_{G},\ \ \log\big({}^{E}R_{G^*}\big)^{\vee}\ \Big]\in\mathbb{R}^6$$

- 앞 3개는 손끝에서 본 잡기 목표점의 위치, 뒤 3개는 손끝에서 본 목표 자세까지의 회전(EE 자기 축 기준 회전 벡터)이다.
- 목표에 도달하면 모두 0이다.

### 3.6 `place_err_ee`: EE에서 본 놓기 목표 (3D)

**목표 추정.** 사람 시범 20개에서 놓는 순간 $t_r$에 EE가 본 box의 위치를 기록했다. 그 수직 성분(base $z$)의 평균이 놓기 높이 $\bar h$다.

$$\bar h=-\frac1N\sum_i\big({}^{B}p_{\text{box}}-{}^{B}p_{E}\big)_{z,\,i}\qquad (t=t_r)$$

- 값은 **25.3 cm**(표준편차 3.4 cm)다. 놓는 순간 손끝은 box 중심보다 평균 25.3 cm 위에 있었다.
- xy 목표는 box 중심이다. 성공 판정이 중심 기준이고, 사람 시범의 평균 xy offset도 (1.7, −1.1) cm로 작기 때문이다.

**매 step 목표.** 현재 EE에서 본 box 위치에, base 위쪽 방향으로 $\bar h$를 더한다.

$$\texttt{place\_err\_ee}={}^{E}p_{P}={}^{E}p_{\text{box}}+\bar h\;{}^{E}\hat z_B,\qquad {}^{E}\hat z_B=({}^{B}R_{E})^{\top}\hat z_B$$

- ${}^{E}\hat z_B$는 손끝에서 본 "위쪽" 방향이다. 놓기 목표 $P$는 box 중심 바로 위 $\bar h$ 지점이다.
- 잡기 전후를 구분하지 않고 항상 관측에 넣는다. 지금이 어느 단계인지는 정책이 `gripper_pos`로 알 수 있다.
- 행동(Δpos, Δrot)은 base 축 기준이므로, 정책은 EE 기준 관측을 다시 base 축으로 돌려 행동을 내야 한다.

### 3.7 평가 시점과 목표 도달 지표

baseline, `gepl`, 사람 시범, Mimic 데이터 모두에 같은 정의를 쓴다(`tools_c/target_reach.py`). baseline은 상대 좌표를 관측으로 받지 않지만, 같은 목표를 기준으로 사후에 계산했다.

- **잡는 순간 $t_g$**: 트럭이 처음 2 cm 이상 들린 시점보다 앞선 100 step 안에서, 마지막으로 그리퍼 명령이 열림에서 닫힘으로 바뀐 step. 즉 실제로 트럭을 들어 올린 닫힘이다.
- **놓는 순간 $t_r$**: 트럭이 2 cm 이상 들린 뒤 처음 열림 명령이 나온 step.
- **잡기 도달 오차**: 잡는 순간 EE에서 본 잡기 목표 $e_{G,p}^{E}={}^{E}p_{G}(t_g)$와 $e_{G,R}^{E}=\log({}^{E}R_{G^*}(t_g))^{\vee}$의 크기, 그리고 EE 축 성분별 평균 $\overline{e_{G,p}^{E}}$(방향 치우침).
- **놓기 도달 오차**: 놓는 순간 EE와 $P$의 수평 거리 $\lVert(p_P-p_E)_{xy}\rVert$, 높이 차 $(p_P-p_E)_z$(수평·수직을 나누기 위해 base 축으로 잼), 그리고 EE에서 본 놓기 목표의 평균 $\overline{e_{P}^{E}}=\overline{{}^{E}p_{P}(t_r)}$.

### 3.8 구현

| 파일 | 내용 |
|---|---|
| `place/mdp/grasp_feat.py` | `grasp_err_from_poses(..., frame="ee")`, `place_err_from_poses(..., frame="ee")`, 각 ObsTerm 함수 |
| `place/config/agibot/grasp_offset.json` | ${}^{T}\bar p_{G}$, ${}^{T}\bar R_{G}$, 접근축 $a$ |
| `place/config/agibot/place_offset.json` | $\bar h$, xy offset 통계 (`tools_c/compute_place_offset.py`로 생성) |
| `place_toy2box_rmp_rel_env_cfg.py` | `PolicyCfg`에 `grasp_err_ee`, `place_err_ee` ObsTerm 추가. 환경·성공 판정·종료 조건은 변경 없음 |
| `agents/robomimic/bc_rnn_low_dim_ge_ee_place.json` | baseline 설정(`bc_rnn_low_dim.json`)에서 `low_dim`에 `grasp_err_ee`, `place_err_ee`를 추가하고 실험 이름만 변경 |
| `tools_c/add_grasp_err_to_hdf5.py` | 데이터셋의 `obs`/`next_obs`에 새 키 추가 → `mimic_500_rel.hdf5` |
| `tools_c/play_record.py`, `analyze_rollouts.py`, `target_reach.py` | rollout 기록·영상 저장, 실패 분류, 목표 도달 지표 |

**학습 전 검증**:
- env ObsTerm 값과 오프라인 계산 값의 최대 차이가 0이었다(`grasp_err_ee` shape (1,6), `place_err_ee` shape (1,3)).
- `gepl`의 학습 데이터(`mimic_500_rel.hdf5`)와 baseline의 학습 데이터(`mimic_500_ge.hdf5`)는 같은 Mimic 데이터셋에서 만들었다. 시범 500개, 행동, 기존 관측이 모두 같았다(최대 차이 0, 총 278,852 step). 새 관측 키만 다르다.

### 3.9 실험 설계 (관측만 다름)

| | baseline | 실험 `gepl` |
|---|---|---|
| 정책 관측 | 23D (EE pos/quat, gripper, truck pos/quat, box pos/quat) | 23D + `grasp_err_ee` 6D + `place_err_ee` 3D = **32D** |
| 학습 데이터 | 같은 시범 500개 (사람 키보드 시범 20개 → Isaac Lab Mimic 생성) | 〃 |
| 모델·학습 | BC-RNN (LSTM 2×400, 길이 10) + GMM 5 modes, batch 100, lr 1e-3, **2000 epoch, 마지막 체크포인트** | 〃 |
| 학습 seed | 101, 111 | 101, 111 |
| 평가 | **seed 42**, rollout 100, horizon 800 | 〃 |

---

## 4. 결과

### 4.1 성공률과 실패 단계

| run | 성공 | 접근 실패 | 파지 실패 | 운반 실패 | 배치 실패 | 박스 밖 놓기 |
|---|---|---|---|---|---|---|
| base_s101 | 0.43 | 0.34 | 0.05 | 0.18 | 0.00 | 0.18 |
| base_s111 | 0.53 | 0.23 | 0.08 | 0.12 | 0.04 | 0.14 |
| gepl_s101 | 0.50 | 0.29 | 0.02 | 0.16 | 0.03 | 0.18 |
| gepl_s111 | 0.59 | 0.20 | 0.03 | 0.17 | 0.01 | 0.19 |
| **baseline 평균** | **0.480 ± 0.050** | 0.285 | 0.065 | 0.150 | 0.020 | 0.160 |
| **gepl 평균** | **0.545 ± 0.045** | 0.245 | 0.025 | 0.165 | 0.020 | 0.185 |
| **변화** | **+6.5%p** | **−4.0%p** | **−4.0%p** | +1.5%p | 0 | +2.5%p |

(± 는 seed 2개의 표준편차)

실패 단계는 아래 규칙을 순서대로 적용해 나눴다. "박스 밖 놓기"는 이와 별개인 지표로, 전체 rollout 대비 비율이다.

| 실패 단계 | 판정 규칙 |
|---|---|
| 접근 | 에피소드 전체에서 $\min_t\lVert e_{G,p}\rVert > 3$ cm (잡기 목표 근처에 한 번도 못 감) |
| 파지 | close 명령이 없거나, close 후 트럭 상승 < 2 cm |
| 운반 | 들어 올린 뒤 박스 xy 0.10 m 안에 한 번도 들어가지 못함 |
| 배치 | 그 외 |
| 박스 밖 놓기 | 실패했고, $t_r$에 트럭–박스 xy ≥ 10 cm |

- **성공률은 6.5%p 올랐지만 유의하지 않다.** 200 대 200 이표본 비율 검정에서 96회 대 109회, z = 1.30, p = 0.19다.
  - 같은 모델에서도 seed에 따라 성공률이 9–10%p 달라, 이 흔들림이 개선 폭보다 크다.
  - 다만 seed별로 보면 두 seed 모두에서 올랐다(0.43 → 0.50, 0.53 → 0.59).
- **개선은 모두 잡기 단계에서 나왔다.** 접근 실패 −4.0%p, 파지 실패 −4.0%p, 합계 −8.0%p다.
- **놓기 단계는 나빠졌다.** 운반 실패가 +1.5%p, 박스 밖 놓기가 +2.5%p 늘었다. 잡기에서 얻은 8.0%p 중 1.5%p를 운반·놓기에서 잃었다.

### 4.2 잡기 단계: baseline보다 잡기 목표에 잘 갔나

잡는 순간 $t_g$의 오차다.

| 데이터 | n | 잡은 비율 | ‖e_G,p‖ < 3 cm | ‖e_G,p‖ (cm, 중앙값) | EE 축 평균 $\overline{e_{G,p}^{E}}$ (cm) | ‖e_G,R‖ (°, 중앙값) |
|---|---|---|---|---|---|---|
| 사람 시범 (목표 정의) | 20 | 1.00 | 0.65 | 2.9 | (−1.2, +0.4, −0.4) | 7.0 |
| Mimic 데이터 (학습) | 500 | 1.00 | 0.59 | 2.7 | (−1.2, +0.2, −0.3) | 7.2 |
| **baseline** rollout | 200 | **0.67** | 0.78 | 2.2 | (−0.8, +0.2, +0.2) | 9.9 |
| └ 성공 | 96 | 1.00 | 0.79 | 2.2 | (−0.7, +0.2, +0.2) | 9.9 |
| └ 실패 | 104 | 0.37 | 0.74 | 2.3 | (−1.0, +0.1, +0.4) | 9.9 |
| **gepl** rollout | 200 | **0.77** | 0.70 | 2.1 | (−0.6, +0.3, −0.1) | **8.7** |
| └ 성공 | 109 | 1.00 | 0.72 | 2.1 | (−0.9, +0.3, −0.3) | 7.8 |
| └ 실패 | 91 | 0.49 | 0.67 | 2.3 | (+0.1, +0.2, +0.4) | 9.4 |

- "잡은 비율"은 트럭을 2 cm 이상 들어 올린 rollout의 비율이다. 오차 열은 잡은 rollout만으로 계산했다.
- run별 잡은 비율: baseline 0.64, 0.70 → gepl 0.72, 0.82 (seed 101, 111). 두 seed 모두에서 늘었다.

**해석**:
- **트럭을 잡는 경우가 늘었다 (H_G 일부 지지).**
  - 잡은 비율이 67%에서 **77%로 10%p 늘었다**. 134회 대 154회로, 이표본 비율 검정 z = 2.23, p = 0.03이다.
  - 성공률 개선의 출처(접근 실패와 파지 실패 감소)와 같은 방향이다.
- **잡은 경우의 위치 정확도는 baseline도 이미 높았다.**
  - baseline도 실제로 잡은 경우에는 위치 오차 중앙값이 2.2 cm로, gepl(2.1 cm)과 같다. 둘 다 학습 데이터(2.7 cm)보다 작다.
  - 즉 상대 좌표는 "잡을 때 더 정확하게"보다 **"잡을 수 있는 위치까지 더 자주 가게"** 했다.
  - 자세 오차는 9.9°에서 8.7°로 1.2° 줄었다.
- **잘못된 닫힘도 줄었다.** 공중에서 닫은 경우를 포함해, 처음 그리퍼를 닫은 순간의 오차는 다음과 같다(`analyze_rollouts.py`).
  - 위치 오차 평균: 10.3 cm → 8.0 cm
  - 자세 오차 평균: 22.1° → 16.4°

접근 실패는 두 유형으로 나뉜다(rollout 수).

| 유형 | 정의 | baseline | gepl |
|---|---|---|---|
| A. 기울어진 채 위에서 멈춤 | $\min_t\lVert e_{G,p}\rVert > 10$ cm | 32 | 29 |
| B. 근처에서 아깝게 놓침 | 3 cm < $\min_t\lVert e_{G,p}\rVert$ ≤ 10 cm | 25 | 20 |

- 두 유형 모두 조금씩 줄었다. 하지만 seed에 따라 A와 B의 비율이 크게 다르다(예: gepl_s101은 A 21건, B 8건, gepl_s111은 A 8건, B 12건). 그래서 유형별 차이는 해석하지 않는다.
- 접근 실패 49건(24.5%)은 여전히 가장 큰 실패 원인이다.

### 4.3 놓기 단계: baseline보다 놓기 목표에 잘 갔나

놓는 순간 $t_r$의 오차다.

| 데이터 | n | 놓은 비율 | xy(E,P) < 10 cm | xy(E,P) (cm, 중앙값) | z(E,P) (cm, 중앙값) | EE 축 평균 $\overline{e_{P}^{E}}$ (cm) |
|---|---|---|---|---|---|---|
| 사람 시범 (목표 정의) | 20 | 1.00 | **0.90** | 6.4 | +1.5 | (−0.4, −0.4, −1.9) |
| Mimic 데이터 (학습) | 500 | 1.00 | 0.57 | 8.9 | +1.9 | (−1.6, −1.5, −2.1) |
| **baseline** rollout | 200 | 0.64 | **0.48** | 10.3 | +1.4 | (+3.5, −2.8, −1.9) |
| └ 성공 | 96 | 1.00 | 0.64 | 8.4 | +1.6 | (+2.7, −2.6, −1.8) |
| └ 실패 | 104 | 0.31 | 0.00 | 18.0 | +1.0 | (+6.1, −3.2, −1.9) |
| **gepl** rollout | 200 | 0.73 | **0.43** | 11.4 | +1.4 | (+3.7, −3.0, −3.0) |
| └ 성공 | 109 | 1.00 | 0.58 | 9.0 | +1.4 | (+1.6, −2.9, −2.5) |
| └ 실패 | 91 | 0.41 | 0.00 | 21.0 | +0.2 | (+9.8, −3.3, −4.6) |

- "놓은 비율"은 트럭을 들어 올린 뒤 그리퍼를 연 rollout의 비율이다. 오차 열은 놓은 rollout만으로 계산했다.
- run별 xy < 10 cm 비율: baseline 0.39, 0.55 → gepl 0.37, 0.49 (seed 101, 111). 두 seed 모두에서 줄었다.

**해석**:
- **놓기 목표에 더 가까이 가지 않았다 (H_P 기각).**
  - 놓는 순간 $P$ xy 10 cm 안 비율이 48%에서 43%로, xy 중앙값이 10.3 cm에서 11.4 cm로 오히려 나빠졌다.
  - 박스 밖 놓기도 32건에서 37건으로 줄지 않았다(z = 0.66, p = 0.51로 차이는 유의하지 않음).
- **두 모델 모두 $P$가 아니라 Mimic 데이터의 놓기 분포를 따라 했다.**
  - 성공 rollout의 xy 중앙값(baseline 8.4 cm, gepl 9.0 cm)은 Mimic 데이터(8.9 cm)와 비슷하다.
  - Mimic 데이터 자체가 사람 시범보다 $P$에서 멀리 놓는다(6.4 → 8.9 cm). 증강 과정에서 놓기 위치가 퍼졌고, 두 정책 모두 그 분포를 배웠다.
- **gepl은 높이만 맞추고 더 일찍 놓았다.**
  - gepl 실패 rollout은 놓는 순간 높이 차가 **+0.2 cm**로 목표 높이에 거의 정확히 있었다(baseline +1.0 cm). 반면 xy로는 **21.0 cm** 떨어져 있었다(baseline 18.0 cm).

놓는 시점도 비교했다(각 값은 중앙값).

| 모델 | 구분 | 건수 | 트럭–박스 xy (cm) | 들어 올린 뒤 놓기까지 (step) |
|---|---|---|---|---|
| baseline | 박스 밖 놓기 | 32 | 20.1 | 87 |
| baseline | 성공 | 96 | 8.4 | 92 |
| **gepl** | **박스 밖 놓기** | **37** | **24.3** | **58** |
| gepl | 성공 | 109 | 8.3 | 93 |

- 성공 rollout의 놓기 시점은 두 모델이 같다(92, 93 step).
- 박스 밖 놓기는 gepl이 baseline보다 약 **29 step(1.45 s) 일찍** 그리퍼를 열었고, 박스에서 더 먼 곳(24.3 cm)에서 놓았다.
- 해석: 학습 시범에서 "그리퍼를 연다"는 행동은 $e_P$의 높이 성분이 0에 가까워지는 순간과 맞물려 있다. `place_err_ee`를 받은 정책이 xy보다 **높이 성분을 놓기 신호로 학습**했을 가능성이 있다. 그러면 높이를 먼저 맞추는 궤적에서는 박스에 도착하기 전에 놓는다.

### 4.4 결론

| 지표 | baseline | gepl | 변화 |
|---|---|---|---|
| **성공률** | 0.480 | 0.545 | **+6.5%p** (p = 0.19) |
| 잡은 비율 | 0.67 | 0.77 | **+10%p** (p = 0.03) |
| 접근 실패 + 파지 실패 | 35.0% | 27.0% | **−8.0%p** |
| 잡는 순간 위치 오차 (중앙값) | 2.2 cm | 2.1 cm | 차이 없음 |
| 잡는 순간 자세 오차 (중앙값) | 9.9° | 8.7° | −1.2° |
| 놓는 순간 $P$ xy 10 cm 안 | 48% | 43% | **−5%p** (악화) |
| 박스 밖 놓기 | 16.0% | 18.5% | +2.5%p (악화, p = 0.51) |

1. **잡기 단계는 개선됐다.**
   - EE 기준 잡기 목표 오차를 받은 정책은 baseline보다 트럭을 잡는 경우가 10%p 늘었다(p = 0.03).
   - 잡은 경우의 위치 정확도는 baseline도 이미 학습 데이터 수준이었다. 개선은 "목표 근처까지 가서 잡는 횟수"에서 나왔다.
2. **놓기 단계는 개선되지 않았다.**
   - EE 기준 놓기 목표 오차를 받아도 놓기 목표에 더 가까이 가지 않았다. 박스 밖 놓기도 줄지 않았다.
   - 두 모델 모두 놓기 목표가 아니라 Mimic 데이터의 놓기 위치 분포(xy 약 9 cm)를 따라 했다.
   - gepl은 높이만 맞추고 baseline보다 일찍 놓는 실패가 나타났다.
3. **종합: 성공률은 +6.5%p 올랐지만 유의하지 않다.**
   - 잡기에서 얻은 이득(−8.0%p 실패)의 일부를 운반·놓기에서 잃었다(+1.5%p 실패).
   - `place_err_ee`는 이미 관측에 있는 `box_positions − eef_pos`에 상수 높이를 더하고 회전만 곱한 값이다. 새 정보가 아니라 표현만 바꾼 것이어서 데이터 분포를 넘어서는 효과가 없었고, 오히려 높이가 놓기 신호로 쓰였을 수 있다.
   - 반면 `grasp_err_ee`는 두 대칭 목표 중 가까운 쪽을 고르는 불연속 판단을 대신해 준다. 정책이 스스로 배우기 어려운 계산을 대신할 때 상대 좌표가 효과를 낸 것으로 본다.
4. **다음 단계 후보**:
   - 놓기 단계: Mimic 데이터에서 놓는 위치가 박스 중심 근처인 시범만 골라 학습하거나, 그런 시범을 더 생성하기
   - 잡기 단계: 남은 접근 실패(24.5%)를 줄이기 위해, 그리퍼가 기울어진 상태에서 복구하는 시범 추가하기

### 4.5 한계

- **표본 크기**: 학습 seed 2개와 rollout 100회로는 6.5%p 차이를 검출하기에 검정력이 부족하다. 같은 모델에서도 seed에 따라 성공률이 9–10%p 달랐다.
- **평가 초기 상태**: 같은 seed 42를 써도 rollout 8번째부터 run마다 초기 배치가 달라진다(에피소드 길이에 따라 RNG 소비량이 달라짐). 그래서 baseline과 gepl을 같은 초기 상태끼리 짝지어 비교하지 못하고, 비짝 비교만 했다.
- **두 관측을 함께 추가**: gepl은 `grasp_err_ee`와 `place_err_ee`를 동시에 추가했다. 잡기 개선과 놓기 악화가 각각 어느 관측 때문인지는 이 비교만으로 분리할 수 없다. 단계별 해석(잡기 ↔ `grasp_err_ee`, 놓기 ↔ `place_err_ee`)은 추정이다.
- **사후 평가**: 4.2, 4.3절의 목표 도달 지표, 잡는 순간 정의, 놓기 시점 분석, 그리고 이 baseline 대 gepl 비교의 판정 방법은 결과를 본 뒤 정했다. 사전에 정한 판정 기준으로 평가한 것이 아니다.
- **놓기 목표 높이**: $\bar h$는 사람 시범 20개에서 구했다. 학습 데이터(Mimic)의 놓기 높이와 약 0.4 cm 차이가 있다(높이 차 중앙값 +1.5 대 +1.9 cm).

---

### 부록 A: 재현 명령

```bash
conda activate lerobot-arena && cd ~/IsaacLab_RS
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0

# 학습 데이터: Mimic 데이터에 새 관측 키 추가 → datasets/mimic_500_rel.hdf5
bash tools_c/run_c_pipeline.sh convert_rel

# 학습 (seed 101, 111). baseline은 --algo bc, 실험은 --algo bc_ge_ee_place
for SEED in 101 111; do
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
    --algo bc --dataset ./datasets/mimic_500_rel.hdf5 --seed $SEED
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
    --algo bc_ge_ee_place --dataset ./datasets/mimic_500_rel.hdf5 --seed $SEED
done

# 평가 (검증 seed 42, rollout 기록 + 영상). 체크포인트와 출력 폴더만 바꿔 4 run 반복
./isaaclab.sh -p tools_c/play_record.py --task $TASK \
  --checkpoint ./logs/robomimic/$TASK/bc_rnn_low_dim_agibot_toy2box_ge_ee_place_seed111/20261030170827/models/model_epoch_2000.pth \
  --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir ./outputs/rollouts/gepl_s111 \
  --video --video_length 800 --video_folder ./videos/rel/gepl_s111

# 실패 단계 비교 (4.1절)
python tools_c/analyze_rollouts.py --baseline outputs/rollouts/base_s101 outputs/rollouts/base_s111 \
  --exp outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111

# 목표 도달 지표 (4.2, 4.3절 표)
python tools_c/target_reach.py --split \
  --hdf5 datasets/team10_lee_s3.hdf5 datasets/team10_lee_s3_mimic_500.hdf5 \
  --rollouts outputs/rollouts/base_s101 outputs/rollouts/base_s111 outputs/rollouts/gepl_s101 outputs/rollouts/gepl_s111
```

- baseline 설정(`bc`)은 새 관측 키를 읽지 않으므로, `mimic_500_rel.hdf5`로 학습해도 위 baseline과 같은 입력이다.

### 부록 B: 데이터, 체크포인트, 영상

| 단계 | 파일 |
|---|---|
| 사람 시범 기록 (`record_demos.py`) | `datasets/team10_lee_s3.hdf5` (20개) |
| Annotation | `datasets/team10_lee_s3_annotated.hdf5` |
| Mimic 생성 | `datasets/team10_lee_s3_mimic_500.hdf5` (성공 500개) |
| 학습 데이터 | `datasets/mimic_500_rel.hdf5` (위 파일 + 새 관측 키) |

| run | 체크포인트 (`logs/robomimic/<task>/…/models/model_epoch_2000.pth`) | rollout 기록 | 영상 |
|---|---|---|---|
| base_s101 | `bc_rnn_low_dim_agibot_toy2box_seed101/20261030112049` | `outputs/rollouts/base_s101/` | 없음 |
| base_s111 | `bc_rnn_low_dim_agibot_toy2box_seed111/20261030115923` | `outputs/rollouts/base_s111/` | 없음 |
| gepl_s101 | `bc_rnn_low_dim_agibot_toy2box_ge_ee_place_seed101/20261030163015` | `outputs/rollouts/gepl_s101/` | `videos/rel/gepl_s101/` |
| gepl_s111 | `bc_rnn_low_dim_agibot_toy2box_ge_ee_place_seed111/20261030170827` | `outputs/rollouts/gepl_s111/` | `videos/rel/gepl_s111/` |

- 영상 `rollout-episode-K.mp4`는 같은 run의 기록 `outputs/rollouts/<run>/rollout_K.npz`와 같은 rollout이다.
- gepl 대표 영상 번호:

| 현상 | gepl_s101 | gepl_s111 |
|---|---|---|
| 성공 | 0, 1, 6 | 0, 1, 2 |
| 접근 실패: 기울어진 채 위에서 멈춤 | 2, 3, 5 | 3, 15, 44 |
| 접근 실패: 근처에서 놓침 | 8, 46, 56 | 5, 6, 8 |
| 박스 근처에 못 가고 놓음 | 9, 17, 18 | 9, 12, 21 |
| 파지 실패 | 4, 70 | 59, 80, 84 |
