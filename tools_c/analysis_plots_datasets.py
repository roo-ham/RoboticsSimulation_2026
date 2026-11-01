#!/usr/bin/env python3
"""IsaacLab 데모(hdf5) 결과를 4-panel 서브플롯으로 시각화한다.

`summarize_demos.py` 와 동일한 데이터 규약(`data/demo_i/obs/*`, `data/demo_i/actions`)을
사용하며, hdf5 안의 모든 데모(기본: 최대 20개)를 한 화면에 겹쳐 그린다.

패널 구성 (모두 x 축 = 정규화 시간 t_reg)
    1) 그리퍼-트럭 위치 오차 norm  : 월드(베이스) 프레임 xyz, ‖Δp‖ = ‖p_truck - p_eef‖
    2) 그리퍼-박스 위치 오차 norm  : 월드 xy 평면, ‖Δp_xy‖ = ‖[p_box - p_eef]_xy‖
    3) 그리퍼-트럭 자세 오차 norm  : θ = 2·arccos(|q_gripper · q_truck|)   [도, deg]
    4) 그리퍼 open / close 시점    : open = 동그라미(○), close = 다이아몬드(◆)
    추가로 ① 의 각 꺾은선 위에 그 데이터의 close 시점(◆), ② 의 각 꺾은선 위에
    open 시점(○)을 같은 마커 디자인으로 찍는다(마커는 해당 궤적 선 위의 y 값).

용어 정리
    * "xyz 상대 도메인(공간 상대, 회전축 절대)" 의 뜻
      → 위치(상대): **월드 프레임의 두 점을 그냥 뺀 것** Δp = p_target - p_eef.
         그리퍼(EE) 좌표계로 회전시키지 않는다. 따라서 ‖Δp‖ 는 두 점 사이의
         보통 유클리드 거리이고, xy 성분은 world xy(수평면) 기준 거리다.
      → 회전(절대): 축을 월드/베이스 축 그대로 두고 쿼터니언을 합성한다.
         각도는 이미지의 정의대로 θ = 2·arccos(|q1·q2|) (회전축이 월드 축).
    * z 는 높이(수직) 축이다: 로봇 base 자세가 identity 이고 base 는 z 로만
      평행이동했으므로 base 축 = world 축 (obs_z + base_z = world z).
      물체는 수평 이동 시 z 가 일정하고, 들어올릴 때만 z 가 커진다.
    * 쿼터니언은 IsaacLab 규약 (w, x, y, z) 이며, 내적은 이미지의 정의대로
      4차원 벡터 내적 (w1w2 + x1x2 + y1y2 + z1z2) 이다.
    * 그리퍼 open/close 신호는 `actions[:, 6]` 의 부호(+1 = open, -1 = close) 전환이다
      (BinaryJointPositionActionCfg 규약).

사용 예 (실행하면 matplotlib 창이 떠서 바로 볼 수 있고, 창을 닫으면 종료된다)
    ./isaaclab.sh -p tools_c/analysis_plots_datasets.py \
        --input datasets/team10_lee_s3.hdf5 \
        --out outputs/analysis_plots/team10_lee_s3_overview.png

    # matplotlib 만 있는 환경에서도 동작한다.
    python tools_c/analysis_plots_datasets.py --input datasets/team10_lee_s3_annotated.hdf5

    # 창 없이 PNG 만 만들기 (배치/원격 실행)
    python tools_c/analysis_plots_datasets.py --no-show
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# matplotlib 캐시 디렉터리 (읽기 전용 HOME 이어도 죽지 않도록 matplotlib import 전에 지정)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import h5py
import numpy as np

import matplotlib

# 백엔드는 plot_dataset() 에서 고른다(GUI 창을 우선, 없으면 Agg 로 자동 대체).
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_INPUT = os.path.join(REPO, "datasets", "team10_lee_s3.hdf5")
DEFAULT_OUTDIR = os.path.join(REPO, "outputs", "analysis_plots")

OPEN_CMD = 1.0   # 그리퍼 open 명령
CLOSE_CMD = -1.0  # 그리퍼 close 명령

# 창을 띄울 수 있는 백엔드 후보 (앞에서부터 시도)
INTERACTIVE_BACKENDS = ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "MacOSX")


# --------------------------------------------------------------------------------------
# 백엔드 / 창
# --------------------------------------------------------------------------------------
def has_display() -> bool:
    """GUI 창을 띄울 수 있는 디스플레이가 있는지 확인한다."""
    if sys.platform.startswith(("win", "darwin")):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def select_backend(requested: str = "auto", allow_interactive: bool = True, verbose: bool = True) -> str:
    """matplotlib 백엔드를 고른다. 가능한 한 GUI 창(TkAgg 등)을 우선한다.

    Args:
        requested: "auto" 이면 자동 선택, 그 외에는 사용할 백엔드 이름(예: "TkAgg", "Agg").
        allow_interactive: False 면 창 없는 Agg 를 쓴다(배치 실행).
        verbose: 선택 결과를 콘솔에 알린다.

    Returns:
        실제로 사용 중인 백엔드 이름.
    """
    def _try(name: str) -> str | None:
        try:
            plt.switch_backend(name)
            return matplotlib.get_backend()
        except Exception:  # noqa: BLE001 - 백엔드를 못 쓰면 다음 후보로
            return None

    if requested and requested.lower() != "auto":
        got = _try(requested)
        if got is not None:
            if verbose:
                print(f"[ui    ] matplotlib backend = {got} (--backend 지정)", flush=True)
            return got
        if verbose:
            print(f"[warn  ] 백엔드 '{requested}' 를 사용할 수 없어 자동 선택으로 대체합니다.")

    if allow_interactive and has_display():
        for name in INTERACTIVE_BACKENDS:
            got = _try(name)
            if got is not None:
                if verbose:
                    print(f"[ui    ] matplotlib backend = {got} (창 표시)", flush=True)
                return got

    got = _try("Agg") or "Agg"
    if verbose:
        if not allow_interactive:
            print(f"[ui    ] --no-show: 창을 띄우지 않고 저장만 합니다 (backend={got}).")
        elif not has_display():
            print("[ui    ] 디스플레이가 없어 창을 띄우지 않습니다(Agg). PNG 로 저장합니다.")
        else:
            print(f"[ui    ] 사용 가능한 GUI 백엔드가 없어 Agg 로 저장만 합니다 (backend={got}).")
    return got


def screen_size() -> tuple[int, int] | None:
    """주 디스플레이의 해상도를 (w, h) 로 반환한다(알 수 없으면 None)."""
    if not has_display():
        return None
    try:  # tkinter 로 조회 (matplotlib Tk 백엔드와 별개 인스턴스)
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
        root.destroy()
        return size
    except Exception:  # noqa: BLE001
        return None


def set_window_title(fig, title: str) -> None:
    """창 제목을 설정한다(백엔드에 따라 지원되지 않으면 무시)."""
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------------------
# 수학 유틸
# --------------------------------------------------------------------------------------
def quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """두 쿼터니언 사이의 회전각 θ = 2·arccos(|q1 · q2|) [rad].

    이미지의 정의대로 4차원 벡터 내적(내적)을 사용하며, q 와 -q 가 같은 회전임을
    반영하기 위해 |·| 를 취한다. 반환값 범위는 [0, π].

    Args:
        q1: shape (..., 4) 쿼터니언 (w, x, y, z).
        q2: shape (..., 4) 쿼터니언 (w, x, y, z).

    Returns:
        shape (...) 회전각 [rad].
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    n1 = np.linalg.norm(q1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(q2, axis=-1, keepdims=True)
    q1 = q1 / np.where(n1 < 1e-12, 1.0, n1)
    q2 = q2 / np.where(n2 < 1e-12, 1.0, n2)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def delta_in_world(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    """월드(베이스) 프레임의 두 점 차이 Δp = p_to - p_from.

    "상대 도메인"은 두 월드 좌표의 차이를 그대로 쓰는 것이며, 그리퍼(EE) 좌표계로
    회전시키지 않는다. 회전(자세)만 월드 축 기준으로 비교한다.

    Args:
        p_from: shape (T, 3) 기준점(그리퍼) 위치 [m].
        p_to: shape (T, 3) 대상(트럭/박스) 위치 [m].

    Returns:
        shape (T, 3) 상대 위치 벡터 [m] (world 축).
    """
    return np.asarray(p_to, dtype=np.float64) - np.asarray(p_from, dtype=np.float64)


# --------------------------------------------------------------------------------------
# 데이터 로딩
# --------------------------------------------------------------------------------------
def _get(ep: h5py.Group, key: str, required: bool = True) -> np.ndarray | None:
    """에피소드 그룹에서 배열을 읽는다(없으면 required 여부에 따라 None 또는 예외)."""
    if key in ep:
        return ep[key][:]
    if required:
        raise KeyError(f"'{ep.name}' 에 '{key}' 가 없습니다. 지원되는 데모 hdf5 인지 확인하세요.")
    return None


def demo_keys(df: h5py.Group) -> list[str]:
    """`demo_0`, `demo_1`, ... 를 숫자 순으로 정렬해 반환한다."""

    def _idx(name: str) -> int:
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return 1 << 30

    return sorted(df.keys(), key=_idx)


def motion_start_index(actions: np.ndarray, eef_pos: np.ndarray | None, eps: float = 1e-6) -> int:
    """\"움직임 시작\" 프레임을 찾는다 (t_reg = 0 의 기준).

    데모는 시작 부분에 arm action 이 모두 0 인 idle 구간을 갖는다. 첫 번째로
    xyz/rpy action 이 0 이 아닌 프레임을 시작으로 본다. 그런 프레임이 없으면
    (예: action 이 항상 0 인 데이터) 그리퍼 위치가 실제로 변하기 시작하는 프레임,
    그것도 없으면 0 을 반환한다.
    """
    arm = np.nan_to_num(np.asarray(actions[:, :6], dtype=np.float64))
    nonzero = np.flatnonzero(np.any(np.abs(arm) > eps, axis=1))
    if nonzero.size:
        return int(nonzero[0])

    if eef_pos is not None and len(eef_pos) > 1:
        step = np.linalg.norm(np.diff(np.asarray(eef_pos, dtype=np.float64), axis=0), axis=-1)
        moved = np.flatnonzero(step > eps)
        if moved.size:
            return int(moved[0])
    return 0


def load_episode(ep: h5py.Group, eps: float = 1e-6) -> dict:
    """한 에피소드에서 시각화에 필요한 시계열을 계산해 dict 로 반환한다."""
    actions = _get(ep, "actions")
    eef_pos = _get(ep, "obs/eef_pos")
    eef_quat = _get(ep, "obs/eef_quat")
    truck_pos = _get(ep, "obs/toy_truck_positions")
    truck_quat = _get(ep, "obs/toy_truck_orientations")
    box_pos = _get(ep, "obs/box_positions")

    n = int(len(actions))
    t0 = motion_start_index(actions, eef_pos, eps=eps)  # 동작 시작 인덱스 (t_reg = 0)
    t0 = int(np.clip(t0, 0, max(n - 1, 0)))
    sl = slice(t0, n)  # idle 프리픽스는 버리고 [t0, n-1] 만 사용

    m = n - t0
    t_reg = np.linspace(0.0, 1.0, m) if m > 1 else np.zeros(1)

    # 위치(상대): 월드 프레임 두 점의 차이. EE 좌표계로 회전하지 않는다.
    d_truck = delta_in_world(eef_pos[sl], truck_pos[sl])
    d_box = delta_in_world(eef_pos[sl], box_pos[sl])

    # 자세(절대): 월드 축 기준 쿼터니언 차 → θ = 2·arccos(|q1·q2|)
    theta_deg = np.degrees(quat_angle(eef_quat[sl], truck_quat[sl]))

    # 그리퍼 open / close 신호 (actions[:, 6] 의 부호 전환: +1 = open, -1 = close)
    cmd = np.asarray(actions[:, 6], dtype=np.float64)
    sign = np.where(cmd > 0.0, OPEN_CMD, np.where(cmd < 0.0, CLOSE_CMD, 0.0))
    events = []
    for i in np.flatnonzero(np.diff(sign) != 0.0):
        idx = int(i + 1)
        if sign[idx] == 0.0 or idx < t0:
            # 0(명령 없음)으로 바뀐 경우와, 움직임 시작 이전(프리픽스)은 버린다.
            continue
        events.append((float(t_reg[idx - t0]), "open" if sign[idx] > 0 else "close", idx))
    if t0 == 0 and sign.size and sign[0] < 0.0:
        # 첫 프레임부터 그리퍼가 닫혀 있는 경우도 close 시점으로 취급
        events.insert(0, (0.0, "close", 0))

    return {
        "name": ep.name.rsplit("/", 1)[-1],
        "n": n,
        "t0": t0,
        "t_reg": t_reg,
        "d_truck": d_truck,
        "d_box": d_box,
        "theta_deg": theta_deg,
        # 접근축 기준 대칭(grasp_offset.json 의 ang_folded_deg 와 같은 규약):
        # 180° 근처 값의 미세 변화를 보기 좋게 접는다.
        "theta_folded_deg": np.minimum(theta_deg, 180.0 - theta_deg),
        "events": events,
        "num_samples": int(ep.attrs.get("num_samples", n)),
    }


# --------------------------------------------------------------------------------------
# 플롯
# --------------------------------------------------------------------------------------
def _labels(use_korean: bool) -> dict:
    """그림에 쓸 문자열(한글 폰트가 없으면 영어로 대체)."""
    if use_korean:
        return {
            "t": "정규화 시간 $t_{reg}$  (0 = 움직임 시작, 1 = 시연 종료)",
            "p1": "$\\|\\Delta p\\|$ [m]",
            "p2": "$\\|\\Delta p_{xy}\\|$ [m]",
            "p3": "$\\theta$ [deg]",
            "p3_folded": "$\\min(\\theta, 180^{\\circ}\\!-\\!\\theta)$ [deg]",
            "p4": "④ 그리퍼 open / close 시점 (데모별)",
            "demo": "데모",
            "mean": "데모 평균 ± 표준편차",
            "open": "open (그리퍼 열기, +1)",
            "close": "close (그리퍼 닫기, −1)",
            "open_pt": "○ open 시점 (궤적 위)",
            "close_pt": "◆ close 시점 (궤적 위)",
            "cb": "데모 번호",
            "suptitle": "데모 결과 시각화 — {stem}  ({n} demos)",
            "p1_title": "① 그리퍼–트럭 거리 (월드 xyz, $\\Delta p=p_{truck}-p_{eef}$)",
            "p2_title": "② 그리퍼–박스 거리 (월드 xy 평면 — z는 높이축이라 제외)",
            "p3_title": "③ 그리퍼–트럭 쿼터니언 각도  $\\theta=2\\arccos(|q_{g}\\!\\cdot\\!q_{t}|)$",
            "p3_title_folded": "③ 그리퍼–트럭 쿼터니언 각도 (180° 기준 접힘)",
            "p4_title": "④ 그리퍼 open / close 시점",
        }
    return {
        "t": "normalized time $t_{reg}$  (0 = motion start, 1 = demo end)",
        "p1": "$\\|\\Delta p\\|$ [m]",
        "p2": "$\\|\\Delta p_{xy}\\|$ [m]",
        "p3": "$\\theta$ [deg]",
        "p3_folded": "$\\min(\\theta, 180^{\\circ}\\!-\\!\\theta)$ [deg]",
        "p4": "④ gripper open / close events (per demo)",
        "demo": "demo",
        "mean": "mean ± std over demos",
        "open": "open (+1)",
        "close": "close (−1)",
        "open_pt": "○ open moment (on curve)",
        "close_pt": "◆ close moment (on curve)",
        "cb": "demo index",
        "suptitle": "Demo visualization — {stem}  ({n} demos)",
        "p1_title": "① gripper–truck distance (world xyz, $\\Delta p=p_{truck}-p_{eef}$)",
        "p2_title": "② gripper–box distance (world xy plane, z = height excluded)",
        "p3_title": "③ gripper–truck quaternion angle  $\\theta=2\\arccos(|q_{g}\\!\\cdot\\!q_{t}|)$",
        "p3_title_folded": "③ gripper–truck quaternion angle (folded at 180°)",
        "p4_title": "④ gripper open / close events",
    }


def setup_fonts() -> bool:
    """한글 글리프를 그릴 수 있는 폰트를 찾아 rcParam 에 설정한다.

    Returns:
        한글 폰트를 찾았으면 True.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "Apple SD Gothic Neo"):
        if name in available:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name != "DejaVu Sans"
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


def resample_grid(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """정규화 시간축을 공통 그리드로 보간한다 (평균/표준편차 밴드용)."""
    if t.size < 2:
        return np.full_like(grid, float(y[0]) if y.size else np.nan)
    return np.interp(grid, t, y)


DEFAULT_FIGSIZE = (11.5, 13.5)


def screen_figsize(
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: float = 100.0,
    max_ratio_w: float = 0.94,
    max_ratio_h: float = 0.86,
) -> tuple[float, float]:
    """화면에 들어가는 figsize 를 계산한다(종횡비 유지, 해상도를 모르면 원래 크기)."""
    size = screen_size()
    if size is None:
        return figsize
    # 툴바(약 40 px)와 창 장식이 들어갈 자리를 빼고 계산한다.
    avail_h = max(size[1] * max_ratio_h - 40.0, 0.5 * size[1] * max_ratio_h)
    scale = min(
        1.0,
        (size[0] * max_ratio_w / dpi) / figsize[0],
        (avail_h / dpi) / figsize[1],
    )
    return (figsize[0] * scale, figsize[1] * scale)


OPEN_COLOR = "#1f77b4"   # ○ open 마커
CLOSE_COLOR = "#d62728"  # ◆ close 마커


def scatter_open(ax, xs, ys, size: float, lw: float = 1.6, zorder: float = 5) -> None:
    """open 시점 마커(속이 빈 파란 동그라미)를 찍는다."""
    ax.scatter(xs, ys, marker="o", s=size, facecolors="none", edgecolors=OPEN_COLOR,
               linewidths=lw, zorder=zorder)


def scatter_close(ax, xs, ys, size: float, lw: float = 0.6, zorder: float = 4) -> None:
    """close 시점 마커(빨간 다이아몬드)를 찍는다."""
    ax.scatter(xs, ys, marker="D", s=size, color=CLOSE_COLOR, edgecolors="white",
               linewidths=lw, zorder=zorder)


def event_points(ep: dict, y: np.ndarray, kind: str) -> tuple[list[float], list[float]]:
    """에피소드의 open/close 시점을 그 궤적 선 위 좌표 (t_reg, y) 로 변환한다.

    이벤트는 (정규화 시간, 종류, 원본 프레임 인덱스) 로 저장되어 있으므로, 궤적 배열의
    인덱스(= 프레임 - t0)로 같은 시점의 y 값을 찾는다.
    """
    xs, vs = [], []
    for t_ev, ev_kind, frame_idx in ep["events"]:
        if ev_kind != kind:
            continue
        k = int(frame_idx) - int(ep["t0"])
        if 0 <= k < y.size:
            xs.append(float(t_ev))
            vs.append(float(y[k]))
    return xs, vs


def build_figure(
    episodes: list[dict],
    labels: dict,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: float = 100.0,
    n_grid: int = 200,
    show_mean: bool = True,
    angle_mode: str = "geodesic",
    suptitle: str | None = None,
    height_ratios: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.35),
) -> "plt.Figure":
    """4-panel 그림을 만든다. 저장용과 화면용에서 같은 코드를 재사용한다."""
    n_ep = len(episodes)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(n_ep - 1, 1)) for i in range(n_ep)]
    grid = np.linspace(0.0, 1.0, int(n_grid))
    # 궤적이 많으면(rollout 100개 등) 선을 흐리게, ④ 마커를 작게 그린다.
    line_alpha = 0.75 if n_ep <= 30 else max(0.18, 20.0 / n_ep)
    band_alpha = 0.13 if n_ep <= 30 else 0.24
    s_open = 52 if n_ep <= 25 else max(9, int(1300 / n_ep))
    s_close = 42 if n_ep <= 25 else max(8, int(1050 / n_ep))

    fig, axes = plt.subplots(
        4, 1, figsize=figsize, dpi=dpi, sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": list(height_ratios)},  # ④ 는 궤적 수만큼 행을 그린다
    )
    # 컬러바 라벨이 그림 오른쪽 밖으로 잘리지 않도록 가장자리 여백을 조금 둔다
    fig.set_layout_engine("constrained", w_pad=0.10, h_pad=0.04)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5)

    # ---- 공통: 데모별 꺾은선 + 평균/표준편차 밴드 -----------------------------------
    # ① 은 3D norm(회전 불변이므로 유클리드 거리와 동일), ② 는 상대 도메인의 xy 성분 norm,
    # ③ 은 쿼터니언 내적으로 얻은 회전각[deg] (기본) 또는 180° 기준으로 접은 각도.
    theta_key = "theta_deg" if angle_mode == "geodesic" else "theta_folded_deg"
    theta_label = labels["p3"] if angle_mode == "geodesic" else labels["p3_folded"]
    theta_title = labels["p3_title"] if angle_mode == "geodesic" else labels["p3_title_folded"]
    # ① 은 close 시점, ② 는 open 시점을 각 궤적 선 위에 마커로 표시한다(④와 같은 디자인).
    series = [
        (axes[0], [np.linalg.norm(ep["d_truck"], axis=1) for ep in episodes], labels["p1"], labels["p1_title"], "close"),
        (axes[1], [np.linalg.norm(ep["d_box"][:, :2], axis=1) for ep in episodes], labels["p2"], labels["p2_title"], "open"),
        (axes[2], [ep[theta_key] for ep in episodes], theta_label, theta_title, None),
    ]
    for ax, ys, ylabel, title, mark_kind in series:
        stack = []
        for i, (ep, y) in enumerate(zip(episodes, ys)):
            y = np.asarray(y, dtype=np.float64)
            ax.plot(ep["t_reg"], y, color=colors[i], lw=1.1, alpha=line_alpha)
            stack.append(resample_grid(ep["t_reg"], y, grid))
            if mark_kind:  # 마커는 해당 데이터의 궤적 선 위(y 값)에 놓인다
                xs, vs = event_points(ep, y, mark_kind)
                if xs:
                    if mark_kind == "open":
                        scatter_open(ax, xs, vs, s_open, lw=1.0 if n_ep > 25 else 1.6)
                    else:
                        scatter_close(ax, xs, vs, s_close, lw=0.4 if n_ep > 25 else 0.6)
        if show_mean and n_ep > 1:
            stack = np.vstack(stack)
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            ax.fill_between(grid, mean - std, mean + std, color="0.25", alpha=band_alpha, lw=0)
            ax.plot(grid, mean, color="0.1", lw=2.2, ls="--", zorder=5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(True, alpha=0.30)
        ax.set_axisbelow(True)

    # 컬러바(데모 번호) + 평균선 범례
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(n_ep - 1, 1)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[0], pad=0.01, fraction=0.045)
    cb.set_label(labels["cb"])
    cb.set_ticks(sorted({0, n_ep // 4, n_ep // 2, (3 * n_ep) // 4, n_ep - 1}))

    handles = []
    if show_mean and n_ep > 1:
        handles.append(Line2D([], [], color="0.1", lw=2.2, ls="--", label=labels["mean"]))
    handles.append(Line2D([], [], marker="D", ls="none", mfc=CLOSE_COLOR, mec="white", ms=7,
                          label=labels["close_pt"]))
    axes[0].legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    axes[1].legend(
        handles=[Line2D([], [], marker="o", ls="none", mfc="none", mec=OPEN_COLOR, mew=1.6, ms=8,
                        label=labels["open_pt"])],
        loc="upper right", fontsize=9, framealpha=0.9,
    )

    # ---- 4번 패널: 그리퍼 open / close 이벤트 ---------------------------------------
    ax4 = axes[3]
    for i, ep in enumerate(episodes):
        ax4.plot([0.0, 1.0], [i, i], color="0.85", lw=1.0, zorder=1)
        opened = np.array([e[0] for e in ep["events"] if e[1] == "open"], dtype=float)
        closed = np.array([e[0] for e in ep["events"] if e[1] == "close"], dtype=float)
        if closed.size:
            scatter_close(ax4, closed, np.full(closed.shape, i), s_close, lw=0.4 if n_ep > 25 else 0.6)
        if opened.size:
            scatter_open(ax4, opened, np.full(opened.shape, i), s_open, lw=1.0 if n_ep > 25 else 1.6)
    ax4.set_yticks(np.arange(n_ep))
    ax4.set_yticklabels([f"{i}" for i in range(n_ep)], fontsize=7)
    ax4.tick_params(axis="y", length=2, pad=1)
    ax4.set_ylim(-0.8, n_ep - 0.2)
    ax4.set_ylabel(labels["demo"])
    ax4.set_xlabel(labels["t"])
    ax4.set_title(labels["p4_title"], loc="left", fontsize=10.5)
    ax4.set_xlim(-0.02, 1.02)
    ax4.grid(True, axis="x", alpha=0.30)
    ax4.set_axisbelow(True)
    ax4.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", mfc="none", mec="#1f77b4", mew=1.6, ms=8, label=labels["open"]),
            Line2D([], [], marker="D", ls="none", mfc="#d62728", mec="white", ms=7, label=labels["close"]),
        ],
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )

    # ④ 는 데모 수만큼 행이 있어서, 창이 작으면 tick 라벨이 빽빽해진다.
    # 실제 렌더링 크기를 재서 한 칸 걸러 표시한다(저장용 큰 그림에서는 그대로 전부 표시).
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        heights = [t.get_window_extent(rend).height for t in ax4.get_yticklabels() if t.get_text()]
        span = ax4.get_window_extent(rend).height
        step = 1
        if heights and span > 0:
            step = int(np.ceil(max(heights) * 1.15 * n_ep / span))
        step = max(1, min(step, max(1, n_ep // 2)))
        if step > 1:
            ticks = np.arange(0, n_ep, step)
            ax4.set_yticks(ticks)
            ax4.set_yticklabels([f"{i}" for i in ticks], fontsize=7)
    except Exception:  # noqa: BLE001 - 렌더러를 못 얻으면 그냥 전부 표시
        pass

    return fig


def plot_dataset(
    input_path: str,
    out_path: str,
    max_demos: int | None = 20,
    n_grid: int = 200,
    dpi: int = 150,
    show_mean: bool = True,
    demo_filter: list[str] | None = None,
    eps: float = 1e-6,
    angle_mode: str = "geodesic",
    show: bool = True,
    backend: str = "auto",
) -> dict:
    """hdf5 하나를 읽어 4-panel 그림을 저장하고(요청 시 창으로 표시) 요약 통계를 반환한다.

    Args:
        show: True 면 matplotlib 창을 띄워 화면에서 바로 볼 수 있게 한다(창을 닫으면 종료).
            디스플레이가 없으면 자동으로 저장만 한다.
        backend: "auto" 또는 사용할 matplotlib 백엔드 이름.
    """
    use_korean = setup_fonts()
    L = _labels(use_korean)
    if angle_mode not in ("geodesic", "folded"):
        raise ValueError(f"Unknown angle_mode: {angle_mode!r}")
    backend = select_backend(backend, allow_interactive=show)
    interactive = backend.lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template")

    episodes, skipped = [], []
    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"'{input_path}' 에 'data' 그룹이 없습니다.")
        keys = demo_keys(f["data"])
        if demo_filter:
            wanted = {str(x) for x in demo_filter}
            keys = [k for k in keys if k in wanted or k.split("_")[-1] in wanted]
        if max_demos is not None:
            keys = keys[: int(max_demos)]
        for k in keys:
            try:
                episodes.append(load_episode(f["data"][k], eps=eps))
            except Exception as exc:  # noqa: BLE001 - 데이터 결손은 건너뛰고 보고
                skipped.append(f"{k}: {exc}")

    if not episodes:
        raise RuntimeError(f"'{input_path}' 에서 시각화할 데모를 찾지 못했습니다.")

    n_ep = len(episodes)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    suptitle = L["suptitle"].format(stem=stem, n=n_ep)

    # 1) 저장용 그림: 화면 해상도와 무관하게 항상 같은 크기/해상도로 PNG 를 만든다
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig = build_figure(
        episodes, L, figsize=DEFAULT_FIGSIZE, dpi=100.0, n_grid=n_grid,
        show_mean=show_mean, angle_mode=angle_mode, suptitle=suptitle,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input": input_path,
        "output": out_path,
        "n_demos": n_ep,
        "domain": "world (Δp = p_target - p_eef)",
        "skipped": skipped,
        "episodes": episodes,
    }

    # ---- 콘솔 요약 -------------------------------------------------------------------
    print(f"[input ] {input_path}")
    print(f"[output] {out_path}")
    domain = "월드 프레임 두 점의 차 Δp = p_target - p_eef"
    print(f"[demos ] {n_ep}  (위치 오차 = {domain})")
    if skipped:
        print(f"[skip  ] {len(skipped)}")
        for s in skipped:
            print(f"         - {s}")
    print()
    hdr = f"{'demo':>8s} {'n':>5s} {'t0':>4s} {'d_truck@0':>10s} {'d_truck_min':>12s} {'theta@0':>8s} {'theta_min':>10s} {'open(t)':>16s} {'close(t)':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for ep in episodes:
        dt = np.linalg.norm(ep["d_truck"], axis=1)
        th = ep["theta_deg"]
        op = ",".join(f"{e[0]:.3f}" for e in ep["events"] if e[1] == "open")
        cl = ",".join(f"{e[0]:.3f}" for e in ep["events"] if e[1] == "close")
        print(
            f"{ep['name']:>8s} {ep['n']:5d} {ep['t0']:4d} {dt[0]:10.4f} {dt.min():12.4f} "
            f"{th[0]:8.2f} {th.min():10.2f} {op:>16s} {cl:>16s}"
        )
    print("-" * len(hdr))
    dt0 = np.array([np.linalg.norm(ep["d_truck"], axis=1)[0] for ep in episodes])
    dtm = np.array([np.linalg.norm(ep["d_truck"], axis=1).min() for ep in episodes])
    th0 = np.array([ep["theta_deg"][0] for ep in episodes])
    thm = np.array([ep["theta_deg"].min() for ep in episodes])
    print(f"{'mean':>8s} {'':5s} {'':4s} {dt0.mean():10.4f} {dtm.mean():12.4f} {th0.mean():8.2f} {thm.mean():10.2f}")
    print()
    print("· t_reg = 0 : actions[:, :6] 이 처음 0 이 아닌 프레임 (움직임 시작, idle 프리픽스 제거)")
    print("· t_reg = 1 : 각 데모의 마지막 프레임 (시연 종료)")
    print("· ① 은 월드 xyz 두 점의 차 Δp 의 3D norm(유클리드 거리)")
    print("· ② 는 같은 Δp 의 xy 성분 norm(월드 수평 거리, z=높이 제외), ③ 은 2·arccos(|q_g·q_t|)")
    print("· ④ 는 actions[:, 6] 부호 전환: +1 → ○ open, -1 → ◆ close")

    # ---- 화면 표시: 콘솔 요약을 출력한 뒤 창을 띄워 바로 볼 수 있게 한다 ------------
    if show and interactive:
        try:
            fig_ui = build_figure(
                episodes, L, figsize=screen_figsize(DEFAULT_FIGSIZE), dpi=100.0, n_grid=n_grid,
                show_mean=show_mean, angle_mode=angle_mode, suptitle=suptitle,
            )
            set_window_title(fig_ui, f"demo analysis — {stem} ({n_ep} demos)")
            print()
            print("[ui    ] matplotlib 창을 띄웁니다. 창을 닫으면 프로그램이 종료됩니다.", flush=True)
            plt.show()  # 창이 닫힐 때까지 대기
            summary["shown"] = True
            plt.close(fig_ui)
        except Exception as exc:  # noqa: BLE001 - 창 실패는 치명적이지 않다
            print(f"[warn  ] 창 표시 실패({exc}). PNG 로만 확인하세요.")
    elif show:
        print()
        print("[ui    ] GUI 백엔드를 쓸 수 없어 창을 띄우지 못했습니다. PNG 를 확인하세요.")
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="IsaacLab 데모 hdf5 결과 4-panel 시각화 (그리퍼-트럭/박스 거리, 쿼터니언 각도, 그리퍼 이벤트)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT, help="데모 hdf5 경로")
    ap.add_argument("--out", default=None, help="저장할 그림 경로 (기본: outputs/analysis_plots/<stem>_overview.png)")
    ap.add_argument("--max-demos", type=int, default=20, help="사용할 최대 데모 수 (0 이하면 전체)")
    ap.add_argument("--demos", default=None, help="쉼표로 구분한 데모 선택 (예: 0,3,demo_7)")
    ap.add_argument("--angle-mode", choices=("geodesic", "folded"), default="geodesic",
                    help="③ 패널: geodesic=θ=2·arccos(|q_g·q_t|) 그대로, folded=180° 기준으로 접은 각도")
    ap.add_argument("--grid-n", type=int, default=200, help="평균/표준편차 밴드용 보간 그리드 수")
    ap.add_argument("--dpi", type=int, default=150, help="저장 해상도")
    ap.add_argument("--no-show", action="store_true",
                    help="matplotlib 창을 띄우지 않고 PNG 저장만 한다(기본: 창 표시)")
    ap.add_argument("--backend", default="auto",
                    help="matplotlib 백엔드 지정 (auto, TkAgg, QtAgg, Agg ...)")
    ap.add_argument("--no-mean", action="store_true", help="평균 ± 표준편차 밴드를 그리지 않는다")
    ap.add_argument("--eps", type=float, default=1e-6, help="움직임 시작 판정 임계값")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"[error] 입력 파일이 없습니다: {args.input}")
        return 2
    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        out = os.path.join(DEFAULT_OUTDIR, f"{stem}_overview.png")
    max_demos = None if args.max_demos is None or args.max_demos <= 0 else args.max_demos
    demos = [s.strip() for s in args.demos.split(",")] if args.demos else None
    plot_dataset(
        input_path=args.input,
        out_path=out,
        max_demos=max_demos,
        n_grid=args.grid_n,
        dpi=args.dpi,
        show_mean=not args.no_mean,
        demo_filter=demos,
        eps=args.eps,
        angle_mode=args.angle_mode,
        show=not args.no_show,
        backend=args.backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
