#!/usr/bin/env python3
"""롤아웃(.npz 폴더) 결과를 4-panel 로 시각화한다.

`analysis_plots_datasets.py`(hdf5 데이터셋 버전)의 계산·플롯 코드를 그대로 재사용하고,
입력만 "rollout_*.npz 가 담긴 폴더"로 바꾼 버전이다.

    기존: datasets/team10_lee_s3.hdf5  (data/demo_0 ... demo_19, 20개 시연)
    여기: outputs/rollouts/gepl_s101/  (rollout_000.npz ... rollout_099.npz, 100개)

각 npz 는 hdf5 에피소드의 `obs` 그룹과 같은 배열을 담고 있다
(`actions`, `eef_pos`, `eef_quat`, `gripper_pos`, `toy_truck_positions`,
`toy_truck_orientations`, `box_positions`, `box_orientations`, `grasp_err`, `success`).

그리는 변수는 데이터셋 버전과 동일하다 (x 축 = 정규화 시간 t_reg = 0..1).
    1) 그리퍼-트럭 위치 오차 norm  : 월드 xyz, ‖Δp‖ = ‖p_truck - p_eef‖
    2) 그리퍼-박스 위치 오차 norm  : 월드 xy 평면 (z = 높이축 제외)
    3) 그리퍼-트럭 자세 오차 norm  : θ = 2·arccos(|q_gripper · q_truck|)  [deg]
    4) 그리퍼 open / close 시점    : open = ○, close = ◆  (행 = 롤아웃)
추가로 ① 의 각 궤적 선 위에 그 롤아웃의 close 시점(◆), ② 의 각 궤적 선 위에
open 시점(○)이 같은 마커 디자인으로 찍히고, ④ 의 각 행 배경색은 그 롤아웃의
성공/실패를 나타낸다(초록 = success).

사용 예 (창을 닫으면 종료)
    ./isaaclab.sh -p tools_c/analysis_plots_rollouts.py
    python tools_c/analysis_plots_rollouts.py --dir outputs/rollouts/gepl_s111
    python tools_c/analysis_plots_rollouts.py --only failure --max-rollouts 30
    python tools_c/analysis_plots_rollouts.py --no-show          # PNG 만 저장
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# 롤아웃 시각화는 데이터셋 버전의 함수(로딩/계산/플롯)를 그대로 쓴다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import analysis_plots_datasets as ds
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_DIR = os.path.join(REPO, "outputs", "rollouts", "gepl_s101")
DEFAULT_OUTDIR = os.path.join(REPO, "outputs", "analysis_plots")

# 롤아웃은 개수가 많아(예: 100개) ④ 패널에 세로 공간을 더 준다.
ROLLOUT_HEIGHT_RATIOS = (1.0, 1.0, 1.0, 2.2)

SUCCESS_COLOR = "#2ca02c"
FAILURE_COLOR = "#d62728"


# --------------------------------------------------------------------------------------
# 입력 폴더 → 에피소드 목록
# --------------------------------------------------------------------------------------
class NpzEpisode:
    """`ds.load_episode()` 를 그대로 쓰기 위한 h5py Group 흉내내기 어댑터.

    `ds.load_episode` 는 `ep["actions"]`, `ep["obs/eef_pos"]`, `ep.attrs`, `ep.name` 만
    사용하므로, npz 배열을 같은 방식으로 꺼내주면 로딩/계산 코드를 재사용할 수 있다.
    """

    def __init__(self, path: str, prefix: str = ""):
        with np.load(path) as z:
            self._data = {k: z[k] for k in z.files}
        stem = os.path.splitext(os.path.basename(path))[0]
        self.name = f"{prefix}/{stem}" if prefix else stem  # load_episode 는 '/' 뒤만 사용
        self.attrs: dict = {}
        self.path = path
        self.success = bool(self._data.get("success", False))

    def __contains__(self, key: str) -> bool:
        return self._key(key) in self._data

    def __getitem__(self, key: str):
        return self._data[self._key(key)]

    @staticmethod
    def _key(key: str) -> str:
        return key[4:] if key.startswith("obs/") else key


def rollout_files(folder: str) -> list[str]:
    """폴더 안의 *.npz 를 파일명 숫자 기준으로 자연 정렬해 반환한다."""
    files = glob.glob(os.path.join(folder, "*.npz"))

    def _idx(path: str):
        m = re.search(r"(\d+)", os.path.basename(path))
        return (int(m.group(1)) if m else 1 << 30, os.path.basename(path))

    return sorted(files, key=_idx)


def load_rollouts(
    folder: str,
    only: str = "all",
    max_rollouts: int | None = None,
    eps: float = 1e-6,
) -> tuple[list[dict], list[bool], list[str]]:
    """폴더에서 롤아웃을 읽어 (에피소드 목록, 성공 플래그, 건너뛴 파일) 을 반환한다."""
    files = rollout_files(folder)
    if not files:
        raise FileNotFoundError(f"'{folder}' 에 .npz 파일이 없습니다.")

    prefix = os.path.basename(os.path.normpath(folder))
    episodes, ok_flags, skipped = [], [], []
    for path in files:
        try:
            ep = NpzEpisode(path, prefix)
            if only == "success" and not ep.success:
                continue
            if only == "failure" and ep.success:
                continue
            rec = ds.load_episode(ep, eps=eps)  # 데이터셋 버전과 동일한 계산
        except Exception as exc:  # noqa: BLE001 - 깨진 npz 는 건너뛰고 보고
            skipped.append(f"{os.path.basename(path)}: {exc}")
            continue
        episodes.append(rec)
        ok_flags.append(ep.success)
        if max_rollouts is not None and max_rollouts > 0 and len(episodes) >= max_rollouts:
            break

    if not episodes:
        raise RuntimeError(f"'{folder}' 에서 시각화할 롤아웃을 찾지 못했습니다(only={only}).")
    return episodes, ok_flags, skipped


# --------------------------------------------------------------------------------------
# 라벨 / 성공-실패 표시
# --------------------------------------------------------------------------------------
def rollout_labels(use_korean: bool) -> dict:
    """데이터셋 버전 라벨을 롤아웃 용어로 바꾼다."""
    labels = dict(ds._labels(use_korean))
    if use_korean:
        labels.update(
            {
                "demo": "롤아웃",
                "cb": "롤아웃 번호",
                "mean": "롤아웃 평균 ± 표준편차",
                "p4_title": "④ 그리퍼 open / close 시점  (행 = 롤아웃, ○ open / ◆ close, 초록 = 성공 / 빨강 = 실패)",
                "succ_ok": "success (성공)",
                "succ_ng": "failure (실패)",
            }
        )
    else:
        labels.update(
            {
                "demo": "rollout",
                "cb": "rollout index",
                "mean": "mean ± std over rollouts",
                "p4_title": "④ gripper open / close events  (row = rollout, ○ open / ◆ close, green = success / red = failure)",
                "succ_ok": "success",
                "succ_ng": "failure",
            }
        )
    return labels


def mark_success(ax4, ok_flags: list[bool]) -> None:
    """④ 패널의 각 행을 성공(초록)/실패(빨강)로 칠한다.

    행이 수십~수백 개라 범례 상자는 데이터를 가리므로, 표시는 행 색으로만 하고
    의미는 패널 제목에 적는다(build_figure 가 만든 범례는 제거).
    """
    n = len(ok_flags)
    lw = 4.5 if n <= 30 else 2.2  # 행 간격이 좁으면 얇게
    for i, ok in enumerate(ok_flags):
        ax4.plot(
            [0.0, 1.0], [i, i],
            color=SUCCESS_COLOR if ok else FAILURE_COLOR,
            alpha=0.22 if n <= 30 else 0.45,
            lw=lw, solid_capstyle="butt", zorder=0.5,
        )
    legend = ax4.get_legend()
    if legend is not None:
        legend.remove()


# --------------------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------------------
def plot_rollouts(
    folder: str,
    out_path: str,
    only: str = "all",
    max_rollouts: int | None = None,
    n_grid: int = 200,
    dpi: int = 150,
    show_mean: bool = True,
    angle_mode: str = "geodesic",
    show: bool = True,
    backend: str = "auto",
    eps: float = 1e-6,
) -> dict:
    """롤아웃 폴더를 읽어 4-panel 그림을 저장하고(요청 시 창으로 표시) 요약을 반환한다."""
    use_korean = ds.setup_fonts()
    labels = rollout_labels(use_korean)
    backend = ds.select_backend(backend, allow_interactive=show)
    interactive = backend.lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template")

    episodes, ok_flags, skipped = load_rollouts(folder, only=only, max_rollouts=max_rollouts, eps=eps)
    n_ep = len(episodes)
    succ_rate = float(np.mean(ok_flags)) if ok_flags else 0.0
    stem = os.path.basename(os.path.normpath(folder))
    suptitle = (
        f"롤아웃 결과 시각화 — {stem}  ({n_ep} rollouts, success {succ_rate:.0%})"
        if use_korean
        else f"Rollout visualization — {stem}  ({n_ep} rollouts, success {succ_rate:.0%})"
    )
    common = dict(
        dpi=100.0, n_grid=n_grid, show_mean=show_mean, angle_mode=angle_mode,
        suptitle=suptitle, height_ratios=ROLLOUT_HEIGHT_RATIOS,
    )

    # 1) 저장용 그림: 화면 해상도와 무관하게 항상 같은 크기/해상도로 PNG 를 만든다
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig = ds.build_figure(episodes, labels, figsize=ds.DEFAULT_FIGSIZE, **common)
    mark_success(fig.axes[3], ok_flags)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dir": folder,
        "output": out_path,
        "n_rollouts": n_ep,
        "success_rate": succ_rate,
        "only": only,
        "skipped": skipped,
        "episodes": episodes,
    }

    # ---- 콘솔 요약 -------------------------------------------------------------------
    print(f"[dir   ] {folder}")
    print(f"[output] {out_path}")
    print(f"[rolls ] {n_ep}  (only={only}, success {succ_rate:.1%})")
    if skipped:
        print(f"[skip  ] {len(skipped)}")
        for s in skipped:
            print(f"         - {s}")
    print()
    hdr = (
        f"{'rollout':>12s} {'n':>5s} {'t0':>4s} {'ok':>3s} {'d_truck@0':>10s} {'d_truck_min':>12s}"
        f" {'theta@0':>8s} {'theta_min':>10s} {'open(t)':>20s} {'close(t)':>20s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ep, ok in zip(episodes, ok_flags):
        dt = np.linalg.norm(ep["d_truck"], axis=1)
        th = ep["theta_deg"]
        op = ",".join(f"{e[0]:.3f}" for e in ep["events"] if e[1] == "open")
        cl = ",".join(f"{e[0]:.3f}" for e in ep["events"] if e[1] == "close")
        print(
            f"{ep['name']:>12s} {ep['n']:5d} {ep['t0']:4d} {('O' if ok else 'X'):>3s} "
            f"{dt[0]:10.4f} {dt.min():12.4f} {th[0]:8.2f} {th.min():10.2f} {op:>20s} {cl:>20s}"
        )
    print("-" * len(hdr))
    dt0 = np.array([np.linalg.norm(ep["d_truck"], axis=1)[0] for ep in episodes])
    dtm = np.array([np.linalg.norm(ep["d_truck"], axis=1).min() for ep in episodes])
    th0 = np.array([ep["theta_deg"][0] for ep in episodes])
    thm = np.array([ep["theta_deg"].min() for ep in episodes])
    print(
        f"{'mean':>12s} {'':5s} {'':4s} {'':>3s} {dt0.mean():10.4f} {dtm.mean():12.4f} "
        f"{th0.mean():8.2f} {thm.mean():10.2f}"
    )
    for tag, sel in (("success", np.array(ok_flags, dtype=bool)), ("failure", ~np.array(ok_flags, dtype=bool))):
        if sel.any():
            print(
                f"{tag:>12s} {'':5s} {'':4s} {'':>3s} {dt0[sel].mean():10.4f} {dtm[sel].mean():12.4f} "
                f"{th0[sel].mean():8.2f} {thm[sel].mean():10.2f}   (n={int(sel.sum())})"
            )
    n_no_close = sum(1 for ep in episodes if not any(e[1] == "close" for e in ep["events"]))
    n_no_open = sum(1 for ep in episodes if not any(e[1] == "open" for e in ep["events"]))
    print(f"[events] close 신호 없는 롤아웃 {n_no_close}/{n_ep}, open 신호 없는 롤아웃 {n_no_open}/{n_ep}")
    print()
    print("· 입력 = rollout_*.npz 폴더, 계산/플롯은 analysis_plots_datasets 와 동일한 코드")
    print("· t_reg = 0 : arm action 이 처음 0 이 아닌 프레임, t_reg = 1 : 롤아웃 마지막 프레임")
    print("· ① 은 월드 xyz 두 점의 차 Δp 의 3D norm, ② 는 Δp 의 xy 성분 norm(수평 거리)")
    print("· ③ 은 2·arccos(|q_g·q_t|, 월드 축), ④ 는 actions[:, 6] 부호 전환 (+1 = ○ open, -1 = ◆ close)")
    print("· ④ 의 행 색: 초록 = success, 빨강 = failure  (--only success|failure 로 분리 가능)")

    # ---- 화면 표시: 콘솔 요약을 출력한 뒤 창을 띄워 바로 볼 수 있게 한다 ------------
    if show and interactive:
        try:
            fig_ui = ds.build_figure(episodes, labels, figsize=ds.screen_figsize(ds.DEFAULT_FIGSIZE), **common)
            mark_success(fig_ui.axes[3], ok_flags)
            ds.set_window_title(fig_ui, f"rollout analysis — {stem} ({n_ep} rollouts)")
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
        description="롤아웃(.npz 폴더) 4-panel 시각화 — 그리퍼-트럭/박스 거리, 쿼터니언 각도, 그리퍼 이벤트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("dir", nargs="?", default=DEFAULT_DIR, help="rollout_*.npz 폴더")
    ap.add_argument("--out", default=None,
                    help="저장할 그림 경로 (기본: outputs/analysis_plots/<folder>_overview.png)")
    ap.add_argument("--only", choices=("all", "success", "failure"), default="all", help="성공/실패 롤아웃만 보기")
    ap.add_argument("--max-rollouts", type=int, default=0, help="사용할 최대 롤아웃 수 (0 이하면 전체)")
    ap.add_argument("--angle-mode", choices=("geodesic", "folded"), default="geodesic",
                    help="③ 패널: geodesic=θ=2·arccos(|q_g·q_t|) 그대로, folded=180° 기준으로 접은 각도")
    ap.add_argument("--grid-n", type=int, default=200, help="평균/표준편차 밴드용 보간 그리드 수")
    ap.add_argument("--dpi", type=int, default=150, help="저장 해상도")
    ap.add_argument("--no-show", action="store_true", help="matplotlib 창을 띄우지 않고 PNG 저장만 한다")
    ap.add_argument("--backend", default="auto", help="matplotlib 백엔드 지정 (auto, TkAgg, QtAgg, Agg ...)")
    ap.add_argument("--no-mean", action="store_true", help="평균 ± 표준편차 밴드를 그리지 않는다")
    ap.add_argument("--eps", type=float, default=1e-6, help="움직임 시작 판정 임계값")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isdir(args.dir):
        print(f"[error] 폴더가 없습니다: {args.dir}")
        return 2
    out = args.out
    if out is None:
        stem = os.path.basename(os.path.normpath(args.dir))
        out = os.path.join(DEFAULT_OUTDIR, f"{stem}_overview.png")
    max_rollouts = None if args.max_rollouts is None or args.max_rollouts <= 0 else args.max_rollouts
    plot_rollouts(
        folder=args.dir,
        out_path=out,
        only=args.only,
        max_rollouts=max_rollouts,
        n_grid=args.grid_n,
        dpi=args.dpi,
        show_mean=not args.no_mean,
        angle_mode=args.angle_mode,
        show=not args.no_show,
        backend=args.backend,
        eps=args.eps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
