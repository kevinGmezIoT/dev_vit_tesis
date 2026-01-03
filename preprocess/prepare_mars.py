import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def rel_to_out_root(out_root: Path, path: Path) -> str:
    return path.resolve().relative_to(out_root.resolve()).as_posix()


def parse_mars_filename(fname: str) -> Optional[Tuple[str, int, int, int]]:
    """
    Parse de: 0001C1T0001F001.jpg
    Retorna: (pid_str, cam_int, tracklet_int, frame_int)
    """
    m = re.match(
        r"(?P<pid>\d+)"
        r"C(?P<cam>\d+)"
        r"T(?P<trk>\d+)"
        r"F(?P<frm>\d+)\.jpg$",
        fname,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group("pid"), int(m.group("cam")), int(m.group("trk")), int(m.group("frm"))


def mars_pid_to_iddir(pid_str: str) -> str:
    return f"ID{int(pid_str):04d}"


def cam_to_cxxx(cam_int: int) -> str:
    return f"c{cam_int:03d}"


def tracklet_to_scene(tracklet_id: int, width: int = 4) -> str:
    """
    Convierte tracklet a scene:
      T0001 -> S0001 (width=4)
    """
    return f"S{tracklet_id:0{width}d}"


def build_cycles_from_sorted_frames(
    frames: List[Tuple[int, Path]],
    seq_length: int,
    overlap: int,
) -> List[List[Tuple[int, Path]]]:
    if len(frames) < seq_length:
        return []
    step = seq_length - overlap
    if step <= 0:
        step = 1
    cycles: List[List[Tuple[int, Path]]] = []
    for i in range(0, len(frames) - seq_length + 1, step):
        chunk = frames[i:i + seq_length]
        if len(chunk) == seq_length:
            cycles.append(chunk)
    return cycles


def copy_cycle_rgb(
    cycle: List[Tuple[int, Path]],
    dst_cycle_dir: Path,
    overwrite: bool,
) -> Tuple[List[int], List[Path]]:
    dst_cycle_dir.mkdir(parents=True, exist_ok=True)
    frame_nums: List[int] = []
    dst_paths: List[Path] = []

    for fn, src in cycle:
        dst = dst_cycle_dir / f"{fn:06d}.jpg"
        if overwrite or (not dst.exists()):
            shutil.copy2(src, dst)
        frame_nums.append(fn)
        dst_paths.append(dst)

    return frame_nums, dst_paths


def main():
    ap = argparse.ArgumentParser(description="Organiza MARS a estructura tipo tu dataset (solo RGB) y genera CSVs.")
    ap.add_argument("--raw-root", required=True, help="Raíz MARS crudo (contiene bbox_train, bbox_test).")
    ap.add_argument("--out-root", required=True, help="Raíz de salida (crea rgb/ y metadata/).")
    ap.add_argument("--sequence-len", type=int, default=25)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-existing-cycles", action="store_true")

    # Esta opción controla el padding del scene: S0001 vs S01
    ap.add_argument("--scene-width", type=int, default=4, help="Ancho numérico de escena derivada de tracklet (default=4).")

    # Debug/test
    ap.add_argument("--debug", action="store_true", help="Procesa solo los primeros 2 IDs por split.")
    ap.add_argument("--limit-cycles", type=int, default=None, help="Limita total de ciclos generados.")

    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"No existe raw-root: {raw_root}")

    rgb_out_root = out_root / "rgb"
    meta_out_root = out_root / "metadata"
    meta_out_root.mkdir(parents=True, exist_ok=True)

    sequences_rows: List[Dict] = []
    frames_rows: List[Dict] = []
    total_cycles = 0

    splits = [("bbox_train", "train"), ("bbox_test", "test")]

    for split_dirname, split_name in splits:
        split_dir = raw_root / split_dirname
        if not split_dir.is_dir():
            print(f"[WARN] No existe {split_dir}, se omite.")
            continue

        person_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name))
        if args.debug:
            person_dirs = person_dirs[:2]
            print(f"[DEBUG] {split_name}: procesando solo {len(person_dirs)} IDs.")

        print(f"\nProcesando split {split_name} ({split_dirname}) ...")

        for pid_dir in tqdm(person_dirs, desc=f"{split_name} IDs", unit="id"):
            tracklets: Dict[Tuple[int, int], List[Tuple[int, Path]]] = {}

            jpgs = sorted(pid_dir.glob("*.jpg"), key=lambda p: natural_sort_key(p.name))
            for img_path in jpgs:
                parsed = parse_mars_filename(img_path.name)
                if parsed is None:
                    continue
                pid_str, cam_int, trk_int, frm_int = parsed
                tracklets.setdefault((cam_int, trk_int), []).append((frm_int, img_path))

            if not tracklets:
                continue

            id_dirname = mars_pid_to_iddir(pid_dir.name)  # pid_dir.name suele ser "0001"
            person_global_id = int(pid_dir.name)

            for (cam_int, trk_int), frames in tracklets.items():
                frames_sorted = sorted(frames, key=lambda x: x[0])
                cycles = build_cycles_from_sorted_frames(
                    frames=frames_sorted,
                    seq_length=args.sequence_len,
                    overlap=args.overlap,
                )
                if not cycles:
                    continue

                cam = cam_to_cxxx(cam_int)
                scene = tracklet_to_scene(trk_int, width=args.scene_width)  # <-- CAMBIO CLAVE

                for cyc_idx, cycle in enumerate(cycles, start=1):
                    total_cycles += 1
                    cycle_dir = f"cycle{cyc_idx:04d}"
                    dst_cycle_dir = rgb_out_root / id_dirname / scene / cam / cycle_dir

                    if args.skip_existing_cyles if False else False:
                        pass  # no-op para evitar typos accidentales

                    if args.skip_existing_cycles and dst_cycle_dir.exists():
                        if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
                            break
                        continue

                    frame_nums, dst_paths = copy_cycle_rgb(
                        cycle=cycle,
                        dst_cycle_dir=dst_cycle_dir,
                        overwrite=args.overwrite,
                    )

                    sequence_id = f"{id_dirname}_{scene}_{cam}_cyc{cyc_idx:04d}"

                    sequences_rows.append({
                        "sequence_id": sequence_id,
                        "person_global_id": person_global_id,
                        "dataset": "MARS",
                        "split": split_name,  # si luego lo quieres quitar, lo elimino
                        "scene": scene,
                        "cam_id": cam,
                        "tracklet_id": trk_int,
                        "cycle_index": cyc_idx,
                        "rgb_dir": rel_to_out_root(out_root, dst_cycle_dir),
                        "num_frames": len(frame_nums),
                        "frame_start": int(frame_nums[0]),
                        "frame_end": int(frame_nums[-1]),
                        "frame_numbers_json": json.dumps(frame_nums, ensure_ascii=False),
                    })

                    for fn, dst in zip(frame_nums, dst_paths):
                        frames_rows.append({
                            "sequence_id": sequence_id,
                            "person_global_id": person_global_id,
                            "scene": scene,
                            "cam_id": cam,
                            "tracklet_id": trk_int,
                            "cycle_index": cyc_idx,
                            "frame_number": int(fn),
                            "rgb_path": rel_to_out_root(out_root, dst),
                        })

                    if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
                        break

                if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
                    break

            if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
                break

        if args.limit_cycles is not None and total_cycles >= int(args.limit_cycles):
            break

    if not sequences_rows:
        print("\n[WARN] No se generaron secuencias. Revisa que bbox_train/bbox_test existan y los nombres correspondan.")
        return

    seq_df = pd.DataFrame(sequences_rows)
    frm_df = pd.DataFrame(frames_rows)

    seq_csv = meta_out_root / "sequences.csv"
    frm_csv = meta_out_root / "sequence_frames.csv"

    seq_df.to_csv(seq_csv, index=False, encoding="utf-8")
    frm_df.to_csv(frm_csv, index=False, encoding="utf-8")

    print("\nOK:")
    print(f"- RGB root:            {rgb_out_root}")
    print(f"- sequences.csv:       {seq_csv} (secuencias: {len(seq_df)})")
    print(f"- sequence_frames.csv: {frm_csv} (frames: {len(frm_df)})")


if __name__ == "__main__":
    main()
