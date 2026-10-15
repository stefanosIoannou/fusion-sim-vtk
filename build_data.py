"""Convert the six starter-kit shots into compact JS payloads for the vtk.js viewer.

Usage:  .venv/bin/python build_data.py            (reads data/raw/*.parquet, writes web/data/)

Each shot becomes web/data/<id>.js, loaded by the browser with a plain <script> tag so the
viewer works straight from file:// with no server.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter-kit"))
from data_fixes import fix_d3d_ip_times  # noqa: E402  (numpy-only helper)

RAW = ROOT / "data" / "raw"
OUT = ROOT / "web" / "data"
PSIN_LO, PSIN_HI = -0.1, 4.0   # quantisation range for normalised flux
TRACE_POINTS = 600


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def bilinear(psi, R, Z, r, z):
    """Sample a (nZ, nR) grid at points (r, z) on a uniform grid."""
    fr = np.clip((np.asarray(r) - R[0]) / (R[1] - R[0]), 0, len(R) - 1.0001)
    fz = np.clip((np.asarray(z) - Z[0]) / (Z[1] - Z[0]), 0, len(Z) - 1.0001)
    i, j = fz.astype(int), fr.astype(int)
    tz, tr = fz - i, fr - j
    return ((1 - tz) * (1 - tr) * psi[i, j] + (1 - tz) * tr * psi[i, j + 1]
            + tz * (1 - tr) * psi[i + 1, j] + tz * tr * psi[i + 1, j + 1])


def binned_trace(t, y, t0, t1, n=TRACE_POINTS):
    """Bin-averaged trace on [t0, t1] for plotting (raw signals have up to ~500k samples)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    edges = np.linspace(t0, t1, n + 1)
    idx = np.digitize(t, edges) - 1
    ok = (idx >= 0) & (idx < n) & np.isfinite(y)
    s = np.bincount(idx[ok], y[ok], n)
    c = np.bincount(idx[ok], minlength=n)
    tc = 0.5 * (edges[1:] + edges[:-1])
    with np.errstate(invalid="ignore"):
        m = s / c
    good = c > 0
    return np.interp(tc, tc[good], m[good]) if good.any() else np.zeros(n)


def coil_polygon(r, z, w, h, a1, a2):
    """EFIT-style coil cross-section: rectangle, or parallelogram when AF1/AF2 != 0."""
    if a1 == 0 and a2 == 0:
        return [(r - w / 2, z - h / 2), (r + w / 2, z - h / 2), (r + w / 2, z + h / 2), (r - w / 2, z + h / 2)]
    if a2 == 0:
        s = w / 2 * np.tan(np.radians(a1))
        return [(r - w / 2, z - h / 2 - s), (r + w / 2, z - h / 2 + s), (r + w / 2, z + h / 2 + s), (r - w / 2, z + h / 2 - s)]
    s = h / 2 / np.tan(np.radians(a2))
    return [(r - w / 2 - s, z - h / 2), (r + w / 2 - s, z - h / 2), (r + w / 2 + s, z + h / 2), (r - w / 2 + s, z + h / 2)]


def pretty_coil(col):
    name = col.replace("magnetics_", "").replace("_current", "")
    return {"sol": "Solenoid", "ECOILA": "E-coil (OH)"}.get(name, name.upper())


def build(path):
    row = pq.read_table(path).to_pylist()[0]
    machine = row["source"]
    shot_id = path.stem
    times = np.asarray(row["efit_times"], float)
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    psi = np.asarray(row["efit_psirz"], float)              # (T, nZ, nR)
    r_ax = np.asarray(row["efit_r_axis"], float)
    z_ax = np.asarray(row["efit_z_axis"], float)
    n_lcfs = np.asarray(row["efit_lcfs_n"], int)
    lcfs_r = np.asarray(row["efit_lcfs_r"], float)
    lcfs_z = np.asarray(row["efit_lcfs_z"], float)

    # Normalised flux: 0 on the magnetic axis, 1 on the last closed flux surface.
    psin = np.empty_like(psi)
    lcfs = []
    for k in range(len(times)):
        lr, lz = lcfs_r[k, :n_lcfs[k]], lcfs_z[k, :n_lcfs[k]]
        p_ax = bilinear(psi[k], R, Z, r_ax[k], z_ax[k])
        p_b = np.mean(bilinear(psi[k], R, Z, lr, lz))
        psin[k] = (psi[k] - p_ax) / (p_b - p_ax)
        lcfs.append({"r": np.round(lr, 4).tolist(), "z": np.round(lz, 4).tolist()})
    psin = np.nan_to_num(psin, nan=PSIN_HI)
    q = np.round((np.clip(psin, PSIN_LO, PSIN_HI) - PSIN_LO) / (PSIN_HI - PSIN_LO) * 65535).astype(np.uint16)

    # Coil currents: per circuit, sampled at EFIT times plus a binned trace for the plot.
    mt = np.asarray(row["magnetics_time"], float)
    t0, t1 = times[0], times[-1]
    circuits = list(dict.fromkeys(row["coil_input_column"]))
    circ_out = []
    for col in circuits:
        y = np.asarray(row[col], float)
        circ_out.append({
            "name": pretty_coil(col),
            "atEfit": np.round(np.interp(times, mt, y), 3).tolist(),
            "trace": np.round(binned_trace(mt, y, t0, t1), 3).tolist(),
        })
    coils = []
    for i, col in enumerate(row["coil_input_column"]):
        poly = coil_polygon(row["coil_R"][i], row["coil_Z"][i], row["coil_width"][i],
                            row["coil_height"][i], row["coil_angle1"][i], row["coil_angle2"][i])
        coils.append({"c": circuits.index(col), "p": np.round(np.ravel(poly), 4).tolist()})

    ip_t = fix_d3d_ip_times(row) if machine == "DIII-D" else np.asarray(row["magnetics_plasma_current_times"]
                                                                       if "magnetics_plasma_current_times" in row else mt)
    ip = np.asarray(row["magnetics_plasma_current"], float)
    tf_col = "magnetics_bcoil" if machine == "DIII-D" else "magnetics_tf_current"

    def scalar(key):
        return np.round(np.nan_to_num(np.asarray(row[key], float)), 4).tolist() if key in row else None

    return {
        "id": shot_id,
        "machine": machine,
        "shot": int(shot_id.rsplit("_", 1)[1]),
        "times": times.tolist(),
        "R": R.tolist(), "Z": Z.tolist(),
        "psin": {"b64": b64(q, "<u2"), "lo": PSIN_LO, "hi": PSIN_HI},
        "lcfs": lcfs,
        "axis": {"r": np.round(r_ax, 4).tolist(), "z": np.round(z_ax, 4).tolist()},
        "scalars": {"q95": scalar("efit_q95"), "betaN": scalar("efit_beta_n"), "li": scalar("efit_li")},
        "ip": {"atEfit": np.round(np.interp(times, ip_t, ip), 2).tolist(),
               "trace": np.round(binned_trace(ip_t, ip, t0, t1), 2).tolist()},
        "tf": {"atEfit": np.round(np.interp(times, mt, np.asarray(row[tf_col], float)), 2).tolist()},
        "traceTimes": np.round(np.linspace(t0, t1, TRACE_POINTS + 1)[:-1] + (t1 - t0) / TRACE_POINTS / 2, 2).tolist(),
        "circuits": circ_out,
        "coils": coils,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for path in sorted(RAW.glob("*.parquet")):
        d = build(path)
        js = f"window.FusionScope.register({json.dumps(d, separators=(',', ':'))});\n"
        (OUT / f"{d['id']}.js").write_text(js)
        index.append({"id": d["id"], "machine": d["machine"], "shot": d["shot"], "frames": len(d["times"]),
                      "t0": d["times"][0], "t1": d["times"][-1]})
        print(f"{d['id']:>18}  {d['machine']:6}  {len(d['times']):4} frames  {len(d['coils']):4} coil elements"
              f"  {len(js) / 1e6:.1f} MB")
    (OUT / "index.js").write_text(f"window.FUSION_INDEX = {json.dumps(index)};\n")


if __name__ == "__main__":
    main()
