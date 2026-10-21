"""
petro.py - core petrophysics for wireline log interpretation.

Pipeline: load LAS -> QC (caliper / washout) -> Vsh -> porosity -> Sw (Archie)
          -> cutoffs -> net pay summary.

All equations are standard textbook petrophysics (Archie 1942, Larionov 1969).
"""
from __future__ import annotations

import lasio
import numpy as np
import pandas as pd

# Candidate mnemonics so the tool works on different LAS files.
CURVE_ALIASES = {
    "gr": ["GR", "GAMN", "GAMMA", "GRC", "GR_EDTC"],
    "rho": ["RHOB", "DEN", "DFAR", "RHOZ", "DENS"],
    "nphi": ["NPHI", "NEUT", "TNPH", "NPOR"],
    "rt": ["RT", "ILD", "LLD", "RDEP", "AT90", "PR", "RES_DEEP"],
    "sp": ["SP"],
    "cali": ["CALI", "CAL", "HCAL", "C1"],
}


def load_las(path_or_file) -> pd.DataFrame:
    """Read a LAS file into a DataFrame indexed by depth. Nulls become NaN."""
    las = lasio.read(path_or_file)
    df = las.df()
    df.index.name = "DEPTH"
    return df


def guess_curves(df: pd.DataFrame) -> dict:
    """Map our roles (gr, rho, rt...) to whatever the LAS file calls them."""
    cols_upper = {c.upper(): c for c in df.columns}
    found = {}
    for role, names in CURVE_ALIASES.items():
        found[role] = next((cols_upper[n] for n in names if n in cols_upper), None)
    return found



# ---------------------------------------------------------------- data QC
# Physically plausible ranges per log type. Values outside are tool errors,
# spikes, casing effects or saturation -> set to NaN (never silently kept).
VALID_RANGE = {
    "gr": (0, 300),        # API
    "rho": (1.0, 3.2),     # g/cc
    "rt": (0.1, 20000),    # ohm.m (upper bound also rejects tool-saturation plateau)
    "cali": (50, 500),     # mm / inch handled by median reference
    "sp": (-300, 300),
}


def qc_curves(df: pd.DataFrame, curves: dict, despike_window=5):
    """Mask out-of-range values, drop saturation plateaus, remove spikes with
    a rolling median. Returns (clean_df, qc_report_df)."""
    clean = df.copy()
    rows = []
    for role, col in curves.items():
        if not col or col not in clean or role not in VALID_RANGE:
            continue
        s = clean[col].astype(float)
        n0 = int(s.notna().sum())
        lo, hi = VALID_RANGE[role]
        s = s.where((s >= lo) & (s <= hi))
        # saturation plateau: a long run of one identical max value
        top = s.max()
        if (s == top).sum() > 20:
            s = s.where(s != top)
        n1 = int(s.notna().sum())
        med = s.rolling(despike_window, center=True, min_periods=1).median()
        mad = (s - med).abs().rolling(51, center=True, min_periods=5).median()
        spike = (s - med).abs() > 6 * 1.4826 * mad.replace(0, np.nan)
        s = s.where(~spike.fillna(False))
        clean[col] = s
        rows.append({"curve": col, "role": role, "valid_before": n0,
                     "removed_out_of_range": n0 - n1,
                     "removed_spikes": int(spike.fillna(False).sum()),
                     "valid_after": int(s.notna().sum())})
    return clean, pd.DataFrame(rows)

# ---------------------------------------------------------------- Vsh
def shale_volume(gr: pd.Series, gr_clean=None, gr_shale=None, method="linear"):
    """Shale volume from gamma ray.

    IGR = (GR - GRclean) / (GRshale - GRclean), clipped to [0, 1].
    method: 'linear' | 'larionov_tertiary' | 'larionov_older'
    Baselines default to the 5th / 95th percentile of the log.
    """
    gr_clean = np.nanpercentile(gr, 5) if gr_clean is None else gr_clean
    gr_shale = np.nanpercentile(gr, 95) if gr_shale is None else gr_shale
    igr = ((gr - gr_clean) / (gr_shale - gr_clean)).clip(0, 1)
    if method == "linear":
        vsh = igr
    elif method == "larionov_tertiary":
        vsh = 0.083 * (2 ** (3.7 * igr) - 1)
    elif method == "larionov_older":
        vsh = 0.33 * (2 ** (2 * igr) - 1)
    else:
        raise ValueError(f"unknown method {method}")
    return vsh.clip(0, 1), gr_clean, gr_shale


# ---------------------------------------------------------------- porosity
def density_porosity(rho: pd.Series, rho_ma=2.65, rho_fl=1.0):
    """phi_D = (rho_ma - rho_b) / (rho_ma - rho_fl), clipped to [0, 0.5]."""
    return ((rho_ma - rho) / (rho_ma - rho_fl)).clip(0, 0.5)


def effective_porosity(phi, vsh, phi_shale=0.0):
    """Simple shale correction: phi_e = phi - Vsh * phi_shale."""
    return (phi - vsh * phi_shale).clip(0, 0.5)


# ---------------------------------------------------------------- saturation
def archie_sw(rt: pd.Series, phi: pd.Series, rw=0.1, a=1.0, m=2.0, n=2.0):
    """Archie water saturation: Sw = ((a * Rw) / (phi^m * Rt)) ** (1/n)."""
    phi_safe = phi.where(phi > 0.01)  # avoid divide-by-zero in tight rock
    sw = ((a * rw) / ((phi_safe ** m) * rt)) ** (1.0 / n)
    return sw.clip(0, 1)


# ---------------------------------------------------------------- QC
def caliper_flag(cali: pd.Series, bit_size=None, tol=0.15):
    """True where hole is washed out / rugose (> tol above bit size).
    If bit size is unknown, uses the median caliper as the reference.
    """
    ref = np.nanmedian(cali) if bit_size is None else bit_size
    return cali > ref * (1 + tol)


# ---------------------------------------------------------------- pipeline
def interpret(df, curves, vsh_method="linear", rho_ma=2.65, rho_fl=1.0,
              rw=0.1, a=1.0, m=2.0, n=2.0,
              vsh_cut=0.4, phi_cut=0.10, sw_cut=0.60,
              gr_clean=None, gr_shale=None, bit_size=None):
    """Run the full workflow and return (results_df, summary_dict)."""
    df, qc_report = qc_curves(df, curves)
    out = pd.DataFrame(index=df.index)
    for role in ("gr", "rho", "rt", "sp", "cali"):
        if curves.get(role):
            out[role.upper()] = df[curves[role]]

    out["VSH"], gc, gs = shale_volume(out["GR"], gr_clean, gr_shale, vsh_method)
    out["PHID"] = density_porosity(out["RHO"], rho_ma, rho_fl)
    out["PHIE"] = effective_porosity(out["PHID"], out["VSH"])
    out["SW"] = archie_sw(out["RT"], out["PHIE"], rw, a, m, n)

    if "CALI" in out:
        out["BAD_HOLE"] = caliper_flag(out["CALI"], bit_size)
    else:
        out["BAD_HOLE"] = False

    valid = out[["VSH", "PHIE", "SW"]].notna().all(axis=1)
    out["RESERVOIR"] = valid & (out["VSH"] < vsh_cut) & (out["PHIE"] > phi_cut)
    out["PAY"] = out["RESERVOIR"] & (out["SW"] < sw_cut) & ~out["BAD_HOLE"]

    step = float(np.nanmedian(np.diff(out.index.values)))
    gross = float(valid.sum() * step)
    net_res = float(out["RESERVOIR"].sum() * step)
    net_pay = float(out["PAY"].sum() * step)
    pay = out[out["PAY"]]
    summary = {
        "depth_top": float(out.index.min()),
        "depth_base": float(out.index.max()),
        "sample_step": step,
        "gross_interval": gross,
        "net_reservoir": net_res,
        "net_pay": net_pay,
        "net_to_gross": net_res / gross if gross else np.nan,
        "avg_phi_pay": float(pay["PHIE"].mean()) if len(pay) else np.nan,
        "avg_sw_pay": float(pay["SW"].mean()) if len(pay) else np.nan,
        "avg_vsh_pay": float(pay["VSH"].mean()) if len(pay) else np.nan,
        "bad_hole_pct": float(100 * out["BAD_HOLE"].mean()),
        "gr_clean": float(gc),
        "gr_shale": float(gs),
    }
    summary['qc_report'] = qc_report
    return out, summary


def pay_zones(res: pd.DataFrame, min_thickness=0.0) -> pd.DataFrame:
    """Collapse the PAY flag into a table of contiguous zones."""
    flag = res["PAY"].values
    depth = res.index.values
    step = float(np.nanmedian(np.diff(depth)))
    zones, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if (not f or i == len(flag) - 1) and start is not None:
            end = i if f else i - 1
            seg = res.iloc[start:end + 1]
            thick = (end - start + 1) * step
            if thick >= min_thickness:
                zones.append({
                    "Top": depth[start], "Base": depth[end] + step,
                    "Thickness": thick,
                    "Avg PHIE": seg["PHIE"].mean(),
                    "Avg Sw": seg["SW"].mean(),
                    "Avg Vsh": seg["VSH"].mean(),
                })
            start = None
    return pd.DataFrame(zones)
