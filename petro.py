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
    "gr": ["GR", "GAMN", "GAMMA", "GRC", "GR_EDTC", "GRD", "GRR", "GAMMA_RAY"],
    "rho": ["RHOB", "DEN", "DFAR", "RHOZ", "DENS", "RHOM", "RHOC"],
    "nphi": ["NPHI", "NEUT", "TNPH", "NPOR", "CNPOR", "NPSS", "NPLS", "CNL"],
    "rt": ["RT", "ILD", "LLD", "RDEP", "AT90", "PR", "RES_DEEP", "RILD", "RT90",
           "RESD", "LL3", "RLL3", "RLLD", "IND", "RDEEP", "RES", "RESDEEP", "M2R9"],
    "sp": ["SP", "SPR"],
    "cali": ["CALI", "CAL", "HCAL", "C1", "CALS", "CAL1", "CALIPER"],
    # an already-computed total porosity curve (used instead of density porosity)
    "phi": ["PHIND", "PHIT", "PHI", "POR", "POROSITY"],
}


def load_las(path_or_file) -> pd.DataFrame:
    """Read a LAS file into a DataFrame indexed by depth. Nulls become NaN.
    The depth unit from the LAS header is kept in df.attrs['depth_unit']."""
    las = lasio.read(path_or_file)
    df = las.df()
    df.index.name = "DEPTH"
    unit = (las.curves[0].unit or "").strip().upper()
    df.attrs["depth_unit"] = "ft" if unit in ("F", "FT", "FEET", "FOOT") else (
        "m" if unit in ("M", "METRE", "METER", "METRES", "METERS") else (unit.lower() or "m"))
    return df


def read_csv_logs(path_or_file) -> pd.DataFrame:
    """Read a multi-well CSV of logs (e.g. the Kansas facies dataset).
    Any column named like 'X_log10' is converted to linear units (column 'X'),
    since resistivity is often stored as log10(ohm.m)."""
    raw = pd.read_csv(path_or_file)
    for c in list(raw.columns):
        if c.lower().endswith("_log10"):
            raw[c[:-6]] = 10.0 ** raw[c]
    return raw


def well_frame(raw: pd.DataFrame, well=None, depth_col="Depth",
               well_col="Well Name", depth_unit="ft") -> pd.DataFrame:
    """Pick one well out of a multi-well table and index it by depth."""
    d = raw
    if well_col in raw.columns and well is not None:
        d = raw[raw[well_col] == well]
    d = d.sort_values(depth_col).set_index(depth_col)
    d.index.name = "DEPTH"
    d = d.select_dtypes("number")
    d.attrs["depth_unit"] = depth_unit
    return d


def _as_fraction(phi: pd.Series, max_valid=0.5) -> pd.Series:
    """Porosity may arrive as percent (0-100) or fraction (0-1). Convert to a
    fraction and blank out physically impossible values instead of clipping."""
    s = phi.astype(float)
    if s.quantile(0.99) > 1.5:  # percent
        s = s / 100.0
    return s.where((s >= 0) & (s <= max_valid))


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
    "cali": (0.5, 500),    # inches (Kansas) or mm; washout test uses the median as reference
    "sp": (-300, 300),
}


def _isolated_spikes(s: pd.Series, window=5, k=8.0) -> pd.Series:
    """Flag single-sample spikes only.

    A sample is a spike when ALL of these hold:
      1. it differs from the rolling median by more than k * noise, where
         noise is one GLOBAL robust estimate for the whole curve (1.4826 * median
         absolute deviation from the rolling median, floored at 2% of the curve's
         5th-95th percentile spread so smooth/quantised curves don't get a
         near-zero threshold);
      2. it jumps away from BOTH neighbours in the same direction (a real bed
         boundary differs from only one side, so it is never removed);
      3. each of those jumps is at least half the threshold.
    """
    med = s.rolling(window, center=True, min_periods=1).median()
    dev = (s - med).abs()
    spread = np.nanpercentile(s.dropna(), 95) - np.nanpercentile(s.dropna(), 5) if s.notna().sum() > 10 else 0.0
    noise = max(1.4826 * float(np.nanmedian(dev)), 0.02 * spread, 1e-12)
    thr = k * noise
    up, dn = s - s.shift(1), s - s.shift(-1)
    isolated = (up * dn > 0) & (up.abs() > thr / 2) & (dn.abs() > thr / 2)
    return ((dev > thr) & isolated).fillna(False)


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
        # only treat a repeated maximum as tool saturation when the curve
        # otherwise varies (a flat/constant curve is not a saturation plateau)
        if s.nunique() > 10 and (s == top).sum() > 20:
            s = s.where(s != top)
        n1 = int(s.notna().sum())
        # resistivity spans orders of magnitude, so test it on a log scale
        spike = _isolated_spikes(np.log10(s) if role == "rt" else s, despike_window)
        s = s.where(~spike)
        clean[col] = s
        rows.append({"curve": col, "role": role, "valid_before": n0,
                     "removed_out_of_range": n0 - n1,
                     "removed_spikes": int(spike.sum()),
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


def shale_resistivity(rt: pd.Series, vsh: pd.Series, vsh_min=0.8):
    """Rsh = median Rt in the shaliest zones (Vsh >= vsh_min). Falls back to the
    5th-percentile of Rt if no such zone exists."""
    shale_rt = rt[(vsh >= vsh_min) & rt.notna()]
    if len(shale_rt) >= 10:
        return float(shale_rt.median())
    return float(np.nanpercentile(rt, 5))


def simandoux_sw(rt: pd.Series, phi: pd.Series, vsh: pd.Series, rsh: float,
                 rw=0.1, a=1.0, m=2.0):
    """Shaly-sand water saturation (Simandoux form, n = 2).

    1/Rt = (phi^m / (a Rw)) * Sw^2 + (Vsh / Rsh) * Sw
    -> quadratic in Sw: A Sw^2 + B Sw - 1/Rt = 0, A = phi^m/(a Rw), B = Vsh/Rsh.
    With Vsh = 0 it reduces exactly to Archie (n = 2).
    """
    phi_safe = phi.where(phi > 0.01)
    A = (phi_safe ** m) / (a * rw)
    B = vsh / rsh
    sw = (-B + np.sqrt(B ** 2 + 4 * A / rt)) / (2 * A)
    return sw.clip(0, 1)


def estimate_rw(rt: pd.Series, phi: pd.Series, vsh: pd.Series, a=1.0, m=2.0,
                vsh_max=0.2, phi_min=0.08):
    """Apparent-water-resistivity method: Rwa = Rt * phi^m / a.
    In a clean, porous, water-bearing zone Rwa ~ Rw, and hydrocarbons only raise
    Rwa, so the low end of Rwa over clean porous rock is a reasonable Rw estimate.
    Returns the 5th percentile of Rwa (NaN if there is no clean porous rock).
    This ASSUMES at least one wet zone exists - confirm with water-sample data
    or a Pickett plot whenever you can.
    """
    ok = (vsh < vsh_max) & (phi > phi_min) & rt.notna()
    if ok.sum() < 20:
        return float("nan")
    rwa = rt[ok] * (phi[ok] ** m) / a
    return float(np.nanpercentile(rwa, 5))


# ---------------------------------------------------------------- QC
def caliper_flag(cali: pd.Series, bit_size=None, tol=0.15):
    """True where hole is washed out / rugose (> tol above bit size).
    If bit size is unknown, uses the median caliper as the reference.
    """
    if bit_size is None and cali.notna().sum() == 0:
        return pd.Series(False, index=cali.index)
    ref = np.nanmedian(cali) if bit_size is None else bit_size
    return cali > ref * (1 + tol)


# ---------------------------------------------------------------- pipeline
def interpret(df, curves, vsh_method="linear", rho_ma=2.65, rho_fl=1.0,
              rw=0.1, a=1.0, m=2.0, n=2.0,
              vsh_cut=0.4, phi_cut=0.10, sw_cut=0.60,
              gr_clean=None, gr_shale=None, bit_size=None,
              sw_model="archie", rsh=None):
    """Run the full workflow and return (results_df, summary_dict).
    sw_model: 'archie' (clean rock) or 'simandoux' (shaly sand)."""
    df, qc_report = qc_curves(df, curves)
    out = pd.DataFrame(index=df.index)
    for role in ("gr", "rho", "rt", "sp", "cali", "phi"):
        if curves.get(role):
            out[role.upper()] = df[curves[role]]

    out["VSH"], gc, gs = shale_volume(out["GR"], gr_clean, gr_shale, vsh_method)
    if "PHI" in out:                       # use the supplied porosity curve
        out["PHID"] = _as_fraction(out["PHI"])
    else:                                  # otherwise derive it from bulk density
        out["PHID"] = density_porosity(out["RHO"], rho_ma, rho_fl)
    out["PHIE"] = effective_porosity(out["PHID"], out["VSH"])
    rsh_used = np.nan
    if sw_model == "simandoux":
        rsh_used = rsh if rsh else shale_resistivity(out["RT"], out["VSH"])
        out["SW"] = simandoux_sw(out["RT"], out["PHIE"], out["VSH"], rsh_used, rw, a, m)
    else:
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
        "sw_model": sw_model,
        "rsh_used": float(rsh_used) if np.isfinite(rsh_used) else np.nan,
        "rw_estimate": estimate_rw(out["RT"], out["PHIE"], out["VSH"], a, m),
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
