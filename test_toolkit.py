"""
test_toolkit.py - verifies every calculation in the toolkit.

Run:  python test_toolkit.py
Every line prints PASS or FAIL. Expected values are worked out by hand (the
arithmetic is shown next to each check) or recomputed independently with plain
pandas/numpy, so a PASS means the toolkit agrees with the maths, not with itself.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

import ops
import petro

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def close(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


S = lambda *v: pd.Series(v, dtype=float)

# ===================================================================== porosity
print("\n-- Density porosity: (rho_ma - rho_b) / (rho_ma - rho_fl)")
v = petro.density_porosity(S(2.30), 2.65, 1.0)[0]
check("rho_b 2.30 -> 0.35/1.65 = 0.2121", close(v, 0.35 / 1.65), f"got {v:.4f}")
check("rho_b above matrix (3.0) clipped to 0", petro.density_porosity(S(3.0))[0] == 0)
check("rho_b equal to fluid (1.0) clipped to 0.5", petro.density_porosity(S(1.0))[0] == 0.5)
check("limestone matrix: rho_b 2.45, rho_ma 2.71 -> 0.26/1.71", close(petro.density_porosity(S(2.45), 2.71, 1.0)[0], 0.26 / 1.71))

print("\n-- Supplied porosity: percent vs fraction, impossible values removed")
pct = petro._as_fraction(S(10, 15, 20, 25, 84.4))
check("percent input converted to fraction", close(pct[0], 0.10) and close(pct[3], 0.25))
check("84.4 % (impossible) becomes NaN, not clipped", np.isnan(pct[4]))
frac = petro._as_fraction(S(0.10, 0.15, 0.20))
check("fraction input left unchanged", close(frac[1], 0.15))

# ===================================================================== Vsh
print("\n-- Shale volume (GR clean 20, GR shale 120, GR 50 -> IGR 0.30)")
gr = S(50)
check("linear: Vsh = IGR = 0.30", close(petro.shale_volume(gr, 20, 120, "linear")[0][0], 0.30))
check("Larionov tertiary: 0.083*(2^(3.7*0.3)-1) = 0.0961", close(petro.shale_volume(gr, 20, 120, "larionov_tertiary")[0][0], 0.083 * (2 ** 1.11 - 1)))
check("Larionov older: 0.33*(2^0.6-1) = 0.1702", close(petro.shale_volume(gr, 20, 120, "larionov_older")[0][0], 0.33 * (2 ** 0.6 - 1)))
check("GR below clean baseline gives Vsh 0 (clipped)", petro.shale_volume(S(5), 20, 120)[0][0] == 0)
check("GR above shale baseline gives Vsh 1 (clipped)", petro.shale_volume(S(200), 20, 120)[0][0] == 1)

# ===================================================================== saturation
print("\n-- Archie: Sw = ((a*Rw)/(phi^m * Rt))^(1/n)")
v = petro.archie_sw(S(20), S(0.2), 0.1)[0]
check("Rt 20, phi 0.2, Rw 0.1 -> sqrt(0.1/(0.04*20)) = 0.3536", close(v, math.sqrt(0.1 / (0.04 * 20))), f"got {v:.4f}")
check("Sw above 1 is clipped to 1", petro.archie_sw(S(1), S(0.1), 0.1)[0] == 1.0)
check("n = 3 changes the exponent: (0.1/(0.04*20))^(1/3)", close(petro.archie_sw(S(20), S(0.2), 0.1, n=3.0)[0], (0.125) ** (1 / 3)))
check("tight rock (phi 0.005) returns NaN, not infinity", np.isnan(petro.archie_sw(S(20), S(0.005), 0.1)[0]))

print("\n-- Simandoux (quadratic: A*Sw^2 + B*Sw - 1/Rt = 0, A = phi^m/(a*Rw), B = Vsh/Rsh)")
check("Rt 10, phi .2, Vsh .3, Rsh 3, Rw .1 -> 0.3904 (hand: (-0.1+sqrt(0.17))/0.8)",
      close(petro.simandoux_sw(S(10), S(0.2), S(0.3), 3.0, 0.1)[0], (-0.1 + math.sqrt(0.17)) / 0.8))
a_ = petro.archie_sw(S(5, 20, 80), S(.15, .2, .25), 0.1)
s_ = petro.simandoux_sw(S(5, 20, 80), S(.15, .2, .25), S(0, 0, 0), 3.0, 0.1)
check("Vsh = 0 reduces exactly to Archie", np.allclose(a_, s_))
check("shaly rock: Simandoux Sw <= Archie Sw", petro.simandoux_sw(S(10), S(.2), S(.3), 3.0, .1)[0] <= petro.archie_sw(S(10), S(.2), .1)[0])
check("shale resistivity = median Rt where Vsh >= 0.8",
      close(petro.shale_resistivity(pd.Series([1.0] * 5 + [3.0] * 12 + [50.0] * 3), pd.Series([0.1] * 5 + [0.9] * 12 + [0.0] * 3)), 3.0))

print("\n-- Rw from logs: Rwa = Rt * phi^m / a, 5th percentile in clean porous rock")
rng = np.random.default_rng(0)
phi = pd.Series(rng.uniform(.12, .28, 400)); vsh = pd.Series(rng.uniform(0, .15, 400))
rt = 0.08 / phi ** 2
rt.iloc[:150] *= 8  # hydrocarbon zone only raises Rt
check("recovers the known Rw 0.08 from a mixed wet/hydrocarbon interval", close(petro.estimate_rw(rt, phi, vsh), 0.08, 1e-6))
check("returns NaN when there is no clean porous rock", np.isnan(petro.estimate_rw(rt, phi, pd.Series([0.9] * 400))))

# ===================================================================== QC
print("\n-- Quality control")
check("caliper 10.0 in with bit 8.5 in -> washout (10 > 8.5*1.15 = 9.775)", petro.caliper_flag(S(10.0), 8.5)[0])
check("caliper 9.5 in with bit 8.5 in -> in gauge", not petro.caliper_flag(S(9.5), 8.5)[0])
check("all-NaN caliper does not crash", not petro.caliper_flag(S(np.nan, np.nan)).any())

x = np.r_[np.full(100, 60.), np.full(100, 110.), np.full(100, 70.)] + np.random.default_rng(3).normal(0, 2, 300)
inj = [20, 70, 130, 180, 250]; x2 = x.copy(); x2[inj] += np.array([60, -45, 70, 55, -50])
flag = petro._isolated_spikes(pd.Series(x2))
check("despike finds exactly the 5 injected spikes", sorted(np.where(flag)[0]) == inj, f"flagged {list(np.where(flag)[0])}")
check("despike never removes a genuine bed boundary (steps at 100 and 200)", not flag[98:102].any() and not flag[198:202].any())
ramp = pd.Series([8.65, 11.43, 15.67, 19.68, 18.75, 16.71, 14.35, 12.0, 10.0, 9.0] * 3)
check("despike keeps a smooth resistivity peak (tested on log scale)", not petro._isolated_spikes(np.log10(ramp)).any())

df = pd.DataFrame({"G": np.r_[np.random.default_rng(1).normal(80, 10, 200), 400.0, -50.0]})
clean, rep = petro.qc_curves(df, {"gr": "G"})
check("GR outside 0-300 API removed (400 and -50)", rep.loc[0, "removed_out_of_range"] == 2)
const = pd.DataFrame({"C": np.full(100, 8.5)})
check("a constant caliper is NOT mistaken for a saturation plateau", petro.qc_curves(const, {"cali": "C"})[0]["C"].notna().all())
plat = pd.DataFrame({"R": np.r_[np.linspace(1, 5000, 80), np.full(40, 50499.9)]})
cl, _ = petro.qc_curves(plat, {"rt": "R"})
check("tool-saturation plateau (50499.9) removed from resistivity", cl["R"].isna().sum() >= 40)

# ===================================================================== pay zones + end-to-end
print("\n-- Pay zones + end-to-end on a synthetic well with KNOWN answers")
depth = np.arange(1000.0, 1100.0, 0.5)         # 200 samples, 0.5 ft step
n_ = len(depth)
GR = np.where((depth < 1030) | (depth >= 1090), 120.0, 20.0)   # shale / clean sand / shale
PHI = np.where(GR < 50, 0.20, 0.05)
RW = 0.05
RT = RW / PHI ** 2                              # wet everywhere ...
hc = (depth >= 1060) & (depth < 1075)           # ... except a 15 ft hydrocarbon zone
RT = np.where(hc, RT * 10, RT)
syn = pd.DataFrame({"GR": GR, "RT": RT, "PHI": PHI}, index=pd.Index(depth, name="DEPTH"))
syn.attrs["depth_unit"] = "ft"
res, s = petro.interpret(syn, {"gr": "GR", "rt": "RT", "phi": "PHI"}, gr_clean=20, gr_shale=120, rw=RW)
check("clean sand = 60 ft (1030-1090) -> net reservoir 60.0 ft", close(s["net_reservoir"], 60.0), f"got {s['net_reservoir']}")
check("hydrocarbon zone = 15 ft -> net pay 15.0 ft", close(s["net_pay"], 15.0), f"got {s['net_pay']}")
check("wet sand: Sw = 1.0", close(res.loc[1040:1050, "SW"].mean(), 1.0))
check("hydrocarbon zone: Sw = 1/sqrt(10) = 0.3162", close(res.loc[1062:1072, "SW"].mean(), 1 / math.sqrt(10), 1e-3))
check("average porosity in pay = 0.20", close(s["avg_phi_pay"], 0.20))
check("Rw estimate recovers the true 0.05", close(s["rw_estimate"], RW, 1e-3), f"got {s['rw_estimate']:.4f}")
z = petro.pay_zones(res, 0.0)
check("pay-zone table: one zone, 1060 to 1075, thickness 15", len(z) == 1 and close(z.loc[0, "Top"], 1060) and close(z.loc[0, "Base"], 1075) and close(z.loc[0, "Thickness"], 15))
check("zone thicknesses add up to net pay", close(z["Thickness"].sum(), s["net_pay"]))
rs, ss = petro.interpret(syn, {"gr": "GR", "rt": "RT", "phi": "PHI"}, gr_clean=20, gr_shale=120, rw=RW, sw_model="simandoux")
check("Simandoux on clean rock gives the same net pay as Archie", close(ss["net_pay"], s["net_pay"]))

# ===================================================================== file readers
print("\n-- File readers")
las_df = petro.load_las("data/6038187_v1.2.las")
check("Scorpio LAS: 2732 samples, depth unit read as metres", len(las_df) == 2732 and las_df.attrs["depth_unit"] == "m")
check("LAS nulls (-99999) become NaN", las_df.isna().any().any() and not (las_df == -99999).any().any())
raw = petro.read_csv_logs("data/sample_kansas_well.csv")
check("CSV: ILD = 10 ** ILD_log10 (hand: log10 value 1.0 -> 10 ohm.m)", close(10 ** 1.0, 10.0) and close(petro.read_csv_logs(pd.io.common.StringIO("ILD_log10\n1.0\n0.0\n"))["ILD"].tolist()[0], 10.0) and close(petro.read_csv_logs(pd.io.common.StringIO("ILD_log10\n1.0\n0.0\n"))["ILD"].tolist()[1], 1.0))
check("CSV resistivity conversion is exact for every row", np.allclose(raw["ILD"], 10 ** raw["ILD_log10"]))
guess = petro.guess_curves(petro.well_frame(raw, "CHURCHMAN BIBLE"))
check("curve names recognised: GR, ILD, PHIND", guess["gr"] == "GR" and guess["rt"] == "ILD" and guess["phi"] == "PHIND", str(guess))

# ===================================================================== bundled Kansas sample, independent recomputation
print("\n-- Bundled sample (well CHURCHMAN BIBLE): toolkit vs independent plain-pandas calculation")
W = "CHURCHMAN BIBLE"
d = petro.well_frame(raw, W, depth_unit="ft"); cur = petro.guess_curves(d)
check("sample: 404 rows, 2917.5-3122.5 ft, 0.5 ft step", len(d) == 404 and d.index.min() == 2917.5 and d.index.max() == 3122.5)
RWV = 0.037
resK, sK = petro.interpret(d, cur, rw=RWV)
g = raw["GR"].copy(); rtv = 10 ** raw["ILD_log10"]; ph = (raw["PHIND"] / 100).where(lambda q: (q >= 0) & (q <= .5))
med = g.rolling(5, center=True, min_periods=1).median(); dv = (g - med).abs()
thr = 8 * max(1.4826 * np.median(dv), .02 * (np.percentile(g, 95) - np.percentile(g, 5)))
u, dn = g - g.shift(1), g - g.shift(-1)
spk = (dv > thr) & (u * dn > 0) & (u.abs() > thr / 2) & (dn.abs() > thr / 2)
g = g.where(~spk)
gc, gs = np.nanpercentile(g, 5), np.nanpercentile(g, 95)
vs = ((g - gc) / (gs - gc)).clip(0, 1)
sw = np.sqrt(RWV / ((ph.where(ph > .01) ** 2) * rtv)).clip(0, 1)
ok = vs.notna() & ph.notna() & sw.notna(); rr = ok & (vs < .4) & (ph > .10); py = rr & (sw < .6)
check("5 gamma-ray spikes removed (1.2 % of 404)", int(spk.sum()) == 5 and sK["qc_report"].loc[0, "removed_spikes"] == 5)
check("GR baselines: clean 30.57, shale 119.92", close(sK["gr_clean"], gc, 1e-6) and close(gc, 30.57, 0.01) and close(gs, 119.92, 0.01), f"{gc:.2f}/{gs:.2f}")
check("Rw estimated from logs = 0.037", close(sK["rw_estimate"], 0.037, 5e-4), f"{sK['rw_estimate']:.4f}")
check("gross interval 198.5 ft  (independent agrees)", close(sK["gross_interval"], ok.sum() * .5) and close(sK["gross_interval"], 198.5))
check("net reservoir 49.0 ft    (independent agrees)", close(sK["net_reservoir"], rr.sum() * .5) and close(sK["net_reservoir"], 49.0))
check("net pay 19.0 ft          (independent agrees)", close(sK["net_pay"], py.sum() * .5) and close(sK["net_pay"], 19.0))
check("average porosity in pay 0.224 (independent agrees)", close(sK["avg_phi_pay"], ph[py].mean(), 1e-9) and close(sK["avg_phi_pay"], 0.2242, 5e-4))
check("average Sw in pay 0.430  (independent agrees)", close(sK["avg_sw_pay"], sw[py].mean(), 1e-9) and close(sK["avg_sw_pay"], 0.4297, 5e-4))
check("average Vsh in pay 0.221 (independent agrees)", close(sK["avg_vsh_pay"], vs[py].mean(), 1e-9) and close(sK["avg_vsh_pay"], 0.2206, 5e-4))
check("zone table adds up to net pay (15 zones)", close(petro.pay_zones(resK, 0)["Thickness"].sum(), 19.0) and len(petro.pay_zones(resK, 0)) == 15)
grid = {}
for model in ("archie", "simandoux"):
    for rw in (0.037, 0.05, 0.08):
        grid[(model, rw)] = petro.interpret(d, cur, rw=rw, sw_model=model)[1]["net_pay"]
check("net pay sensitivity, Archie: 19.0 / 11.0 / 8.0 ft at Rw 0.037 / 0.05 / 0.08", [grid[("archie", r)] for r in (0.037, 0.05, 0.08)] == [19.0, 11.0, 8.0], str([grid[("archie", r)] for r in (0.037, 0.05, 0.08)]))
check("net pay sensitivity, Simandoux: 26.5 / 14.5 / 10.0 ft", [grid[("simandoux", r)] for r in (0.037, 0.05, 0.08)] == [26.5, 14.5, 10.0], str([grid[("simandoux", r)] for r in (0.037, 0.05, 0.08)]))
check("net reservoir does not depend on Rw (49.0 ft in every case)", all(petro.interpret(d, cur, rw=r)[1]["net_reservoir"] == 49.0 for r in (0.037, 0.05, 0.08)))

# ===================================================================== wireline job planner
print("\n-- Wireline job planner (ops.py)")
bf = ops.buoyancy_factor(1200)
check("buoyancy factor at 1200 kg/m3 = 1 - 1200/7850 = 0.8471", close(bf, 1 - 1200 / 7850))
p = ops.cable_tension_profile([0, 3000], 300, 0.30, 1200, 40)
check("tension at surface = 300 * 0.8471 = 254.1 kgf", close(p[0]["RIH tension (kgf)"], 300 * bf, 0.01))
check("tension at 3000 m RIH = (300+900)*0.8471 - 120 = 896.6", close(p[1]["RIH tension (kgf)"], 1200 * bf - 120, 0.01))
check("tension at 3000 m POOH = (300+900)*0.8471 + 120 = 1136.6", close(p[1]["POOH tension (kgf)"], 1200 * bf + 120, 0.01))
md = ops.max_depth_for_cable(300, 0.30, 1200, 6000, 2.0, 1000)
check("max depth = ((6000/2 - 1000)/0.8471 - 300)/0.30 = 6870 m", close(md, ((6000 / 2 - 1000) / bf - 300) / 0.30, 1e-6) and round(md) == 6870, f"{md:.0f}")
check("tool heavier than allowance gives depth 0", ops.max_depth_for_cable(5000, 0.3, 1200, 6000, 2.0, 1000) == 0.0)
check("weak point 4500 between tension 2500 and cable 6000 -> PASS", ops.weak_point_check(2500, 4500, 6000)[0].startswith("PASS"))
check("weak point 6500 above cable strength 6000 -> FAIL", ops.weak_point_check(2500, 6500, 6000)[0].startswith("FAIL"))
check("weak point 2000 below operating tension 2500 -> FAIL", ops.weak_point_check(2500, 2000, 6000)[0].startswith("FAIL"))
t = ops.logging_time_estimate(3000, 9, 30, 100)
check("job time: 100 RIH + 333.3 log + 11.1 repeat + 150 rig = 594.4 min", close(t["Total (min)"], 100 + 3000 / 9 + 100 / 9 + 150, 1e-6), f"{t['Total (min)']:.1f}")
check("HSE checklist has 6 sections and 25 items (4+5+4+4+4+4)", len(ops.HSE_CHECKLIST) == 6 and sum(len(v) for v in ops.HSE_CHECKLIST.values()) == 25,
      f"{len(ops.HSE_CHECKLIST)} sections, {sum(len(v) for v in ops.HSE_CHECKLIST.values())} items")

# ===================================================================== the app itself
print("\n-- The Streamlit app")
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120).run()
    check("app starts on the Kansas sample with no errors", not at.exception)
    m = {x.label: x.value for x in at.metric}
    check("app shows net pay 19.0 ft and net reservoir 49.0 ft on the sample", m.get("Net pay (ft)") == "19.0" and m.get("Net reservoir (ft)") == "49.0", str({k: m[k] for k in list(m)[:3]}))
    sel = [x for x in at.sidebar.selectbox if x.label == "Water saturation model"][0]
    sel.set_value("simandoux").run()
    check("Simandoux selected in the app: no errors, net pay 26.5 ft", not at.exception and {x.label: x.value for x in at.metric}.get("Net pay (ft)") == "26.5")
    at.sidebar.radio[0].set_value("Scorpio shallow bore (LAS, test only)").run()
    check("Scorpio sample loads with no errors (metres)", not at.exception and "Gross interval (m)" in {x.label for x in at.metric})
except ImportError:
    print("SKIP  streamlit not installed - app checks skipped")

print(f"\n{sum(results)} / {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
