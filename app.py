"""
Wireline Log Interpretation & Job Planning Toolkit
Run:  streamlit run app.py
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import ops
import petro

st.set_page_config(page_title="Wireline Toolkit", layout="wide")
st.title("Wireline Log Interpretation & Job Planning Toolkit")
st.caption("Open-hole log QC and petrophysics (Vsh, porosity, Archie Sw, net pay) "
           "plus wireline cable-tension planning and an HSE pre-job checklist.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Upload a LAS file", type=["las", "LAS"])
    src = up if up is not None else "data/6038187_v1.2.las"
    if up is None:
        st.info("Using the bundled sample (Scorpio E1, 137 m shallow bore). "
                "It is NOT a hydrocarbon well - upload a reservoir LAS for "
                "meaningful pay results.")

    @st.cache_data(show_spinner=False)
    def _load(s):
        if hasattr(s, "getvalue"):  # uploaded file -> text stream for lasio
            return petro.load_las(io.StringIO(s.getvalue().decode("utf-8", "ignore")))
        return petro.load_las(s)

    df = _load(src)
    guess = petro.guess_curves(df)
    cols = ["(none)"] + list(df.columns)

    st.header("2. Curve mapping")
    curves = {}
    for role, label in [("gr", "Gamma ray"), ("rho", "Bulk density"),
                        ("rt", "Deep resistivity"), ("sp", "SP"), ("cali", "Caliper")]:
        g = guess.get(role)
        idx = cols.index(g) if g in cols else 0
        pick = st.selectbox(label, cols, index=idx, key=role)
        curves[role] = None if pick == "(none)" else pick

    st.header("3. Parameters")
    vsh_method = st.selectbox("Vsh method", ["linear", "larionov_tertiary", "larionov_older"])
    rho_ma = st.number_input("Matrix density (g/cc)", 2.0, 3.2, 2.65, 0.01,
                             help="Sandstone 2.65, limestone 2.71, dolomite 2.87")
    rho_fl = st.number_input("Fluid density (g/cc)", 0.6, 1.3, 1.0, 0.05)
    rw = st.number_input("Rw (ohm.m)", 0.005, 5.0, 0.10, 0.005, format="%.3f")
    a = st.number_input("Archie a", 0.5, 2.0, 1.0, 0.05)
    m = st.number_input("Archie m", 1.5, 3.0, 2.0, 0.05)
    n = st.number_input("Archie n", 1.5, 3.0, 2.0, 0.05)
    st.subheader("Cutoffs")
    vsh_cut = st.slider("Vsh <", 0.05, 0.8, 0.40, 0.05)
    phi_cut = st.slider("PHIE >", 0.02, 0.30, 0.10, 0.01)
    sw_cut = st.slider("Sw <", 0.2, 1.0, 0.60, 0.05)

if not curves["gr"] or not curves["rho"] or not curves["rt"]:
    st.warning("Map at least gamma ray, bulk density and deep resistivity to continue.")
    st.stop()

res, summ = petro.interpret(df, curves, vsh_method, rho_ma, rho_fl, rw, a, m, n,
                            vsh_cut, phi_cut, sw_cut)
qc = summ.pop("qc_report")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Log display", "Net pay summary", "Wireline job planner", "HSE checklist"])

# ------------------------------------------------------------------ tab 1
with tab1:
    d0, d1 = float(res.index.min()), float(res.index.max())
    lo, hi = st.slider("Depth window (m)", d0, d1, (d0, d1))
    w = res.loc[lo:hi]
    tracks = [
        ("Gamma ray / SP", [("GR", "green", False), ("SP", "blue", False)]),
        ("Resistivity (log)", [("RT", "red", True)]),
        ("Density", [("RHO", "black", False)]),
        ("Vsh / PHIE / Sw", [("VSH", "saddlebrown", False), ("PHIE", "blue", False),
                             ("SW", "cyan", False)]),
    ]
    fig, axes = plt.subplots(1, len(tracks) + 1, figsize=(14, 9), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.35]})
    for ax, (title, curs) in zip(axes, tracks):
        for name, color, logx in curs:
            if name in w:
                ax.plot(w[name], w.index, color=color, lw=0.8, label=name)
                if logx:
                    ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (m)")
    pay = w["PAY"].astype(float)
    axes[-1].fill_betweenx(w.index, 0, pay, color="gold", step="mid")
    axes[-1].set_xlim(0, 1)
    axes[-1].set_title("Pay flag", fontsize=9)
    axes[-1].set_xticks([])
    fig.tight_layout()
    st.pyplot(fig)
    st.subheader("Data QC report")
    st.caption("Out-of-range values, saturation plateaus and spikes are masked "
               "before any calculation.")
    st.dataframe(qc, hide_index=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    c = st.columns(4)
    c[0].metric("Gross interval (m)", f"{summ['gross_interval']:.1f}")
    c[1].metric("Net reservoir (m)", f"{summ['net_reservoir']:.1f}")
    c[2].metric("Net pay (m)", f"{summ['net_pay']:.1f}")
    c[3].metric("Net / gross", f"{summ['net_to_gross']:.2f}")
    c = st.columns(4)
    c[0].metric("Avg PHIE (pay)", f"{summ['avg_phi_pay']:.3f}")
    c[1].metric("Avg Sw (pay)", f"{summ['avg_sw_pay']:.3f}")
    c[2].metric("Avg Vsh (pay)", f"{summ['avg_vsh_pay']:.3f}")
    c[3].metric("Bad hole (%)", f"{summ['bad_hole_pct']:.1f}")
    st.caption(f"Gamma-ray baselines: clean {summ['gr_clean']:.1f}, shale {summ['gr_shale']:.1f}")
    min_t = st.number_input("Minimum zone thickness (m)", 0.0, 10.0, 0.5, 0.25)
    zones = petro.pay_zones(res, min_t)
    st.subheader("Pay zones")
    st.dataframe(zones.round(3), hide_index=True)
    if len(zones):
        st.download_button("Download zones (CSV)", zones.to_csv(index=False),
                           "pay_zones.csv", "text/csv")
    st.warning(
        "Interpretation is only as good as Rw, matrix density and the cutoffs. "
        "Archie assumes clean, water-wet rock; shaly sands need Simandoux/"
        "Indonesia models. Always calibrate with core or test data.")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Cable tension and depth capability")
    c1, c2, c3 = st.columns(3)
    tool_w = c1.number_input("Tool string weight in air (kgf)", 10.0, 5000.0, 300.0)
    cable_w = c1.number_input("Cable weight in air (kgf/m)", 0.1, 2.0, 0.30, 0.01)
    mud = c2.number_input("Mud density (kg/m3)", 800.0, 2400.0, 1200.0, 10.0)
    brk = c2.number_input("Cable breaking strength (kgf)", 1000.0, 20000.0, 6000.0, 100.0)
    sf = c3.number_input("Safety factor", 1.2, 4.0, 2.0, 0.1)
    fric = c3.number_input("Friction (kgf per km)", 0.0, 500.0, 40.0, 5.0)
    td = c1.number_input("Target depth (m)", 100.0, 8000.0, 3000.0, 100.0)
    wp = c2.number_input("Weak point rating (kgf)", 500.0, 20000.0, 4500.0, 100.0)
    overpull = c3.number_input("Planned stuck-tool overpull (kgf)", 0.0, 5000.0, 1000.0, 100.0)

    depths = np.linspace(0, td, 25)
    prof = pd.DataFrame(ops.cable_tension_profile(depths, tool_w, cable_w, mud, fric))
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(prof["Depth (m)"], prof["RIH tension (kgf)"], label="RIH")
    ax2.plot(prof["Depth (m)"], prof["POOH tension (kgf)"], label="POOH")
    ax2.axhline(brk / sf, color="orange", ls="--", label="Allowable (break / SF)")
    ax2.axhline(wp, color="red", ls=":", label="Weak point")
    ax2.set_xlabel("Depth (m)")
    ax2.set_ylabel("Surface tension (kgf)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

    maxd = ops.max_depth_for_cable(tool_w, cable_w, mud, brk, sf, overpull)
    peak = float(prof["POOH tension (kgf)"].max())
    st.metric("Max depth within safety factor (m)", f"{maxd:,.0f}",
              delta="OK" if maxd >= td else "Target depth exceeds limit",
              delta_color="normal" if maxd >= td else "inverse")
    for msg in ops.weak_point_check(peak + overpull, wp, brk):
        (st.success if msg.startswith("PASS") else st.error)(msg)

    st.subheader("Job duration estimate")
    d1_, d2_, d3_ = st.columns(3)
    ls = d1_.number_input("Logging speed (m/min)", 1.0, 30.0, 9.0)
    rs = d2_.number_input("RIH speed (m/min)", 5.0, 60.0, 30.0)
    rp = d3_.number_input("Repeat section (m)", 0.0, 500.0, 100.0)
    est = ops.logging_time_estimate(td, ls, rs, rp)
    st.dataframe(pd.DataFrame([est]).round(0).astype(int), hide_index=True)
    st.caption(f"About {est['Total (min)'] / 60:.1f} h end to end.")
    st.info("Inputs are user-supplied planning numbers, not vendor tool specifications.")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.subheader("Wireline pre-job HSE checklist")
    done = total = 0
    for section, items in ops.HSE_CHECKLIST.items():
        with st.expander(section, expanded=False):
            for i, item in enumerate(items):
                total += 1
                done += st.checkbox(item, key=f"{section}-{i}")
    st.progress(done / total if total else 0, text=f"{done}/{total} items confirmed")
    if done == total:
        st.success("All checks confirmed. Proceed per site procedure.")
    else:
        st.warning("Do not start the job until every item is confirmed.")
