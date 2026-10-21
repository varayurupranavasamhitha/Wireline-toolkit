"""
Wireline Log Interpretation & Job Planning Toolkit
Run:  streamlit run app.py
"""
import io
import os

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
    up = st.file_uploader("Upload a LAS or CSV file", type=["las", "LAS", "csv"])
    SAMPLE_KS = "data/sample_kansas_well.csv"   # one Kansas gas-field well (bundled)
    SAMPLE_LAS = "data/6038187_v1.2.las"        # Scorpio shallow bore (test only)
    if up is not None:
        src, name = up, up.name
    else:
        choice = st.radio("No file uploaded - use a built-in sample:",
                          ["Kansas gas-field well (CSV)", "Scorpio shallow bore (LAS, test only)"])
        if choice.startswith("Kansas") and os.path.exists(SAMPLE_KS):
            src, name = SAMPLE_KS, os.path.basename(SAMPLE_KS)
            st.success("Sample: well CHURCHMAN BIBLE, Hugoton/Panoma gas fields, Kansas. "
                       "Upload your own file above to replace it.")
        else:
            src, name = SAMPLE_LAS, os.path.basename(SAMPLE_LAS)
            st.info("Scorpio E1 is a 137 m shallow bore, NOT a hydrocarbon well. "
                    "It only shows that the pipeline runs.")
    is_csv = name.lower().endswith(".csv")

    with st.expander("Need a sample file?"):
        st.caption("Download a sample, then upload it above to see how uploading works. "
                   "Kansas data: public Hugoton/Panoma gas-field well logs (Hall, 2016, "
                   "The Leading Edge); check the source page for licence and attribution. "
                   "Scorpio: a shallow bore, for testing only.")
        for label, path, mime in [
            ("Download Kansas well (CSV, 1 well)", SAMPLE_KS, "text/csv"),
            ("Download Kansas wells (CSV, all 8 wells)", "data/facies_data.csv", "text/csv"),
            ("Download Scorpio shallow bore (LAS)", SAMPLE_LAS, "text/plain"),
        ]:
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    st.download_button(label, fh.read(), file_name=os.path.basename(path),
                                       mime=mime, key="dl_" + os.path.basename(path))

    def _text(s):
        return io.StringIO(s.getvalue().decode("utf-8", "ignore")) if hasattr(s, "getvalue") else s

    @st.cache_data(show_spinner=False)
    def _load_las(s):
        return petro.load_las(_text(s))

    @st.cache_data(show_spinner=False)
    def _load_csv(s):
        return petro.read_csv_logs(_text(s))

    data_id = name
    if is_csv:
        raw = _load_csv(src)
        well = None
        if "Well Name" in raw.columns:
            well = st.selectbox("Well", sorted(raw["Well Name"].unique()), key="well_" + data_id)
            data_id = f"{name}:{well}"
        depth_col = "Depth" if "Depth" in raw.columns else st.selectbox(
            "Depth column", list(raw.columns))
        dunit = st.radio("Depth unit", ["ft", "m"], horizontal=True,
                         help="CSV files don't state the unit. Kansas data is normally in feet.")
        df = petro.well_frame(raw, well, depth_col=depth_col, depth_unit=dunit)
    else:
        df = _load_las(src)
    guess = petro.guess_curves(df)
    cols = ["(none)"] + list(df.columns)

    st.header("2. Curve mapping")
    curves = {}
    for role, label in [("gr", "Gamma ray"), ("rho", "Bulk density"),
                        ("phi", "Porosity (already computed, optional)"),
                        ("rt", "Deep resistivity"), ("sp", "SP"), ("cali", "Caliper")]:
        g = guess.get(role)
        idx = cols.index(g) if g in cols else 0
        pick = st.selectbox(label, cols, index=idx, key=f"{role}_{data_id}")
        curves[role] = None if pick == "(none)" else pick
    if curves["phi"]:
        st.caption("A porosity curve is mapped, so it is used instead of density porosity.")

    st.header("3. Parameters")
    vsh_method = st.selectbox("Vsh method", ["linear", "larionov_tertiary", "larionov_older"])
    rho_ma = st.number_input("Matrix density (g/cc)", 2.0, 3.2, 2.65, 0.01,
                             help="Sandstone 2.65, limestone 2.71, dolomite 2.87")
    rho_fl = st.number_input("Fluid density (g/cc)", 0.6, 1.3, 1.0, 0.05)
    use_est_rw = st.checkbox(
        "Use Rw estimated from the logs", value=True,
        help="Apparent-Rw method (needs a clean, water-bearing zone in the interval). "
             "Only applied when the estimate is between 0.01 and 2 ohm.m; "
             "otherwise the Rw typed below is used.")
    rw_typed = st.number_input("Rw (ohm.m)", 0.005, 5.0, 0.10, 0.005, format="%.3f",
                               disabled=use_est_rw)
    rw = rw_typed
    a = st.number_input("Archie a", 0.5, 2.0, 1.0, 0.05)
    m = st.number_input("Archie m", 1.5, 3.0, 2.0, 0.05)
    n = st.number_input("Archie n", 1.5, 3.0, 2.0, 0.05)
    sw_model = st.selectbox("Water saturation model", ["archie", "simandoux"],
                            help="Archie: clean rock. Simandoux: shaly sand (uses shale resistivity, n fixed at 2).")
    st.subheader("Cutoffs")
    vsh_cut = st.slider("Vsh <", 0.05, 0.8, 0.40, 0.05)
    phi_cut = st.slider("PHIE >", 0.02, 0.30, 0.10, 0.01)
    sw_cut = st.slider("Sw <", 0.2, 1.0, 0.60, 0.05)

if not curves["gr"] or not curves["rt"] or not (curves["rho"] or curves["phi"]):
    st.warning("Map gamma ray, deep resistivity, and either bulk density or a porosity curve.")
    st.stop()

rw_source = "typed"
if use_est_rw:
    # The Rw estimate depends only on Vsh, porosity and Rt, so a first pass with any Rw is enough.
    _, s0 = petro.interpret(df, curves, vsh_method, rho_ma, rho_fl, rw_typed, a, m, n,
                            vsh_cut, phi_cut, sw_cut, sw_model=sw_model)
    if np.isfinite(s0["rw_estimate"]) and 0.01 <= s0["rw_estimate"] <= 2.0:
        rw, rw_source = round(float(s0["rw_estimate"]), 3), "estimated from logs"
res, summ = petro.interpret(df, curves, vsh_method, rho_ma, rho_fl, rw, a, m, n,
                            vsh_cut, phi_cut, sw_cut, sw_model=sw_model)
qc = summ.pop("qc_report")
with st.sidebar:
    st.subheader("Rw check")
    if np.isfinite(summ["rw_estimate"]):
        st.metric("Rw estimated from logs (ohm.m)", f"{summ['rw_estimate']:.3f}")
        st.caption("Apparent-Rw method (5th percentile of Rwa in clean, porous rock). "
                   "Assumes a water-bearing zone exists; confirm with a water sample "
                   "or Pickett plot.")
    else:
        st.caption("No clean, porous zone to estimate Rw from.")
    st.info(f"Rw used in all results: **{rw:.3f} ohm.m** ({rw_source}).")
    if sw_model == "simandoux" and np.isfinite(summ["rsh_used"]):
        st.caption(f"Shale resistivity used: {summ['rsh_used']:.2f} ohm.m")
du = df.attrs.get("depth_unit", "m")  # depth unit read from the LAS header (ft or m)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Log display", "Net pay summary", "Wireline job planner", "HSE checklist"])

# ------------------------------------------------------------------ tab 1
with tab1:
    d0, d1 = float(res.index.min()), float(res.index.max())
    lo, hi = st.slider(f"Depth window ({du})", d0, d1, (d0, d1))
    w = res.loc[lo:hi]
    tracks = [
        ("Gamma ray / SP", [("GR", "green", False), ("SP", "blue", False)]),
        ("Resistivity (log)", [("RT", "red", True)]),
        ("Density" if "RHO" in w else "Porosity (supplied)",
         [("RHO", "black", False)] if "RHO" in w else [("PHID", "navy", False)]),
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
    axes[0].set_ylabel(f"Depth ({du})")
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
    c[0].metric(f"Gross interval ({du})", f"{summ['gross_interval']:.1f}")
    c[1].metric(f"Net reservoir ({du})", f"{summ['net_reservoir']:.1f}")
    c[2].metric(f"Net pay ({du})", f"{summ['net_pay']:.1f}")
    c[3].metric("Net / gross", f"{summ['net_to_gross']:.2f}")
    c = st.columns(4)
    c[0].metric("Avg PHIE (pay)", f"{summ['avg_phi_pay']:.3f}")
    c[1].metric("Avg Sw (pay)", f"{summ['avg_sw_pay']:.3f}")
    c[2].metric("Avg Vsh (pay)", f"{summ['avg_vsh_pay']:.3f}")
    c[3].metric("Bad hole (%)", f"{summ['bad_hole_pct']:.1f}")
    st.caption(f"Gamma-ray baselines: clean {summ['gr_clean']:.1f}, shale {summ['gr_shale']:.1f} | "
               f"Rw {rw:.3f} ohm.m ({rw_source}) | model: {sw_model} | "
               f"cutoffs: Vsh < {vsh_cut}, PHIE > {phi_cut}, Sw < {sw_cut}")
    min_t = st.number_input(f"Minimum zone thickness ({du})", 0.0, 20.0, 1.0, 0.5)
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
