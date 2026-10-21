# Wireline Log Interpretation & Job Planning Toolkit

Python + Streamlit toolkit that takes open-hole wireline logs (LAS) from raw
curves to a net-pay summary, and adds first-principles wireline job planning
(cable tension, depth capability, weak-point check) with an HSE pre-job checklist.

## What it does
1. **Load and map** any LAS file (lasio); auto-detects GR, density, resistivity, SP, caliper mnemonics.
2. **Data QC**: masks out-of-range values, tool-saturation plateaus and spikes (rolling-median/MAD) before any calculation; reports what was removed.
3. **Petrophysics**: shale volume (linear / Larionov), density porosity, shale-corrected effective porosity, Archie water saturation, caliper washout flag.
4. **Net pay**: Vsh / porosity / Sw cutoffs -> net reservoir, net pay, net-to-gross, zone table (CSV export).
5. **Wireline job planner**: buoyed cable tension (RIH / POOH), max depth within safety factor, weak-point check, job-duration estimate.
6. **HSE checklist**: JSA, pressure control, cable/tool checks, radioactive and explosive source handling, rig-down.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Equations
- IGR = (GR - GRclean) / (GRshale - GRclean); Larionov (tertiary): Vsh = 0.083 (2^(3.7 IGR) - 1)
- phi_D = (rho_ma - rho_b) / (rho_ma - rho_fl)
- Archie: Sw = ((a Rw) / (phi^m Rt))^(1/n)
- Buoyancy factor BF = 1 - rho_mud / rho_steel; surface tension = (W_tool + w_cable D) BF +/- friction

Formulas were verified against hand calculations (see test snippets in the commit/notes).

## Data and honest limitations
- Bundled sample: Scorpio E1 (`data/6038187_v1.2.las`), a 137 m shallow bore from
  the open `lasio` test set. **It is not a hydrocarbon reservoir.** Its resistivity
  curve is flat (~2,700 ohm.m) below ~55 m, so Sw and "pay" values on it only
  demonstrate that the pipeline runs - they are not geological conclusions.
- For a meaningful interpretation, upload a reservoir LAS (e.g. open Volve or
  Kansas Geological Survey wells) and set Rw, matrix density and cutoffs from
  core / water-sample data.
- Archie assumes clean, water-wet rock; shaly sands need Simandoux / Indonesia.
- Job-planner inputs (tool weight, cable strength, speeds) are user-supplied
  planning numbers, not vendor tool specifications.
