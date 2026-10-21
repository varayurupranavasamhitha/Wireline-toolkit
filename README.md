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

## Sample data (built into the app)
The app opens on a bundled sample, so it works the moment you run it. In the sidebar, "Need a sample file?"
has download buttons; download one, then drop it into the uploader to see how uploading works.
- **Kansas gas-field well (CSV):** `data/sample_kansas_well.csv`, well CHURCHMAN BIBLE (Hugoton/Panoma
  gas fields, public dataset used in a 2016 machine-learning contest; credit: Brendon Hall, *The Leading
  Edge*, 2016 - confirm citation and licence on the dataset's source page). 404 readings, 2917.5-3122.5 ft.
- **Scorpio shallow bore (LAS):** `data/6038187_v1.2.las`, from the open `lasio` test set. Test only: it is
  not a hydrocarbon reservoir and its resistivity curve is flat.
- `data/facies_data.csv` (optional): all 8 wells of the Kansas dataset, for the well picker.

## Verified results (sample well, Rw 0.037 ohm.m estimated from the logs)
Linear Vsh, Archie (a=1, m=2, n=2), cutoffs Vsh < 0.4, PHIE > 0.10, Sw < 0.6:
gross interval 198.5 ft, net reservoir 49.0 ft, net pay 19.0 ft, average porosity 22.4%,
average Sw 43.0%, average Vsh 22.1%. Net pay depends strongly on Rw: with Archie it is 19.0 / 11.0 / 8.0 ft at
Rw 0.037 / 0.05 / 0.08 (Simandoux: 26.5 / 14.5 / 10.0 ft). Net reservoir stays 49.0 ft.
These were recomputed independently with plain pandas; run `python test_toolkit.py` (74 checks) to reproduce.

## Honest limitations
- Public benchmark data, not field data recorded by the author. The wells are real; the method and code are original.
- The CSV has no bulk-density curve (the supplied porosity PHIND is used) and does not state its depth unit
  (feet assumed). The fluid type is not in the file; the gas-field label comes from the dataset's description,
  and the neutron-density difference in the pay zones (mean -1.5 porosity units) is consistent with gas.
- Rw is estimated from the logs (apparent-Rw method), which assumes a water-bearing zone exists; confirm with a
  water sample or Pickett plot. The demo well was chosen because it shows the most pay of the wells screened.
- Archie assumes clean, water-wet rock; the app also offers a Simandoux model (n fixed at 2).
- Job-planner inputs (tool weight, cable strength, speeds) are user-supplied planning numbers, not vendor specs.

## Files
`app.py` (Streamlit app) | `petro.py` (log calculations) | `ops.py` (job planner, HSE checklist) |
`test_toolkit.py` (74 checks) | `INTERVIEW_PREP.md` (explains every number)
