# Resume entry (edit the bracketed parts once you've run it on a reservoir LAS)

**Wireline Log Interpretation & Job Planning Toolkit** | Python, Streamlit, lasio, pandas, matplotlib | [GitHub link]

- Built an open-hole log interpretation workflow (LAS import, QC/despiking, Vsh via Larionov, density porosity, Archie Sw, cutoff-based net pay) for gamma, density, resistivity, SP and caliper logs.
- Implemented automated log QC that masks tool-saturation, out-of-range and spike values, and flags washout zones from caliper, reporting data removed per curve.
- Developed a wireline job planner computing buoyed cable tension (RIH/POOH), maximum depth within safety factor, weak-point validation and job-time estimates; validated against hand calculations.
- Added an HSE pre-job checklist covering JSA, pressure control, cable/tool inspection and radioactive/explosive source handling.
- [After you test on a reservoir well: "Applied to the [Volve / KGS] well, identifying X m net pay at avg porosity Y%."]

## For the application / SOVA interview, be ready to explain
- Why density porosity needs the matrix density, and what changes in limestone vs sandstone
- Why Archie breaks down in shaly sands
- What a weak point is and why it must be weaker than the cable
- Why you QC logs before interpreting (casing effects, washouts, saturation)
- Your honest limitation: sample well is shallow, non-hydrocarbon
