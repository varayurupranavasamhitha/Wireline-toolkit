"""
ops.py - wireline job-planning calculations and HSE checklist.

Everything here is first-principles mechanics. No vendor tool specs are
assumed: the user supplies tool weight, cable weight and breaking strength.
"""
from __future__ import annotations

STEEL_DENSITY = 7850.0  # kg/m3


def buoyancy_factor(fluid_density_kgm3: float) -> float:
    """BF = 1 - rho_fluid / rho_steel."""
    return 1.0 - fluid_density_kgm3 / STEEL_DENSITY


def cable_tension_profile(depths_m, tool_weight_air_kgf, cable_weight_air_kgf_per_m,
                          mud_density_kgm3, friction_kgf_per_km=0.0):
    """Surface tension (kgf) while running in hole (RIH) and pulling out (POOH).

    T_rih  = (W_tool + w_cable * D) * BF - friction
    T_pooh = (W_tool + w_cable * D) * BF + friction
    Friction is a simple distributed drag term (kgf per km of cable in hole).
    """
    bf = buoyancy_factor(mud_density_kgm3)
    rows = []
    for d in depths_m:
        w = (tool_weight_air_kgf + cable_weight_air_kgf_per_m * d) * bf
        fr = friction_kgf_per_km * d / 1000.0
        rows.append({"Depth (m)": d, "RIH tension (kgf)": w - fr,
                     "POOH tension (kgf)": w + fr})
    return rows


def max_depth_for_cable(tool_weight_air_kgf, cable_weight_air_kgf_per_m,
                        mud_density_kgm3, breaking_strength_kgf,
                        safety_factor=2.0, stuck_pull_kgf=0.0):
    """Deepest run (m) such that POOH tension + planned stuck-pull overpull
    stays within breaking_strength / safety_factor."""
    bf = buoyancy_factor(mud_density_kgm3)
    allowed = breaking_strength_kgf / safety_factor - stuck_pull_kgf
    avail = allowed / bf - tool_weight_air_kgf
    if avail <= 0:
        return 0.0
    return avail / cable_weight_air_kgf_per_m


def weak_point_check(max_surface_tension_kgf, weak_point_rating_kgf, cable_breaking_kgf):
    """The weak point must break before the cable does, and above max operating tension."""
    msgs = []
    if weak_point_rating_kgf <= max_surface_tension_kgf:
        msgs.append("FAIL: weak point rated below expected operating tension.")
    if weak_point_rating_kgf >= cable_breaking_kgf:
        msgs.append("FAIL: weak point rated at/above cable breaking strength - "
                    "cable could part before the weak point.")
    return msgs or ["PASS: weak point sits between operating tension and cable strength."]


def logging_time_estimate(total_depth_m, logging_speed_m_per_min, rih_speed_m_per_min,
                          repeat_section_m=0.0, rig_up_min=90, rig_down_min=60):
    """Rough job duration: rig-up + RIH + logging up + repeat section + rig-down (minutes)."""
    rih = total_depth_m / rih_speed_m_per_min
    log_up = total_depth_m / logging_speed_m_per_min
    repeat = repeat_section_m / logging_speed_m_per_min
    total = rig_up_min + rih + log_up + repeat + rig_down_min
    return {"RIH (min)": rih, "Logging up (min)": log_up, "Repeat (min)": repeat,
            "Rig-up/down (min)": rig_up_min + rig_down_min, "Total (min)": total}


# Generic wireline pre-job HSE checklist (industry-standard themes).
HSE_CHECKLIST = {
    "Pre-job planning": [
        "Job Safety Analysis (JSA) completed and signed by all crew",
        "Well data, expected pressures, H2S/CO2 and BHT reviewed",
        "Pre-job safety meeting held; roles and stop-work authority confirmed",
        "Emergency response plan and muster point briefed",
    ],
    "Rig-up and pressure control": [
        "Exclusion zone barricaded; unnecessary personnel cleared",
        "Lifting equipment and slings inspected and within certification",
        "Sheaves anchored and rated for expected line tension",
        "Pressure test of lubricator / BOP / stuffing box done to required rating",
        "Wellhead isolation valves confirmed and positions verified",
    ],
    "Cable and tools": [
        "Cable condition inspected (broken wires, kinks, bird-caging)",
        "Weak point selected and verified against max tension calculation",
        "Tool string measured, tallied and sketched; fishing neck confirmed",
        "Tool function-tested and calibrated before running in hole",
    ],
    "Radioactive and explosive sources": [
        "Source handling only by trained and authorised personnel",
        "Source location in container checked; radiation survey recorded",
        "Radio silence / RF-safe procedure active when explosives are on location",
        "Source accounted for before leaving location",
    ],
    "Operations": [
        "Max line tension and speed limits posted at the winch",
        "Depth control checked against casing shoe / known marker",
        "Communication with driller maintained throughout the run",
        "Stuck-tool and cable-parted contingency plan reviewed",
    ],
    "Rig-down and handover": [
        "Well declared safe before breaking connections",
        "Housekeeping done; waste and spills handled per procedure",
        "Equipment inventory and post-job inspection logged",
        "Job report, tally and data delivered; incidents / near-misses reported",
    ],
}
