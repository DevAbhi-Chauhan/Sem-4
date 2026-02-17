import pandas as pd
import numpy as np
from pathlib import Path


def analytical_beb_fleet(
    trips_df,
    diesel_results,
    battery_kwh=350,
    usable_fraction=0.85,
    save_outputs=True
):
    """
    Analytical BEB fleet sizing (overnight charging only).
    """

    df = trips_df.copy()

    # --------------------------
    # ENERGY PER CYCLE
    # --------------------------
    cycle_energy = (
        df.groupby("cycle_id")
        .energy_kwh.sum()
    )

    avg_cycle_energy = cycle_energy.mean()

    # --------------------------
    # USABLE BATTERY
    # --------------------------
    usable_energy = battery_kwh * usable_fraction

    # --------------------------
    # CYCLES PER CHARGE
    # --------------------------
    cycles_per_charge = usable_energy / avg_cycle_energy

    # --------------------------
    # TOTAL DAILY CYCLES
    # --------------------------
    total_cycles = df.cycle_id.nunique()

    # --------------------------
    # RANGE-CONSTRAINED FLEET
    # --------------------------
    N_range = int(np.ceil(total_cycles / cycles_per_charge))

    # --------------------------
    # FINAL BEB FLEET
    # --------------------------
    N_diesel = diesel_results["N_diesel_block"]

    N_BEB = max(N_diesel, N_range)

    replacement_ratio = N_BEB / N_diesel

    results = {
        "battery_kwh": battery_kwh,
        "usable_energy_kwh": usable_energy,
        "avg_cycle_energy_kwh": avg_cycle_energy,
        "cycles_per_charge": cycles_per_charge,
        "total_cycles": total_cycles,
        "N_range": N_range,
        "N_BEB": N_BEB,
        "replacement_ratio": replacement_ratio
    }

    # --------------------------
    # SAVE
    # --------------------------
    if save_outputs:
        results_dir = Path.cwd() / "results" / "synthetic"
        results_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([results]).to_csv(
            results_dir / "beb_analytical_summary.csv",
            index=False
        )

    return results


if __name__ == "__main__":

    trips = pd.read_csv(
        Path.cwd() / "data" / "synthetic" / "trips.csv"
    )

    diesel_summary = pd.read_csv(
        Path.cwd() / "results" / "synthetic" / "diesel_validation_summary.csv"
    ).iloc[0].to_dict()

    results = analytical_beb_fleet(trips, diesel_summary)

    print("\nAnalytical BEB Fleet")
    print("-------------------")
    for k, v in results.items():
        print(f"{k}: {v}")
