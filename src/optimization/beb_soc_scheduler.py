import pandas as pd
from datetime import timedelta


def beb_soc_scheduler(
    trips_df,
    battery_kwh=350,
    soc_min=0.15,
    start_soc=1.0
):
    """
    Greedy BEB scheduler with SOC tracking.
    Overnight charging only.
    """

    df = trips_df.copy()
    df = df.sort_values("start_time").reset_index(drop=True)

    buses = []  
    # each bus: (bus_id, last_end, soc_remaining)

    assignments = []

    for _, trip in df.iterrows():

        assigned = False

        for i, (bus_id, last_end, soc) in enumerate(buses):

            # time feasibility
            if trip.start_time < last_end + timedelta(minutes=5):
                continue

            # SOC feasibility
            soc_after = soc - trip.energy_kwh / battery_kwh

            if soc_after >= soc_min:
                buses[i] = (bus_id, trip.end_time, soc_after)

                assignments.append({
                    "trip_id": trip.trip_id,
                    "bus_id": bus_id,
                    "soc_after": soc_after
                })

                assigned = True
                break

        # if no bus feasible → new bus
        if not assigned:
            bus_id = len(buses)

            soc_after = start_soc - trip.energy_kwh / battery_kwh

            buses.append((bus_id, trip.end_time, soc_after))

            assignments.append({
                "trip_id": trip.trip_id,
                "bus_id": bus_id,
                "soc_after": soc_after
            })

    assign_df = pd.DataFrame(assignments)

    return {
        "N_BEB_SOC": len(buses),
        "assignments": assign_df
    }
