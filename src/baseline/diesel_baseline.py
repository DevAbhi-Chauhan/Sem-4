import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path


def diesel_fleet_sizing(trips_df, min_connection_min=5, save_outputs=True):
    """
    Diesel fleet sizing using:
    1. Theoretical cycle-time method
    2. Operational block scheduling (balanced greedy)
    """

    df = trips_df.copy()
    df = df.sort_values("start_time").reset_index(drop=True)

    # -----------------------------
    # TRUE CYCLE TIME (data-driven)
    # -----------------------------
    cycle_times = (
        df.groupby('cycle_id')
        .apply(lambda x: (x.end_time.max() - x.start_time.min()).total_seconds()/60)
    )

    cycle_time_min = cycle_times.mean()

    # -----------------------------
    # HEADWAY FROM PEAK HOUR
    # -----------------------------
    df['hour'] = df.start_time.dt.hour
    hourly_trips = df.groupby('hour').size()

    peak_hour = hourly_trips.idxmax()
    peak_trips = df[df.start_time.dt.hour == peak_hour]

    headways_by_dir = []

    for direction in peak_trips.direction.unique():
        dir_trips = peak_trips[peak_trips.direction == direction].sort_values("start_time")
        headways = dir_trips.start_time.diff().dropna()

        if len(headways) > 0:
            headways_by_dir.append(headways.dt.total_seconds()/60)

    if len(headways_by_dir) > 0:
        avg_headway_min = pd.concat(headways_by_dir).mean()
        N_diesel_theoretical = int(np.ceil(cycle_time_min / avg_headway_min))
    else:
        avg_headway_min = np.nan
        N_diesel_theoretical = 0

    # -----------------------------
    # BALANCED BLOCK SCHEDULING
    # -----------------------------
    active_buses = []  # (bus_id, last_end, trip_count)
    blocks = []

    for _, trip in df.iterrows():

        available_indices = []

        # Find feasible buses
        for i, (bus_id, last_end, trip_count) in enumerate(active_buses):
            if trip.start_time >= last_end + timedelta(minutes=min_connection_min):
                available_indices.append(i)

        if available_indices:
            # Choose least-used feasible bus
            best_i = min(
                available_indices,
                key=lambda i: active_buses[i][2]
            )

            bus_id, _, trip_count = active_buses[best_i]

            active_buses[best_i] = (
                bus_id,
                trip.end_time,
                trip_count + 1
            )

        else:
            # Create new bus
            bus_id = len(active_buses)
            active_buses.append((bus_id, trip.end_time, 1))

        blocks.append({
            'trip_id': trip.trip_id,
            'bus_id': bus_id
        })

    blocks_df = pd.DataFrame(blocks)
    N_diesel_block = int(blocks_df.bus_id.nunique())

    # -----------------------------
    # SAVE
    # -----------------------------
    if save_outputs:
        results_dir = Path.cwd() / 'results' / 'synthetic'
        results_dir.mkdir(parents=True, exist_ok=True)
        blocks_df.to_csv(results_dir / 'diesel_blocks.csv', index=False)

    return {
        'N_diesel_theoretical': N_diesel_theoretical,
        'N_diesel_block': N_diesel_block,
        'cycle_time_min': cycle_time_min,
        'avg_headway_min': avg_headway_min,
        'peak_hour': peak_hour
    }


if __name__ == "__main__":

    trips = pd.read_csv(
        Path.cwd() / 'data' / 'synthetic' / 'trips.csv',
        parse_dates=['start_time', 'end_time']
    )

    results = diesel_fleet_sizing(trips)

    print("\nDiesel Fleet Baseline")
    print("---------------------")
    print(f"Theoretical fleet: {results['N_diesel_theoretical']}")
    print(f"Block fleet: {results['N_diesel_block']}")
    print(f"Cycle time: {results['cycle_time_min']:.1f} min")
    print(f"Avg peak headway: {results['avg_headway_min']:.2f} min")
