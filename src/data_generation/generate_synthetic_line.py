import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_line(
    route_length_km=12,
    num_stops=24,
    service_start='06:00',
    service_end='22:00',
    peak_start='07:00',
    peak_end='10:00',
    peak_headway_min=10,
    offpeak_headway_min=20,
    layover_min=6,
    seed=42,
    save_outputs=True
):
    """
    Generate a realistic bidirectional transit line with timetable and energy estimates.

    Modeling assumptions aligned with project documentation:
    --------------------------------------------------------
    • Single-line corridor
    • Deterministic travel times with mild stochasticity
    • Speed-based running time
    • Round-trip vehicle cycles
    • Linear energy model
    • Homogeneous fleet
    """

    np.random.seed(seed)

    # ----------------------------
    # Route Geometry
    # ----------------------------
    stop_spacing_m = route_length_km * 1000 / (num_stops - 1)

    stops = pd.DataFrame({
        'stop_id': range(1, num_stops + 1),
        'position_m': np.linspace(0, route_length_km * 1000, num_stops),
        # Gentle rolling elevation profile
        'elevation_m': np.cumsum(np.random.normal(0, 2, num_stops))
    })

    # Compute directional elevation work (simple but meaningful)
    total_ascent = stops.elevation_m.diff().clip(lower=0).sum()
    elevation_kwh_per_trip = 0.0003 * total_ascent  # small but realistic penalty

    # ----------------------------
    # Time Setup
    # ----------------------------
    service_start_dt = datetime.strptime(service_start, '%H:%M')
    service_end_dt = datetime.strptime(service_end, '%H:%M')

    peak_start_time = datetime.strptime(peak_start, '%H:%M').time()
    peak_end_time = datetime.strptime(peak_end, '%H:%M').time()

    trips = []
    current_time = service_start_dt
    cycle_id = 0

    while current_time < service_end_dt:

        is_peak = peak_start_time <= current_time.time() <= peak_end_time

        # Headway with mild variability
        if is_peak:
            headway = peak_headway_min + np.random.randint(-1, 2)
            avg_speed = 18  # km/h
        else:
            headway = offpeak_headway_min + np.random.randint(-2, 3)
            avg_speed = 25  # km/h

        # Running time from physics
        run_time_min = (route_length_km / avg_speed) * 60
        run_time_min += np.random.normal(0, 1.5)  # small stochasticity

        run_time = timedelta(minutes=max(run_time_min, 5))  # safety bound
        layover = timedelta(minutes=layover_min)

        # -------- OUTBOUND --------
        start_out = current_time
        end_out = start_out + run_time

        # -------- INBOUND --------
        start_in = end_out + layover
        end_in = start_in + run_time

        trips.append({
            'trip_id': len(trips) + 1,
            'cycle_id': cycle_id,
            'direction': 0,
            'start_time': start_out,
            'end_time': end_out,
            'distance_km': route_length_km,
            'num_stops': num_stops,
            'is_peak': is_peak
        })

        trips.append({
            'trip_id': len(trips) + 1,
            'cycle_id': cycle_id,
            'direction': 1,
            'start_time': start_in,
            'end_time': end_in,
            'distance_km': route_length_km,
            'num_stops': num_stops,
            'is_peak': is_peak
        })

        cycle_id += 1
        current_time += timedelta(minutes=headway)

    trips_df = pd.DataFrame(trips)

    # Enforce dtypes (optimization-friendly)
    trips_df['direction'] = trips_df['direction'].astype(int)
    trips_df['cycle_id'] = trips_df['cycle_id'].astype(int)

    # ----------------------------
    # ENERGY MODEL
    # E = α*distance + β*stops + γ + elevation + noise
    # ----------------------------

    alpha = 1.2
    beta = 0.04
    gamma = 2

    base_energy = (
        alpha * trips_df.distance_km +
        beta * trips_df.num_stops +
        gamma +
        elevation_kwh_per_trip
    )

    noise = np.abs(np.random.normal(0, 0.05 * trips_df.distance_km))

    trips_df['energy_kwh'] = base_energy + noise

    # Safety clamp (never allow unrealistic low energy)
    trips_df['energy_kwh'] = np.maximum(
        trips_df['energy_kwh'],
        0.6 * trips_df.distance_km
    )

    # ----------------------------
    # Save Outputs
    # ----------------------------
    if save_outputs:

        base_dir = Path.cwd()

        data_dir = base_dir / 'data' / 'synthetic'
        results_dir = base_dir / 'results' / 'synthetic'

        data_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        trips_df.to_csv(data_dir / 'trips.csv', index=False)
        stops.to_csv(data_dir / 'stops.csv', index=False)

    return trips_df, stops


if __name__ == "__main__":
    trips, stops = generate_synthetic_line()

    print("\nSynthetic Line Generated Successfully")
    print("------------------------------------")
    print(f"Total trips: {len(trips)}")
    print(f"Total cycles: {trips.cycle_id.nunique()}")
    print(f"Average trip energy: {trips.energy_kwh.mean():.2f} kWh")
    print(trips[['trip_id', 'cycle_id', 'direction', 'start_time', 'energy_kwh']].head())
