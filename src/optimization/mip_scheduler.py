import pandas as pd
import pyomo.environ as pyo


def solve_beb_mip(trips_df, battery_kwh=350, soc_min=0.15):

    df = trips_df.copy().reset_index(drop=True)
    n = len(df)

    usable_energy = battery_kwh * (1 - soc_min)

    model = pyo.ConcreteModel()
    model.T = range(n)

    # -------- Feasible connections --------
    feasible = []
    for i in model.T:
        for j in model.T:
            if i == j:
                continue
            if df.start_time[j] >= df.end_time[i]:
                if df.energy_kwh[i] + df.energy_kwh[j] <= usable_energy:
                    feasible.append((i,j))

    model.A = pyo.Set(initialize=feasible)

    model.x = pyo.Var(model.A, domain=pyo.Binary)
    model.y = pyo.Var(model.T, domain=pyo.Binary)

    # -------- Each trip has one predecessor or start --------
    def one_pred_rule(m,j):
        return sum(m.x[i,j] for (i,j2) in m.A if j2==j) + m.y[j] == 1
    model.one_pred = pyo.Constraint(model.T, rule=one_pred_rule)

    # -------- Each trip has at most one successor --------
    def one_succ_rule(m, i):
        succ = [(i2, j) for (i2, j) in m.A if i2 == i]

        if len(succ) == 0:
            return pyo.Constraint.Skip

        return sum(m.x[i, j] for (i2, j) in succ) <= 1


    # -------- Objective --------
    model.obj = pyo.Objective(
        expr=sum(model.y[i] for i in model.T),
        sense=pyo.minimize
    )

    solver = pyo.SolverFactory("highs")
    solver.solve(model)

    fleet = sum(pyo.value(model.y[i]) for i in model.T)

    return int(fleet)
