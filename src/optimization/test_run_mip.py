from src.optimization.mip_scheduler import solve_beb_mip
import pandas as pd
import sys

print('Python', sys.version)
df = pd.DataFrame({'start_time':[0,10,20],'end_time':[10,20,30],'energy_kwh':[100,100,100]})
print('Input df:\n', df)
print('Result ->', solve_beb_mip(df))
