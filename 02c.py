import math
import pandas as pd

df = pd.read_csv('/Users/rahulsunilkumar/Desktop/freight_demand_estimates.csv')
DEMAND_COLUMN = "Daily demand (in million lbs)"

optimal_trucks = [0] * len(df)
costs_per_center = [0.0] * len(df)

for s in range(len(df)):

    mean_million = df.loc[s, DEMAND_COLUMN] # Mean Daily Demand for row s.
    D_s = mean_million * 1_000_000 # Convert million lbs to lbs.
    max_trucks = math.ceil(D_s / 100_000) + 1 # Search range for integer number of trucks

    best_cost = float('inf')
    best_N = 0

    for N in range(max_trucks + 1):

        cost_val = 2500 * N # Fixed truck cost: $2,500 per truck
        capacity = 100_000 * N # Outsourcing cost if capacity < demand

        if capacity < D_s:
            cost_val += 0.10 * (D_s - capacity)

        # Update best solution
        if cost_val < best_cost:
            best_cost = cost_val
            best_N = N

    # Store results for sort center s
    optimal_trucks[s] = best_N
    costs_per_center[s] = best_cost

# Include summation of total cost.
total_cost = sum(costs_per_center)

# Print out the results.
for s in range(len(df)):
    print(f"Sort Center #{s + 1}:")
    print(f"  Demand (lbs) approx = {df.loc[s, DEMAND_COLUMN] * 1_000_000:.0f}")
    print(f"  Optimal Trucks = {optimal_trucks[s]}")
    print(f"  Daily Cost = ${costs_per_center[s]:.2f}")
    print("---------------------------------")

print(f"TOTAL DAILY COST ACROSS {len(df)} CENTERS = ${total_cost:.2f}")
