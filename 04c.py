import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Read the CSV file with Pandas
# Load demand estimates where demand is normally distributed
df = pd.read_csv('/Users/rahulsunilkumar/Desktop/freight_demand_estimates.csv')

# Convert demand from "million lbs" to lbs
# We need to work in pounds instead of million pounds
df["mean_lbs"] = df["Daily demand (in million lbs)"] * 1_000_000
df["stdev_lbs"] = df["Standard deviation of demand (in million lbs)"] * 1_000_000

# 3) Define the Expected Cost Function
def expected_cost(N, mu, sigma):
    """
    Computes the expected daily cost for a given number of trucks N. Recall the following parameters.
    
    - N: Number of trucks allocated
    - mu: Mean daily demand (lbs)
    - sigma: Standard deviation of daily demand (lbs)
    
    The cost function consists of:
      - Fixed truck cost: $2,500 per truck per day
      - Outsourcing cost: $0.10 per pound for excess demand above truck capacity
    """

  
    if N < 0:
        return float('inf') # If N is negative (invalid), return infinite cost to prevent selection

    capacity = 100_000 * N                                                       # Each truck has a fixed capacity of 100,000 lbs

    # If sigma is zero, demand is fully deterministic, no randomness involved
    if sigma == 0.0:
        leftover = max(0, mu - capacity)                                         # Demand exceeding truck capacity
        return 2500 * N + 0.10 * leftover

    Z = (capacity - mu) / sigma                                                  # Compute standard normal Z-score for demand exceeding truck capacity.
    expected_excess = (mu - capacity) * (1 - norm.cdf(Z)) + sigma * norm.pdf(Z)  # Compute the expected leftover demand (actually a rectified Gaussian integral).
    expected_excess = max(0, expected_excess)                                    # Ensure numerical stability: expected excess should not be negative.
  
    return 2500 * N + 0.10 * expected_excess                                     # Total expected cost: fixed truck cost + expected outsourcing cost

# Lists to store optimal results for each sort center
optimal_trucks = []
best_costs = []

# Solve for each sort center independently
for index, row in df.iterrows():
    mu = row["mean_lbs"]   # Mean demand in pounds.
    sigma = row["stdev_lbs"]  # Standard deviation of demand (lbs).

    # Determine the search range for optimal truck allocation
    # Upper bound: We assume demand is covered up to (mu + 3σ), plus a buffer
    max_trucks = int(np.ceil((mu + 3.0 * sigma) / 100_000.0)) + 2                # Adding a little buffer.
    candidate_N = np.arange(0, max_trucks + 1)                                   # Define candidate truck allocations from 0 to max_trucks.
    costs = np.array([expected_cost(n, mu, sigma) for n in candidate_N])         # Compute expected cost for each candidate truck allocation.

    # Find the truck count that results in the lowest expected cost. 
    best_N = candidate_N[np.argmin(costs)]
    best_C = np.min(costs)

    # Store the optimal truck count and cost for this sort center.
    optimal_trucks.append(best_N)
    best_costs.append(best_C)

    # Print results for this sort center.
    print(f"Sort Center {index+1}:")
    print(f"  Mean demand: {mu:,.0f} lbs")
    print(f"  Std dev: {sigma:,.0f} lbs")
    print(f"  Range searched: N in [0..{max_trucks}]")
    print(f"  Optimal trucks: {best_N}")
    print(f"  Expected daily cost: ${best_C:,.2f}\n")

# Store results in a DataFrame.
df["Optimal Trucks"] = optimal_trucks
df["Expected Daily Cost"] = best_costs

# Print summary of final truck allocations.
print("\nFinal Results:")
print(df[["Sort center #", "Optimal Trucks", "Expected Daily Cost"]])

# Visualization of Optimal Truck Allocations.
plt.figure(figsize=(12, 6))
plt.bar(df["Sort center #"], df["Optimal Trucks"], color='blue', alpha=0.7)
plt.xlabel("Sort Center")
plt.ylabel("Optimal Number of Trucks")
plt.title("Optimal Truck Allocation Across Sort Centers")
plt.xticks(df["Sort center #"], rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
