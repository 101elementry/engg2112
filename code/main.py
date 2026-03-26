import pandas as pd
from pathlib import Path
import kagglehub
import matplotlib.pyplot as plt

# Download dataset
DATASET_HANDLE = "karmansinghbains/ttc-delays-and-routes-2023"
local_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
print("Dataset downloaded to:", local_path)

# GTFS tables folder
gtfs_path = local_path / "TTC Routes and Schedules Data"

# Load all GTFS tables
tables = {}
for file in gtfs_path.glob("*.txt"):
    tables[file.stem] = pd.read_csv(file)

print("Tables loaded:")
print(tables.keys())

# Example: view stops table
print("\nStops table preview:")
print(tables["stops"].head())

# Load delay datasets
bus_delay = pd.read_excel(local_path / "ttc-bus-delay-data-2023.xlsx")
streetcar_delay = pd.read_excel(local_path / "ttc-streetcar-delay-data-2023.xlsx")
subway_delay = pd.read_excel(local_path / "ttc-subway-delay-data-2023.xlsx")

print("\nBus delay preview:")
print(bus_delay.head())

# Basic cleaning
bus_delay["Date"] = pd.to_datetime(bus_delay["Date"])
bus_delay["Route"] = bus_delay["Route"].astype(str)

print("\nBus delay summary:")
print(bus_delay[["Min Delay", "Min Gap"]].describe())

# ----------------------------
# Improved visualisations
# ----------------------------

# 1. Histogram of delay minutes
plt.figure(figsize=(8, 5))
plt.hist(bus_delay["Min Delay"].dropna(), bins=30)
plt.xlabel("Delay (minutes)")
plt.ylabel("Number of incidents")
plt.title("Distribution of Bus Delay Minutes")
plt.tight_layout()
plt.show()

# 2. Bar chart of top 10 incident types
incident_counts = bus_delay["Incident"].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.bar(incident_counts.index, incident_counts.values)
plt.xlabel("Incident Type")
plt.ylabel("Count")
plt.title("Top 10 Bus Delay Incident Types")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 3. Bar chart of delays by day of week
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_counts = bus_delay["Day"].value_counts().reindex(weekday_order)

plt.figure(figsize=(9, 5))
plt.bar(day_counts.index, day_counts.values)
plt.xlabel("Day")
plt.ylabel("Number of incidents")
plt.title("Bus Delay Incidents by Day of Week")
plt.tight_layout()
plt.show()

# 4. Top 10 routes by average delay
route_delay = (
    bus_delay.groupby("Route")["Min Delay"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
plt.bar(route_delay.index, route_delay.values)
plt.xlabel("Route")
plt.ylabel("Average delay (minutes)")
plt.title("Top 10 Routes by Average Delay")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 5. Monthly total delay trend
monthly_delay = (
    bus_delay.set_index("Date")["Min Delay"]
    .resample("ME")
    .sum()
)

plt.figure(figsize=(10, 5))
plt.plot(monthly_delay.index, monthly_delay.values, marker="o")
plt.xlabel("Month")
plt.ylabel("Total delay (minutes)")
plt.title("Monthly Total Bus Delay in 2023")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# Delay vs Weather Analysis
# ----------------------------

# NOTE: requires meteostat
# install with: pip install meteostat

try:
    from meteostat import Point
    from meteostat import Daily

    # Toronto location (TTC operates in Toronto)
    toronto = Point(43.65107, -79.347015)

    # Fetch daily weather for 2023
    weather = Daily(toronto, "2023-01-01", "2023-12-31").fetch()
    weather = weather.reset_index()

    # Prepare delay data aggregated by date
    daily_delay = (
        bus_delay.groupby("Date")["Min Delay"]
        .sum()
        .reset_index()
    )

    # Merge delay with weather data
    merged = pd.merge(daily_delay, weather, left_on="Date", right_on="time")

    print("\nMerged delay + weather preview:")
    print(merged[["Date", "Min Delay", "tavg", "prcp"]].head())

    # Scatter: Delay vs Temperature
    plt.figure(figsize=(8, 5))
    plt.scatter(merged["tavg"], merged["Min Delay"], alpha=0.6)
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Total Daily Bus Delay (minutes)")
    plt.title("Bus Delay vs Temperature")
    plt.tight_layout()
    plt.show()

    # Scatter: Delay vs Precipitation
    plt.figure(figsize=(8, 5))
    plt.scatter(merged["prcp"], merged["Min Delay"], alpha=0.6)
    plt.xlabel("Daily Precipitation (mm)")
    plt.ylabel("Total Daily Bus Delay (minutes)")
    plt.title("Bus Delay vs Precipitation")
    plt.tight_layout()
    plt.show()

except (ModuleNotFoundError, ImportError) as e:
    print("\nWeather analysis skipped: meteostat could not be imported.")
    print(f"Import error: {e}")
    print("Try reinstalling it with: pip install --upgrade --force-reinstall meteostat")