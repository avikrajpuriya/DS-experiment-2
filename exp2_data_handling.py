import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("urban_climate.csv")
print("Dataset loaded successfully")
print(df.head())


# -----------------------------------------------------------------

df = pd.read_csv("urban_climate.csv")
print("Dataset loaded successfully")
df.head()

# -----------------------------------------------------------------
print("Shape:", df.shape)
print("Columns:", df.columns)
df.info()

# -----------------------------------------------------------------
df.describe()

# -----------------------------------------------------------------
print(df.isnull().sum())

# -----------------------------------------------------------------
# Fill missing numerical values with their column average
df.fillna(df.mean(numeric_only=True), inplace=True)

# Fill missing text values with the most common value
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values after filling:")
print(df.isnull().sum().sum())

# -----------------------------------------------------------------
# Top 5 highest temperature records
df.sort_values("temperature_celsius", ascending=False).head(5)

# -----------------------------------------------------------------
# Top 8 highest UHI intensity records
df.sort_values("urban_heat_island_intensity", ascending=False).head(8)

# -----------------------------------------------------------------
print(df["city"].value_counts())

# -----------------------------------------------------------------
print(df["continent"].value_counts())

# -----------------------------------------------------------------
# Sort rows by city name
df.sort_values("city").head()

# -----------------------------------------------------------------
# Sort column names alphabetically
df[sorted(df.columns)].head()

# -----------------------------------------------------------------
# iloc uses row and column positions
print(df.iloc[0, 0])

# loc uses row and column labels
print(df.loc[0, "city"])

# -----------------------------------------------------------------
# Case 1: Temperature above 25 and humidity above 60
df[(df["temperature_celsius"] > 25) & (df["humidity_percent"] > 60)].head()

# -----------------------------------------------------------------
# Case 2: Cities in Asia with rainfall above 100 mm
df[(df["continent"] == "Asia") & (df["precipitation_mm"] > 100)].head()

# -----------------------------------------------------------------
# Case 3: Records after 2015 with wind speed above 10
df[(df["year"] > 2015) & (df["wind_speed_ms"] > 10)].head()

# -----------------------------------------------------------------
print("Minimum values:")
print(df[["temperature_celsius", "humidity_percent", "precipitation_mm",
          "wind_speed_ms", "urban_heat_island_intensity"]].min())

print("Maximum values:")
print(df[["temperature_celsius", "humidity_percent", "precipitation_mm",
          "wind_speed_ms", "urban_heat_island_intensity"]].max())

# -----------------------------------------------------------------
# Average temperature for each continent
df.groupby("continent")["temperature_celsius"].mean()

# -----------------------------------------------------------------
# Average humidity for each country and city
df.groupby(["country", "city"])["humidity_percent"].mean().head(10)

# -----------------------------------------------------------------
# Convert temperature from Celsius to Fahrenheit
df["temperature_fahrenheit"] = df["temperature_celsius"] * 9/5 + 32

df[["city", "temperature_celsius", "temperature_fahrenheit"]].head()

# -----------------------------------------------------------------
# Average and maximum temperature for each continent
df.groupby("continent")["temperature_celsius"].agg(["mean", "max"])

# -----------------------------------------------------------------
# Average humidity and total rainfall for each country
df.groupby("country").agg({
    "humidity_percent": "mean",
    "precipitation_mm": "sum"
})

# -----------------------------------------------------------------
# Select records from Asia
asia = df[df["continent"] == "Asia"]
asia.head()

# -----------------------------------------------------------------
# Select continents where average temperature is above 15
avg_temp = df.groupby("continent")["temperature_celsius"].mean()
avg_temp[avg_temp > 15]

# -----------------------------------------------------------------
print("Correlation between temperature and humidity:")
print(df["temperature_celsius"].corr(df["humidity_percent"]))

# -----------------------------------------------------------------
# Min-Max normalization changes values to a range between 0 and 1
minimum = df["temperature_celsius"].min()
maximum = df["temperature_celsius"].max()

df["temperature_normalized"] = (
    (df["temperature_celsius"] - minimum) / (maximum - minimum)
)

df[["temperature_celsius", "temperature_normalized"]].head()

# -----------------------------------------------------------------
# Make two small tables from the dataset
city_table = df[["city", "country"]].drop_duplicates()
temp_table = df.groupby("city")["temperature_celsius"].mean().reset_index()

# Merge using the city column
merged = pd.merge(city_table, temp_table, on="city")
merged.head()

# -----------------------------------------------------------------
# Join using city as the index
table1 = city_table.set_index("city")
table2 = temp_table.set_index("city")

joined = table1.join(table2)
joined.head()

# -----------------------------------------------------------------
# Concatenate two parts of the dataset
part1 = df.head(5)
part2 = df.iloc[5:10]

combined = pd.concat([part1, part2])
combined

# -----------------------------------------------------------------
df["temperature_celsius"].plot(kind="hist", bins=15)
plt.title("Temperature Distribution")
plt.xlabel("Temperature in Celsius")
plt.ylabel("Number of Records")
plt.show()
