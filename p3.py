import pandas as pd

# Load the dataset
df = pd.read_csv("Iris.csv")

# Group by Species and calculate summary statistics for SepalLengthCm
grouped_stats = df.groupby("Species")["SepalLengthCm"].agg(['mean', 'median', 'min', 'max', 'std'])

# Display the result
print("Summary statistics (SepalLengthCm) grouped by Species:\n")
print(grouped_stats)

# Optional: Convert to dictionary/list form for each species
grouped_lists = df.groupby("Species")["SepalLengthCm"].apply(list).to_dict()

print("\nSepalLengthCm values grouped by species (as list):\n")
for species, values in grouped_lists.items():
    print(f"{species}: {values}")


species_list = df["Species"].unique()

for species in species_list:
    print(f"\n--- Statistics for {species} ---\n")
    
    species_data = df[df["Species"] == species]
    
    print(species_data.describe())
