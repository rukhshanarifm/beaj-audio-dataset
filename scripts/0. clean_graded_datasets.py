# %%
import pandas as pd
import os

# Define the directory containing the graded CSV files
graded_dir = "../data/graded"

# Initialize an empty list to store DataFrames
dataframes = []

cols_to_keep = ['audio_file_name', 
                'Question', 
                'Human Transcription', 
                'score']

# Define columns to exclude from scoring
exclude_columns = ['audio_file_name', 'Submitted Audio Link', 'Question', 'Human Transcription', 'Notes', 'grader', 'score', 'total_possible_score', 'score_json', 'profile_id']

# Iterate through all files in the directory
for file_name in os.listdir(graded_dir):
    if file_name.endswith(".csv"):
        file_path = os.path.join(graded_dir, file_name)
        df = pd.read_csv(file_path)
        df["grader"] = file_name.split("_")[0]  # Extract grader name from file name

        # Add a profile ID column
        df['profile_id'] = df['audio_file_name'].apply(lambda x: x.split('_')[1]).str.replace("profile", "") # Example logic for profile ID

        # Calculate 'score' and 'total_possible_score' for the current DataFrame
        df['score'] = df.drop(columns=exclude_columns, errors='ignore').sum(axis=1)
        df['total_possible_score'] = df.drop(columns=exclude_columns, errors='ignore').notnull().sum(axis=1) * 2

        # Create a JSON for each row with scores for columns not in exclude_columns
        df['score_json'] = df.drop(columns=exclude_columns, errors='ignore').apply(lambda row: row.dropna().to_dict(), axis=1)

        dataframes.append(df[exclude_columns])

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Display the combined DataFrame
print(combined_df[['audio_file_name', 'profile_id', 'score_json']].head())
# %%
combined_df.to_csv("../../audio-dataset/data/clean/graded_combined.csv", index=False)

# %%
