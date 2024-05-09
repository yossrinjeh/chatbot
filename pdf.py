import pandas as pd
import os
import pdfkit

# Specify the path to the wkhtmltopdf executable
config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

# Read the CSV file into a DataFrame
df = pd.read_csv("Resume.csv")

# Print the first few rows of the DataFrame
print(df.head())

# Check if the 'link' column exists in the DataFrame
if 'link' in df.columns:
    # Drop the 'link' column
    df = df.drop(['link'], axis=1)
    print("The 'link' column has been successfully dropped.")
else:
    print("The 'link' column does not exist in the DataFrame.")

# Drop NaN values and check for null values
df.dropna(inplace=True)
print(df.isnull().sum())

# Shorten the 'ID' column values
#df['ID'] = [str(x)[::5] for x in df['ID']]

# Rename and rearrange the column names
df = df[['ID', 'Resume_str', 'Resume_html', 'Category']]
df.columns = ['ID', 'Resume_str', 'Resume_html', 'Category']
print(df.columns)

# Convert 'Category' values to uppercase
df['Category'] = df['Category'].str.upper()

# Save the DataFrame to a CSV file
df.to_csv("Resume.csv", index=False)

# Get unique category labels
labels = df['Category'].unique()

# Create directories for each category and convert HTML to PDF
parent_dir = os.path.join(os.getcwd(), "data")
for label in labels:
    temp = df[df['Category'] == label].copy()  # Make a copy of the slice
    temp.dropna(inplace=True)
    temp.reset_index(inplace=True, drop=True)
    path = os.path.join(parent_dir, label)
    os.makedirs(path, mode=0o666, exist_ok=True)  # Use exist_ok to avoid errors if directory already exists
    for i in range(temp.shape[0]):
        resume_id = temp['ID'].iloc[i]
        print(resume_id)
        output_file = os.path.join(path, f"{resume_id}.pdf")  # Append index for uniqueness
        with open(output_file, "wb") as f:
            f.write(pdfkit.from_string(temp['Resume_html'].iloc[i], False, configuration=config))
