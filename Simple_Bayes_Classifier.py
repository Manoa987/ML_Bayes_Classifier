import pandas as pd

# Read the Excel file
file_path = "./data/Britanic_Tastes.xlsx"
df = pd.read_excel(file_path)

# Select only the desired columns
desired_columns = ['scones', 'cerveza', 'wiskey', 'avena', 'futbol', 'Nacionalidad']
vector_columns = ['scones', 'cerveza', 'wiskey', 'avena', 'futbol']
class_column = 'Nacionalidad'
filtered_df = df[desired_columns]
print(filtered_df)

# Extract all different class values: "Nacionalidad"
nationality_uniques = df[class_column].unique()

result_columns = [f"P({nationality}|vector)" for nationality in nationality_uniques]
data = {col: [1] for col in result_columns}
df_result = pd.DataFrame(data)

# Ask the user to input a vector
user_input = input(f"Enter the value for each {vector_columns} separated by , : ")
vector_values = user_input.split(',')
# Check if the lengths are not equal
if len(vector_values) != len(vector_columns):
    raise ValueError("Vector has the wrong size")
# Display the entered vector
print("Entered vector:", vector_values)

laplace_correction_needed = False

for nationality in nationality_uniques:
    number_of_class_elem = df[class_column].value_counts().get(nationality, 0)
    prob_nationality = number_of_class_elem / len(df[class_column])
    prob_total = prob_nationality

    if prob_nationality == 0:
        continue
    
    should_continue = True
    for i, vector_value in enumerate(vector_columns):
        val = 0
        for j, value in filtered_df[vector_value].items():
            if df[class_column][j] == nationality and str(value) == vector_values[i]:
                val += 1
        if val == 0:
            should_continue = False
            laplace_correction_needed = True
            continue
        prob_total *= (val / number_of_class_elem)
        print(val / number_of_class_elem)
    df_result[f"P({nationality}|vector)"][0] = prob_total
    
    if not should_continue:
        continue

if laplace_correction_needed:
    print("Correction of Laplace used")
    # Reset the value
    data = {col: [1] for col in result_columns}
    for nationality in nationality_uniques:
        number_of_class_elem = df[class_column].value_counts().get(nationality, 0)
        prob_nationality = number_of_class_elem / len(df[class_column])
        prob_total = prob_nationality

        print(prob_nationality)
        if prob_nationality == 0:
            continue
        
        should_continue = True
        for i, vector_value in enumerate(vector_columns):
            val = 1 # We start at one because of the laplace correction
            for j, value in filtered_df[vector_value].items():
                if df[class_column][j] == nationality and str(value) == vector_values[i]:
                    val += 1
            val /= (number_of_class_elem + df[class_column].nunique())
            print(val)
            prob_total *= val
        df_result[f"P({nationality}|vector)"][0] = prob_total
            
print(df_result)

# Calculate the sum of values in the first row of df_result
x = df_result.iloc[0].sum()
print(f"P(vector) = {x}")

for nationality in nationality_uniques:
    df_result[f"P({nationality}|vector)"][0] /= x
print(df_result)

