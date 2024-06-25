import pandas as pd

def find_s_algorithm(df):
    # Extract the feature columns and the target column
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Initialize the most specific hypothesis (all values '0')
    specific_hypothesis = ['0'] * X.shape[1]
    
    # Iterate through each training example
    for i, example in X.iterrows():
        if y[i] == 'Yes':  # Only consider positive examples
            for attr_index in range(len(specific_hypothesis)):
                if specific_hypothesis[attr_index] == '0':
                    specific_hypothesis[attr_index] = example[attr_index]
                elif specific_hypothesis[attr_index] != example[attr_index]:
                    specific_hypothesis[attr_index] = '?'
    
    return specific_hypothesis

# Load dataset
df = pd.read_csv('training_data.csv')

# Run the FIND-S algorithm
specific_hypothesis = find_s_algorithm(df)

print("Most specific hypothesis:")
print(specific_hypothesis)
