import pandas as pd
import numpy as np

def get_domains(df):
    """Extracts the domain of each attribute (unique values)."""
    domains = {}
    for col in df.columns[:-1]:
        domains[col] = list(df[col].unique())
    return domains

def more_general(h1, h2):
    """Returns True if hypothesis h1 is more general than h2."""
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == "?" or (x != "0" and (x == y or y == "0"))
        more_general_parts.append(mg)
    return all(more_general_parts)

def fulfills(example, hypothesis):
    """Checks if an example fulfills a hypothesis."""
    return all(h == "?" or h == x for h, x in zip(hypothesis, example))

def min_generalizations(h, x):
    """Minimizes generalizations to cover x."""
    h_new = list(h)
    for i in range(len(h)):
        if not fulfills(x, h_new):
            h_new[i] = x[i] if h_new[i] == "0" else "?"
    return [tuple(h_new)]

def min_specializations(h, domains, x):
    """Minimizes specializations to exclude x."""
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if x[i] != val:
                    h_new = list(h)
                    h_new[i] = val
                    results.append(tuple(h_new))
        elif h[i] != "0":
            h_new = list(h)
            h_new[i] = "0"
            results.append(tuple(h_new))
    return results

def candidate_elimination(df):
    """Implements the Candidate-Elimination algorithm."""
    domains = get_domains(df)
    n_features = len(df.columns) - 1

    G = {("?",) * n_features}
    S = {("0",) * n_features}

    for i, row in df.iterrows():
        x = tuple(row[:-1])
        y = row[-1]
        if y:  # Positive example
            G = {g for g in G if fulfills(x, g)}
            S_new = set(S)
            for s in S:
                if not fulfills(x, s):
                    S_new.remove(s)
                    S_new.update([h for h in min_generalizations(s, x) if any([more_general(g, h) for g in G])])
            S = S_new
        else:  # Negative example
            S = {s for s in S if not fulfills(x, s)}
            G_new = set(G)
            for g in G:
                if fulfills(x, g):
                    G_new.remove(g)
                    G_new.update([h for h in min_specializations(g, domains, x) if any([more_general(h, s) for s in S])])
            G = G_new

    return S, G

# Load dataset
df = pd.read_csv('training_data.csv')

# Encode target column
df['PlayTennis'] = df['PlayTennis'].apply(lambda x: 1 if x == 'Yes' else 0)

# Run the Candidate-Elimination algorithm
S, G = candidate_elimination(df)

print("Most specific hypotheses:")
for s in S:
    print(s)

print("\nMost general hypotheses:")
for g in G:
    print(g)
