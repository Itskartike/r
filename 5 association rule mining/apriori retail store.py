transactions = [
    ['Rice', 'Wheat', 'Oil'],
    ['Rice', 'Oil'],
    ['Wheat', 'Sugar'],
    ['Rice', 'Wheat', 'Sugar'],
    ['Rice', 'Oil', 'Sugar']
]
total_transactions = len(transactions)
min_support = 0.4
min_confidence = 0.6
def calculate_support(itemset):
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count / total_transactions
items = set().union(*transactions)
frequent_1 = {}

for item in items:
    sup = calculate_support({item})
    if sup >= min_support:
        frequent_1[frozenset([item])] = sup

print("Frequent 1-Itemsets:")
for k, v in frequent_1.items():
    print(set(k), ":", v)
from itertools import combinations

frequent_2 = {}
for pair in combinations(items, 2):
    sup = calculate_support(set(pair))
    if sup >= min_support:
        frequent_2[frozenset(pair)] = sup

print("\nFrequent 2-Itemsets:")
for k, v in frequent_2.items():
    print(set(k), ":", v)
frequent_3 = {}
for triple in combinations(items, 3):
    sup = calculate_support(set(triple))
    if sup >= min_support:
        frequent_3[frozenset(triple)] = sup
print("\nFrequent 3-Itemsets:")
for k, v in frequent_3.items():
    print(set(k), ":", v)
def calculate_confidence(A, B):
    return calculate_support(A.union(B)) / calculate_support(A)
rules = []
for itemset in frequent_2:
    for item in itemset:
        A = frozenset([item])
        B = itemset - A
        conf = calculate_confidence(set(A), set(B))
        if conf >= min_confidence:
            rules.append((A, B, conf))
print("\nAssociation Rules:")
for rule in rules:
    print(set(rule[0]), "→", set(rule[1]), "Confidence:", rule[2])
def calculate_lift(A, B):
    return calculate_confidence(A, B) / calculate_support(B)
print("\nAssociation Rules with Lift:")
for rule in rules:
    lift = calculate_lift(set(rule[0]), set(rule[1]))
    print(set(rule[0]), "→", set(rule[1]), "Lift:", lift)
