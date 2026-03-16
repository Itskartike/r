#Aim: Association Rule Mining

#Q1.	Association Rule Mining using Apriori Algorithm
#Step 1: Define the Transaction Dataset
transactions = [
    {'Milk', 'Bread'},
    {'Bread', 'Diaper', 'Beer', 'Eggs'},
    {'Milk', 'Diaper', 'Beer', 'Cola'},
    {'Bread', 'Milk', 'Diaper', 'Beer'},
    {'Bread', 'Milk', 'Diaper', 'Cola'}
]
total_transactions = len(transactions)
min_support = 0.4
min_confidence = 0.6

#Step 2: Define Support Calculation Function
def calculate_support(itemset):
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count / total_transactions

#Step 3: Generate Frequent 1-Itemsets
items = set().union(*transactions)
frequent_1 = {}
for item in items:
    sup = calculate_support({item})
    if sup >= min_support:
        frequent_1[frozenset([item])] = sup
print("Frequent 1-Itemsets:")
for k, v in frequent_1.items():
    print(set(k), ":", v)

#Step 4: Generate Frequent 2-Itemsets
from itertools import combinations
frequent_2 = {}
for pair in combinations(items, 2):
    sup = calculate_support(set(pair))
    if sup >= min_support:
        frequent_2[frozenset(pair)] = sup
print("\nFrequent 2-Itemsets:")
for k, v in frequent_2.items():
    print(set(k), ":", v)

#Step 5: Generate Frequent 3-Itemsets
frequent_3 = {}
for triple in combinations(items, 3):
    sup = calculate_support(set(triple))
    if sup >= min_support:
        frequent_3[frozenset(triple)] = sup
print("\nFrequent 3-Itemsets:")
for k, v in frequent_3.items():
    print(set(k), ":", v)

#Step 6: Generate Association Rules
Code
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

#Step 7: Calculate Lift
def calculate_lift(A, B):
    return calculate_confidence(A, B) / calculate_support(B)
print("\nAssociation Rules with Lift:")
for rule in rules:
    lift = calculate_lift(set(rule[0]), set(rule[1]))
    print(set(rule[0]), "→", set(rule[1]), "Lift:", lift)


#Q2.	Association Rule Mining using FP-Growth Algorithm
#Step 1: Define the Transaction Dataset
transactions = [
    ['Milk', 'Bread'],
    ['Bread', 'Diaper', 'Beer', 'Eggs'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk', 'Diaper', 'Cola']
]
min_support = 2   # absolute support count

#Step 2: Count Frequency of Each Item
from collections import defaultdict
item_count = defaultdict(int)
for transaction in transactions:
    for item in transaction:
        item_count[item] += 1
print("Item Frequencies:")
for item, count in item_count.items():
    print(item, ":", count)

#Step 3: Remove Infrequent Items
frequent_items = {item: count for item, count in item_count.items()
                  if count >= min_support}
print("\nFrequent Items:")
print(frequent_items)

#Step 4: Sort Transactions by Frequency
sorted_transactions = []
for transaction in transactions:
    filtered = [item for item in transaction if item in frequent_items]
    sorted_trans = sorted(filtered,
                          key=lambda x: frequent_items[x],
                          reverse=True)
    sorted_transactions.append(sorted_trans)
print("\nSorted Transactions:")
for t in sorted_transactions:
    print(t)

#Step 5: Define FP-Tree Node Structure
class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

#Step 6: Build the FP-Tree
root = FPNode(None, 0, None)

def insert_tree(transaction, node):
    if len(transaction) == 0:
        return
    first = transaction[0]
    if first in node.children:
        node.children[first].count += 1
    else:
        node.children[first] = FPNode(first, 1, node)
    insert_tree(transaction[1:], node.children[first])
for transaction in sorted_transactions:
    insert_tree(transaction, root)

#Step 7: Display FP-Tree (Traversal)
def display_tree(node, indent=0):
    for child in node.children.values():
        print(" " * indent, child.item, ":", child.count)
        display_tree(child, indent + 4)
print("\nFP-Tree:")
display_tree(root)

#Step 8: Mine Frequent Pattern
frequent_patterns = {}
for item, count in frequent_items.items():
    frequent_patterns[(item,)] = count
print("\nFrequent Patterns:")
for pattern, count in frequent_patterns.items():
    print(pattern, ":", count)


#(mainQ1.)	A supermarket wants to analyze customer purchasing behavior in order to discover relationships among products that are frequently bought together. The transaction data collected over five days is shown in the table below.
Transaction ID	Items Purchased
T1	Bread, Milk
T2	Bread, Diaper, Beer, Eggs
T3	Milk, Diaper, Beer, Cola
T4	Bread, Milk, Diaper, Beer
T5	Bread, Milk, Diaper, Cola
Using the above dataset, convert the transaction data into a suitable binary format and apply the Apriori algorithm with a minimum support of 40% to identify frequent itemsets. Further, generate association rules with a minimum confidence of 60% and interpret the discovered rules using support, confidence, and lift measures.

# Step 1: Define Transactions
transactions = [
    {'Bread','Milk'},
    {'Bread','Diaper','Beer','Eggs'},
    {'Milk','Diaper','Beer','Cola'},
    {'Bread','Milk','Diaper','Beer'},
    {'Bread','Milk','Diaper','Cola'}
]

total_transactions = len(transactions)
min_support = 0.4
min_confidence = 0.6

# Step 2: Support Function
def calculate_support(itemset):
    count = 0
    for t in transactions:
        if itemset.issubset(t):
            count += 1
    return count / total_transactions

# Step 3: Generate Frequent Itemsets
from itertools import combinations

items = set().union(*transactions)
frequent = {}

for i in range(1,4):
    for combo in combinations(items, i):
        sup = calculate_support(set(combo))
        if sup >= min_support:
            frequent[frozenset(combo)] = sup

print("Frequent Itemsets:")
for k,v in frequent.items():
    print(set(k), ":", v)

# Step 4: Generate Association Rules
def calculate_confidence(A,B):
    return calculate_support(A.union(B)) / calculate_support(A)

print("\nAssociation Rules:")
for itemset in frequent:
    if len(itemset) > 1:
        for item in itemset:
            A = frozenset([item])
            B = itemset - A
            conf = calculate_confidence(set(A), set(B))
            if conf >= min_confidence:
                print(set(A), "→", set(B), "Confidence:", conf)


#(mainQ2.)An online shopping platform wants to improve its product recommendation system by analyzing customer purchase history. The following dataset represents items purchased by different users.
User ID	Items Purchased
U1	Mobile, Charger
U2	Mobile, Earphones, Power Bank
U3	Laptop, Mouse
U4	Mobile, Charger, Earphones
U5	Laptop, Mouse, Keyboard
Using the given data, preprocess the transactions into an appropriate format and apply the FP-Growth algorithm to discover frequent itemsets. Generate strong association rules from the frequent itemsets and explain how these rules can be useful for recommendation purposes.
#step1
transactions = [
    ['Mobile','Charger'],
    ['Mobile','Earphones','Power Bank'],
    ['Laptop','Mouse'],
    ['Mobile','Charger','Earphones'],
    ['Laptop','Mouse','Keyboard']
]

from collections import defaultdict

min_support = 2
item_count = defaultdict(int)

for t in transactions:
    for item in t:
        item_count[item] += 1

frequent_items = {item:count for item,count in item_count.items()
                  if count >= min_support}

print("Frequent Items:", frequent_items)

frequent_patterns = {}
for item,count in frequent_items.items():
    frequent_patterns[(item,)] = count

print("\nFrequent Patterns:")
for pattern,count in frequent_patterns.items():
    print(pattern,":",count)


#(Main Q3.)A retail store wants to study customer buying patterns to understand which products are frequently purchased together. The transaction data collected over one week is given below.
Transaction ID	Items Purchased
T1	Rice, Wheat, Oil
T2	Rice, Oil
T3	Wheat, Sugar
T4	Rice, Wheat, Sugar
T5	Rice, Oil, Sugar
Using the given data, preprocess the transactions into a suitable binary format and apply the Apriori algorithm with appropriate minimum support to identify frequent itemsets. Further, generate association rules using suitable confidence and lift measures and interpret the results.

#Code:
transactions = [
    {'Rice','Wheat','Oil'},
    {'Rice','Oil'},
    {'Wheat','Sugar'},
    {'Rice','Wheat','Sugar'},
    {'Rice','Oil','Sugar'}
]

total_transactions = len(transactions)
min_support = 0.4
min_confidence = 0.6

def calculate_support(itemset):
    count = 0
    for t in transactions:
        if itemset.issubset(t):
            count += 1
    return count / total_transactions

from itertools import combinations
items = set().union(*transactions)

frequent = {}
for i in range(1,4):
    for combo in combinations(items,i):
        sup = calculate_support(set(combo))
        if sup >= min_support:
            frequent[frozenset(combo)] = sup

print("Frequent Itemsets:", frequent)
 

#(MainQ4.)A pharmacy wants to analyze the medicines commonly purchased together to improve shelf arrangement and inventory planning. The following table shows the transaction data.
Transaction ID	Items Purchased
P1	Paracetamol, Cough Syrup
P2	Paracetamol, Vitamin C
P3	Paracetamol, Cough Syrup, Vitamin C
P4	Vitamin C
P5	Paracetamol, Vitamin C
Using the above dataset, preprocess the transactions and apply the FP-Growth algorithm to discover frequent itemsets. Generate strong association rules from the frequent patterns and explain their significance.

#Code:
transactions = [
    ['Paracetamol','Cough Syrup'],
    ['Paracetamol','Vitamin C'],
    ['Paracetamol','Cough Syrup','Vitamin C'],
    ['Vitamin C'],
    ['Paracetamol','Vitamin C']
]

from collections import defaultdict
min_support = 2
item_count = defaultdict(int)

for t in transactions:
    for item in t:
        item_count[item] += 1

frequent_items = {item:count for item,count in item_count.items()
                  if count >= min_support}

print("Frequent Items:", frequent_items)


#(main Q5.)	An online learning platform wants to analyze course enrollment patterns to recommend relevant courses to students. The enrollment data is shown below.
Student ID	Courses Enrolled
S1	Python, Data Science
S2	Python, Machine Learning
S3	Data Science, Machine Learning
S4	Python, Data Science, Machine Learning
S5	Python, Data Science
Using the given data, convert the transactions into binary form and apply the Apriori algorithm to identify frequent course combinations. Generate association rules and explain how these rules can help in course recommendation.

#Code:
transactions = [
    {'Python','Data Science'},
    {'Python','Machine Learning'},
    {'Data Science','Machine Learning'},
    {'Python','Data Science','Machine Learning'},
    {'Python','Data Science'}
]

total_transactions = len(transactions)
min_support = 0.4
min_confidence = 0.6

def calculate_support(itemset):
    count = 0
    for t in transactions:
        if itemset.issubset(t):
            count += 1
    return count / total_transactions

from itertools import combinations
items = set().union(*transactions)

frequent = {}
for i in range(1,4):
    for combo in combinations(items,i):
        sup = calculate_support(set(combo))
        if sup >= min_support:
            frequent[frozenset(combo)] = sup

print("Frequent Course Combinations:", frequent)


#(mainQ6.)A bank wants to identify patterns in the usage of financial services by customers to improve cross-selling strategies. The following data represents customer service usage.
Customer ID	Services Used
C1	ATM, Debit Card
C2	ATM, Debit Card, Internet Banking
C3	Credit Card, Internet Banking
C4	ATM, Credit Card
C5	ATM, Debit Card, Credit Card
Using the above data, apply the FP-Growth algorithm to find frequent service combinations and generate association rules. Interpret the rules using support, confidence, and lift.

#Code:
transactions = [
    ['ATM','Debit Card'],
    ['ATM','Debit Card','Internet Banking'],
    ['Credit Card','Internet Banking'],
    ['ATM','Credit Card'],
    ['ATM','Debit Card','Credit Card']
]

from collections import defaultdict
min_support = 2
item_count = defaultdict(int)

for t in transactions:
    for item in t:
        item_count[item] += 1

frequent_items = {item:count for item,count in item_count.items()
                  if count >= min_support}

print("Frequent Service Combinations:", frequent_items)


#(main Q7.)A hospital wants to analyze patient symptoms to assist doctors in preliminary diagnosis. The following table shows symptom combinations observed in patients.
Patient ID	Symptoms
H1	Fever, Cough
H2	Fever, Headache
H3	Fever, Cough, Headache
H4	Cough
H5	Fever, Cough
Using the given data, preprocess the transactions and apply the Apriori algorithm to identify frequent symptom patterns. Generate association rules and explain their medical relevance.

#Code:
transactions = [
    {'Fever','Cough'},
    {'Fever','Headache'},
    {'Fever','Cough','Headache'},
    {'Cough'},
    {'Fever','Cough'}
]

total_transactions = len(transactions)
min_support = 0.4
min_confidence = 0.6

def calculate_support(itemset):
    count = 0
    for t in transactions:
        if itemset.issubset(t):
            count += 1
    return count / total_transactions

from itertools import combinations
items = set().union(*transactions)

frequent = {}
for i in range(1,4):
    for combo in combinations(items,i):
        sup = calculate_support(set(combo))
        if sup >= min_support:
            frequent[frozenset(combo)] = sup

print("Frequent Symptom Patterns:", frequent)
