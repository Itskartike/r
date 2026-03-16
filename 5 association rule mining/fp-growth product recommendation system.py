# An online shopping platform wants to improve its product recommendation system by analyzing customer purchase history. The following dataset represents items purchased by different users.
'''User ID	Items Purchased
U1	Mobile, Charger
U2	Mobile, Earphones, Power Bank
U3	Laptop, Mouse
U4	Mobile, Charger, Earphones
U5	Laptop, Mouse, Keyboard
Using the given data, preprocess the transactions into an appropriate format and apply the FP-Growth algorithm to discover frequent itemsets. Generate strong association rules from the frequent itemsets and explain how these rules can be useful for recommendation purposes.
'''

#Association Rule Mining using FP-Growth Algorithm
transactions = [
    ['Mobile', 'Charger'],
    ['Mobile', 'Earphones', 'Power Bank'],
    ['Laptop', 'Mouse'],
    ['Mobile', 'Charger', 'Earphones'],
    ['Laptop', 'Mouse', 'Keyboard']
]
min_support = 2   # absolute support count
from collections import defaultdict

item_count = defaultdict(int)

for transaction in transactions:
    for item in transaction:
        item_count[item] += 1

print("Item Frequencies:")
for item, count in item_count.items():
    print(item, ":", count)
frequent_items = {item: count for item, count in item_count.items()
                  if count >= min_support}

print("\nFrequent Items:")
print(frequent_items)
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
class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
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
def display_tree(node, indent=0):
    for child in node.children.values():
        print(" " * indent, child.item, ":", child.count)
        display_tree(child, indent + 4)
print("\nFP-Tree:")
display_tree(root)
frequent_patterns = {}

for item, count in frequent_items.items():
    frequent_patterns[(item,)] = count
print("\nFrequent Patterns:")
for pattern, count in frequent_patterns.items():
    print(pattern, ":", count)
