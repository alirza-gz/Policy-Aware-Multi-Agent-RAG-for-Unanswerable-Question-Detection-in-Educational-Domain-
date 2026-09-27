Association rule mining finds implications of the form antecedent implies consequent in transactional data. A classic educational example is market-basket analysis of items purchased together.

Support of an itemset is the fraction of transactions that contain all items in the set. Minimum support filters out rare itemsets before rules are formed.

Confidence of a rule X implies Y is the support of X union Y divided by the support of X. Confidence estimates how often Y appears among transactions that contain X.

Lift compares a rule's confidence to the baseline frequency of the consequent. Lift greater than one suggests a positive association beyond chance co-occurrence under independence.

The Apriori principle states that all subsets of a frequent itemset must themselves be frequent. Apriori uses this monotonicity to prune the candidate search space.

Association rules are not causal statements. A strong rule can reflect confounding, store layout, or marketing bundles rather than a mechanism that forces the consequent.
