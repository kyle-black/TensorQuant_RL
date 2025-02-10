from itertools import combinations

# List of forex pairs
#forex_pairs = ['EURUSD', 'EURJPY', 'USDJPY', 'GBPUSD']
forex_pairs = ['eurusd','eurjpy', 'eurgbp','audjpy','audusd','gbpjpy','nzdjpy','usdcad','usdchf','usdhkd','usdjpy']
    

# Generate all unique combinations of pairs
pair_combinations = list(combinations(forex_pairs, 2))

# Print the result
for pair in pair_combinations:
    print(pair)
