import phase2_utils as utils
print('here')
n = 5
m = 3
L = 3
result = utils.generate_partitions(n, m, L)
print(f"All partitions of {n} into {m} parts with each part <= {L}:")
for partition in result:
    print(partition)