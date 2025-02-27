import phase2_utils as utils
print('here')
n = 5
m = 3
L = 3
result_all_assigned = utils.generate_partitions(n, m, L)
result_robust = utils.generate_partitions_robust(n,m,L)

print(f"All partitions of {n} into {m} parts with each part <= {L}:")
for partition in result_all_assigned:
    print(partition)

print(f"All partitions of {n} into {m+1} parts with the initial team representing unassigned robots")
for partition in result_robust:
    print(partition)