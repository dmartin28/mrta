from itertools import combinations

def generate_partitions(n, m, L):
    def partition_helper(n, m, L, current_partition, partitions):
        if m == 0:
            if n == 0:
                partitions.append(current_partition[:])
            return

        for i in range(0, min(n, L) + 1):
            current_partition.append(i)
            partition_helper(n - i, m - 1, L, current_partition, partitions)
            current_partition.pop()

    partitions = []
    partition_helper(n, m, L, [], partitions)
    return partitions