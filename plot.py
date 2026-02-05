#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

for x in sys.stdin.readlines():
    if not x.startswith("sorted sequence"):
        continue

    nums = list(map(int, x.split(":")[1].split()))
    print(nums)
    plt.plot(nums, linestyle='None', marker='.')
    plt.show()

    exit(0)

print("sorted sequence not found")
exit(1)
