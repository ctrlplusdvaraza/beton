#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

# Read numbers from stdin
nums = list(map(int, input().split()))

# Changed linestyle to '-' (solid line) and kept markers
plt.plot(nums, linestyle='-', marker='.', markersize=8)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Sorted Sequence')
plt.grid(True, alpha=0.3)
plt.show()
