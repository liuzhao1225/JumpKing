import sys
nums = [-1, 2, 1, -4]
nums.sort()
target = 1
min_diff = 999999999999
closest = 0

for i in range(0, len(nums)):
    j = i+1
    k = len(nums)-1
    while j < k:
        sum = nums[i] + nums[j] + nums[k]
        diff = abs(sum - target)
        if diff < min_diff:
            min_diff = diff
            closest = sum
        if sum > target:
            k -= 1
        else:
            j += 1

print(closest)

