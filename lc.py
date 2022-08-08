class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        def binary(low,high):
            if low > high:
                return -1
            else:
                mid = low + (high - low)//2

                if nums[mid] == target:
                    return mid

                if target > nums[mid]:
                    return binary(mid+1,high)
                else:
                    return binary(low, mid - 1)

        return binary(0,len(nums)-1)
      
 
        

#Two sum:
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            if target - nums[i] in dic:
                return [i,dic[target - nums[i]]]
            dic[nums[i]] = i
  
#Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit, min_price = 0, float('inf')
        for price in prices:
            min_price = min(min_price, price)
            profit = price - min_price
            max_profit = max(max_profit, profit)
        return max_profit
 #Valid parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        """
        :type s: str
        :rtype: bool
        """
        if len(s) % 2 == 1:
            return False
        d = {'(':')', '{':'}','[':']'}
        stack = []
        for i in s:
            if i in d:  # 1
                stack.append(i)
            elif len(stack) == 0 or d[stack.pop()] != i:  # 2
                return False
        return len(stack) == 0 # 3
#Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    
        if not list1 or not list2:
            return list1 or list2
        if list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2 
#Invert Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
            stack = [root]
            while stack:
                node = stack.pop()
                if node:
                    node.left, node.right = node.right, node.left
                    stack.extend([node.right, node.left])
            return root
        """
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

    # The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        if isBadVersion(n) and n ==1:
            return n
        low,high,mid=1, n,(1+n)//2

        while low < high:
            if isBadVersion(mid):
                high = mid
            else:
                low = mid  + 1
            mid = low + (high - low)//2
            
        return low

 class MyQueue:

    def __init__(self):
        self.stack = []
        

    def push(self, x: int) -> None:
        self.stack.append(x)
        

    def pop(self) -> int:
        return self.stack.pop(0)
        

    def peek(self) -> int:
        return self.stack[0]
        

    def empty(self) -> bool:
        return len(self.stack) == 0
            
        
