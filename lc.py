class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        result = nums[0]
        if len(nums) == 1:
            return nums[0]
        for i in range(1,len(nums)):
            if nums[i-1] + nums[i] > nums[i]:
                nums[i] = nums[i-1] + nums[i] 
            if result <= nums[i]:
                result = nums[i]
                
        return result
                
        
            
        
        
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        res = ""
        for i in range(len(s)):
            res = max(self.helper(s,i,i), self.helper(s,i,i+1), res, key=len)

        return res
       
        
    def helper(self,s,l,r):
        while 0<=l and r < len(s) and s[l]==s[r]:
                l-=1
                r+=1
        return s[l+1:r]
            
      
            
        
        

class Solution:
    def rob(self, nums: List[int]) -> int:
        
        #
        def house_robber(nums):
            dp = [0] * len(nums)
            dp[0] = nums[0]
            dp[1] = max(nums[0], nums[1])
            for i in range(2,len(nums)):
                dp[i] = max(dp[i-1], nums[i]+dp[i-2])
            return dp[-1]

        if len(nums) <=2 : 
            return max(nums)
        return max( house_robber(nums[1:]), house_robber(nums[:-1]) )


class Solution:
    def rob(self, nums: List[int]) -> int:
        
        nums = [0,0,0] +  nums
        
        for i in range(3,len(nums)):
            nums[i] += max(nums[i-3],nums[i-2])

        return max(nums[-1],nums[-2])
        

        

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:

        if not cost:
            return 0
        
        dp = [0] * len(cost) 
        
        dp[0] = cost[0]
        
        if len(cost) > 1:
            dp[1] = cost[1]
            
        for i in range(2,len(cost)):
            dp[i] = cost[i] + min(dp[i-1],dp[i-2])
            
        return min(dp[-1],dp[-2])
            
            
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        if not intervals:
            return True
        
        intervals.sort()

        for i in range(1, len(intervals)):
            if intervals[i-1][1] > intervals[i][0]:
                return False
            
        return True
    
        # time = nlogn and O(1) 
 import heapq
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [i * -1 for i in stones]
        heapify(stones)
        while len(stones) > 1:
            s1 = -heappop(stones)
            s2 = -heappop(stones)
            if s1 != s2:
                heappush(stones, -(s1-s2))
                
        return -heappop(stones) if len(stones) >0 else 0
            
 
 # minHeap => first element of len k is the kth- largest
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums[:min(k, len(nums))]
        
        heapify(self.heap)
        for i in range(k,len(nums)):
            heappushpop(self.heap,nums[i])       
            
        print(self.heap)

    def add(self, val: int) -> int:
        heappush(self.heap,val)
        if len(self.heap) > self.k:
            heappop(self.heap)
         
        print(self.heap)
        return self.heap[0]
            
        


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root: 
            return False
        if self.isSameTree(root, subRoot): 
            return True
        return self.isSubtree(root.left,  subRoot) or self.isSubtree(root.right, subRoot)

    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return p is q
        

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p and q:
            return p.val == q.val and self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
        return p is q

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        def depth(root):
            if not root:
                return 0
            
            left = depth(root.left)
            right = depth(root.right)
            
            return max(left,right) + 1
        
        return depth(root)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        def binary(target, nums, low, high):
            if low > high:
                return -1
            
            mid = low + (high - low)//2
            
            if nums[mid] == target:
                return mid
            
            if nums[mid] > target:
                return binary(target,nums,low,mid - 1)
            else:
                return binary(target,nums,mid + 1,high)
            
        return binary(target,nums, 0, len(nums) - 1)
 
class Solution:
    def isValid(self, s: str) -> bool:
        """
        :type s: str
        :rtype: bool
        """
        if len(s) % 2 == 1:
            return False
        
        dic = {'(': ')', '{': '}', '[': ']'}
        stack = []
        
        for c in s:
            if c in dic:
                stack.append(c)
            else:
                if len(stack) == 0 or c != dic[stack.pop()] :
                    return False
        return len(stack) == 0
class Solution:
    def romanToInt(self, s: str) -> int:
        dic = { 'I':1, 'V':5,'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
        
        s = s.replace("IV","IIII").replace("IX","VIIII")
        s = s.replace("XL","XXXX").replace("XC","LXXXX")
        s = s.replace("CD","CCCC").replace("CM","DCCCC")
        
        result = 0
        
        for char in s:
            result += dic[char]
            
        return result
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        dic = Counter(nums)
        
        # for num in nums:
        #     if num not in dic:
        #         dic[num] = 1
        #     else:
        #         dic[num] +=1
                
        for key, value in dic.items():
            if value >1:
                return True
            
        return False
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        def depth(root):
            if not root:
                return 0
            
            left = depth(root.left)
            right = depth(root.right)
            
            
            return max(left,right) + 1
    
        return depth(root)
        
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head.next:
            return head
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
                
        
            
        return slow
        
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def height(root):
            nonlocal diameter
            if not root:
                return 0
            left = height(root.left)
            right = height(root.right)
            diameter = max(diameter, left + right)
            return max(left,right) + 1
            
        diameter = 0
        height(root)
        return diameter
        
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        result = ""
        i,j,carry= len(a) - 1, len(b) - 1,0
        
        while i >= 0 or j>=0:
            temp = carry
            if i >= 0:
                temp += ord(a[i]) - ord('0')    
                
            
            if j >= 0:
                temp += ord(b[j]) - ord('0')
            
            i, j = i -1, j -1
                
            carry = 1 if temp > 1 else 0
            result += str(temp % 2)
            
        if carry > 0:
            result += str(carry)
        return result[::-1]
        
        
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':    
    if p.val <= root.val <= q.val or q.val <= root.val <= p.val:
            return root
        
        if p.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        return self.lowestCommonAncestor(root.right, p, q)
            
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        current = head
        previous = None
        while current:
            store = current.next 
            current.next = previous 
            previous = current
            current = store 
        return previous # represents the head of reversed linkedlist
        
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        visit = set()
        
        rows, cols = len(image), len(image[0])
        
        def notValid(r,c,value) -> bool:
            return  r < 0 or c < 0 or r >= rows or c >= cols or image[r][c] != value or (r,c) in visit
        
        
        def dfs(r,c, value):
            if notValid(r,c,value):
                return
            
            visit.add((r,c))
            image[r][c] = color
            dfs(r+1,c,value)
            dfs(r-1,c,value)
            dfs(r,c+1,value)
            dfs(r,c-1,value)
            
        dfs(sr,sc,image[sr][sc])
        return image
            
            
        
        
            
        
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
            
        
