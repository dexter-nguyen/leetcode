class Solution:
    def invalidTransactions(self, transactions: List[str]) -> List[str]:
        """
        :type transactions: List[str]
        :rtype: List[str]
        """
        
        r = {}
                
        inv = []        

        for i in transactions:
            name,time,amount,city = i.split(",")
            time = int(time)
            amount = int(amount)
            
            if time not in r:
                r[time] = {
                    name: [city]
                }
            else:
                if name not in r[time]:
                    r[time][name]=[city]
                else:
                    r[time][name].append(city)
        print(r)
        
        for i in transactions:
            name,time,amount,city = i.split(",")
            time = int(time)
            amount = int(amount)
            

            if amount > 1000:
                inv.append(i)
                continue
            
            for j in range(time-60, time+61):
                if j not in r:
                    continue
                if name not in r[j]:
                    continue
                if len(r[j][name]) > 1 or (r[j][name][0] != city):
                    inv.append(i)
                    break
                                        
        return inv 



        
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':

        def helper(root,n):
            if not root: 
                return

            root.next = n

            helper(root.left, root.right)  #connection 1
            helper(root.right, n.left if n else None)  #connection 2

        helper(root,None)
        return root
"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

class Solution:
    def flatten(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        :type head: Node
        :rtype: Node
        """
        # traverse the list and look for nodes, where self.child is not None
        # keep the pointer to the previous node and to the original next node
        # merge a child list to the parent list - connect prev and next pointers
        # continue traversing until encountering another node where self.child is not None, 
        # or reaching the end of the main list
        
        current = head
        
        while current :
            # check for child node
            if current.child:
                # merge child list into the parent list
                self.merge(current)
                
            # move to the next node
            current = current.next
        
        return head
            
    
    def merge(self, current):
        child = current.child
        
        # traverse child list until we get the last node
        while child.next:
            child = child.next
        
        # child is now pointing at the last node of the child list
        # we need to connect child.next to current.next, if there is any
        if current.next:
            child.next = current.next
            current.next.prev = child
        
        # now, we have to connect current to the child list
        # child is currently pointing at the last node of the child list, 
        # so we need to use pointer (current.child) to get to the first node of the child list
        current.next = current.child
        current.child.prev = current
        
        # at the end remove self.child pointer
        current.child = None

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        #Sliding winddow from left to the current position 
        # key = ch : value = last seen index
        left = maxLength = 0 
        seen = {}
        
        for i in range(len(s)):
            if s[i] in seen and left <= seen[s[i]]: #If we seen that ch and left ptr is <= that postion
                left = seen[s[i]] + 1
            else:
                maxLength = max(maxLength, i - left + 1) # get the current length from start to current

            seen[s[i]] = i

        return maxLength


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode()
        result = head
        carry = 0
        while l1 or l2 or carry:
            v1 = (l1.val if l1 else 0)
            v2 = (l2.val if l2 else 0)
            value = v1 + v2 + carry
            if value > 9:
                value= value % 10
                carry = 1
            else:
                carry = 0
           
            result.next = ListNode(value)
            result = result.next
                
            if l1:
                l1 = l1.next
            if l2: 
                l2 = l2.next
          
        return head.next     

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        start = []
        end = []
        for s,e in intervals:
            start.append(s)
            end.append(e)
        
        start.sort()
        end.sort()

        # since start always end first
        s,e = 0,0
        res = 0 
        count = 0   #The current rooms we need

        while s < len(start):
            if start[s] < end[e]:  #if the current start < end
                count +=1
                s +=1
            else:                  # we always incre end even when they are even
                e += 1
                count -= 1
            res = max(res, count)
        return res





import random
class RandomizedSet:

    def __init__(self):
        self.data = []
        self.data_map = {}
    
        

    def insert(self, val: int) -> bool:
        if val not in self.data_map:
            position = len(self.data)
            self.data_map[val] = position
            self.data.append(val)
            return True
        return False
        

    def remove(self, val: int) -> bool:
        #We need to exchange the position with the last element in the map and list
        if val in self.data_map:
            last_element = self.data[-1]
            index_of_remove = self.data_map[val]


            self.data_map[last_element] = index_of_remove

            self.data[-1] = val
            self.data[index_of_remove] = last_element

            self.data_map.pop(val)
            self.data.pop()
            return True
        return False
    def getRandom(self) -> int:
        return random.choice(self.data)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        queue = deque()
        if root:
            queue.append(root)
        while queue:
            node = queue.popleft()
            if node.left and node.right:
                queue.append(node.left)
                queue.append(node.right)
                node.left.next = node.right
                if node.next:
                    node.right.next = node.next.left
        return root
def strongPasswordCheckerII( password: str) -> bool:
    
    n = len(password)
    
    if n < 5 or n >12:
        return False

    seen = set()
    pattern = set()
    for i, c in enumerate(password):
        if i > 0 and c == password[i - 1]:
            return False
        if c.isupper():
            seen.add('u')
        elif c.islower():
            seen.add('l')
        elif c.isdigit():
            seen.add('d')             
        else:
            seen.add('s')
    return  len(password) > 7 and len(seen) == 4

/class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        #maintaining curmax and cur increasing
        if len(nums) == 1:
            return 1
        dp = [1] * len(nums)
        
        
        maxx = nums[0]
        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] =  max(dp[i], dp[j] +1)
                
        return  max(dp)
        #dp sections
class Solution:
    def calculateTime(self, keyboard: str, word: str) -> int:
        dic = defaultdict(str)
        
        for i in range(len(keyboard)):
            dic[keyboard[i]] = i
            
        cur = 0
        ret = 0

        for i in range(len(word)):
            ret += abs(cur - dic[word[i]])
            cur = dic[word[i]]
            
        return ret

class Logger:

    def __init__(self):
        self.map = {}
        

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message not in self.map:
            self.map[message] = timestamp + 10
            return True
        
        if self.map[message] > timestamp:
            return False
        elif self.map[message] < timestamp:
            self.map[message] = timestamp + 10
            return True
            
        else:
            self.map[message] = self.map[message]+ 10
            return True
            
class Solution:
    def minDifference(self, nums: List[int]) -> int:
        nums.sort()
        print(nums)
        if len(nums) <=3 :
            return 0
        ret = float("inf")
        for i in range(4):
            ret = min(abs(nums[-4+i]-nums[i]),ret)
            if ret == 0:
                return ret
        
        return ret
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
            
        
