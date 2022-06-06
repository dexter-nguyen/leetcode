# Complete the 'isPalindrome' function below.
#
# The function is expected to return a BOOLEAN.
# The function accepts INTEGER_SINGLY_LINKED_LIST A as parameter.
#

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
#
#

def isPalindrome(A):
    # Write your code here
    current = A
    temp = list()
    while current:
        temp.append(current.data)
        current = current.next
    return temp == temp[::-1]  #Space O(n)
  #--------------------------------------
   prev = None
    s = A   # single step pointer
    d = A   # double step pointer
    while d:
        d = d.next     # first update the double moving pointer
        if d:
            d = d.next
        else:  # if reached end of the odd - lenght list, add a adummy node after the single-step-pointer
            dummy = SinglyLinkedListNode(s.data)
            t = s.next
            s.next = dummy
            dummy.next = t
        # reverse the first half of the linked list
        tmp = s.next
        s.next = prev
        prev = s
        s = tmp
        
    curr = prev # check if the reversed part is equal to the second half by values
    while curr and s:
        if curr.data != s.data:
            return False
        curr = curr.next
        s = s.next
    return True
