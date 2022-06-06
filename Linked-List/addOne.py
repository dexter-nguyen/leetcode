# Complete the 'addOne' function below.
#
# The function is expected to return an INTEGER_SINGLY_LINKED_LIST.
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

def addOne(A):
    # Write your code here
    A = reverse(A)
    cur = A
    cur.data += 1
    while cur.data == 10:
        cur.data = 0
        if cur.next:
            cur = cur.next
            cur.data += 1
        else:
            cur.next = SinglyLinkedListNode(1)
            break
    return reverse(A)

def reverse(head):
    previous = None
    while head:
        next_head = head.next
        head.next = previous
        previous = head
        head = next_head
    return previous
