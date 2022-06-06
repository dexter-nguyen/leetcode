class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.dict = dict()
        self.capacity = capacity
        self.head = Node()
        self.tail = Node
        
        self.link(self.head, self.tail)
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.dict:
            node = self.dict[key]
            self.remove(node)
            self.put(node.key, node.value)
            return node.value
        
        return -1
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.dict:
            self.remove(self.dict[key])  
            
        node = Node(key, value)
        
        self.link(self.tail.prev, node)
        self.link(node, self.tail)
        self.dict[key] = node

        if len(self.dict) > self.capacity:
            self.remove(self.head.next)
            
    def remove(self, node):
        del self.dict[node.key]
        self.link(node.prev, node.next)
        
    def link(self, a, b):
        a.next = b
        b.prev = a   
        
        
class Node():
    def __init__(self, key = None, value = None):
        self.key = key
        self.value = value
        self.next = None
        self.prev = None
