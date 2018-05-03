
# coding: utf-8

# In[1]:


import math
import random
import numpy as np


# In[2]:


def get_input(x):
    temp = []
    for i,v in enumerate(x):
        temp.append([v,i])
    return temp

def chainHash(InputList, Leafs):
    res = {}
    for tup in InputList:
        if tup[0] not in res:
            temp = []
            temp.append(tup[1])
            res["%s" % tup[0]] = temp
        else:
            parent = list(map(lambda s: find_parentid(Leafs[s]), res["%s" % tup[0]]))
            if (find_parentid(Leafs[tup[1]]) not in parent) | (Leafs[tup[1]].parent is None):
                res["%s" % tup[0]].append(tup[1])
    return res
def find_parentid(Node):
    temp = None
    if Node.parent is not None:
        temp = find_parentid(Node.parent)
    else:
        temp = Node.id
    if temp>=0:
        return None
    else:
        return temp

def find_parentNode(Node):
    if Node.parent is not None:
        return find_parentNode(Node.parent)
    else:
        return Node

def euler_distance(point1, point2):
    """
    imput: point1, point2: list
    output: float
    """
    return np.linalg.norm(point1 - point2)


# In[3]:


class Nodes(object):
    def __init__(self,id):
        """
        :param parent
        :param children
        :param id
        """
        self.parent = None
        self.children = []
        self.id = id
    def add_leaf(self, leaf):
        if leaf not in self.children:
            self.children.append(leaf)
    def set_parent(self, node):
        if self.parent is not None:
            pass
        else:
            self.parent = node
    def show_childrenid(self):
        temp = []
        for child in self.children:
            temp.append(child.id)
        return temp
    def display(self,depth):
        print ('-'*depth + "  " +str(self.id))
        for child in self.children:
            child.display(depth+2)
    
class Leafs(Nodes):
    def __init__(self,id, vec):
        """
        :param vec
        :param parent
        :param children
        :param id
        """
        self.vec = vec
        self.parent = None
        self.children = []
        self.id = id
    def add_leaf(self,leaf):
        raise Exception("Leaf nodes can't insert catalog")
    def set_parent(self, node):
        if self.parent is not None:
            raise Exception("It has a parent already")
        else:
            self.parent = node


# In[4]:


class LSH(object):
    def __init__(self, k, l, C, d):
        """
        k: number of sampled bits
        l: number of hash functions
        C: a constant
        d: number of attributes
        """
        assert l > 0
        assert k > 0
        self.k = k
        self.l = l
        self.C = C
        self.d = d
        self.I = []
    def creat_I(self):
        """
        create l distinct hash functions
        """
        while (len(self.I) < self.l):
            temp = sorted(random.sample(range(self.C*self.d),self.k))
            if temp not in self.I:
                self.I.append(temp)
    def change_unary(self, x):
        """
        change the list into unary expression
        x: list[1*d]
        """
        temp = ''
        for num in x:
            tem = int(self.C - num)
            temp += ("1"*(self.C-tem)+ "0"*tem)
        return temp
    def get_h_value(self, v, fun_I):
        temp = np.array(list(v))
        return ''.join(temp[fun_I])
    def hash_table(self,data): 
        """
        each row shows one hash function
        """
        m,n = np.shape(data)
        h_table = []
        v_table = np.array(list(map(lambda s:self.change_unary(s),data)))
        self.creat_I()
        for fun_I in self.I:
            temp = list(map(lambda s: self.get_h_value(s, fun_I), v_table))
            h_table.append(temp)
        return np.array(h_table)
    def get_buckets(self,Leafs,h_table):
        r = list(map(lambda s: chainHash(get_input(s), Leafs),h_table))
        return r
        
    def nn_search(self,q, r):
        """
        find the nn set for ponit q
        """
        p = list(map(lambda s: list(np.where(q in s, s,[q])), sum(list(map(lambda s: list(s.values()),r)),[])))
        P = list(set(sum(p,[])))
        P.remove(q)
        return P


# In[5]:


class Hierarchical(object):
    def __init__(self):
        self.labels = None
        self.Nodes = []
        self.point_num = 0
    def merge_nodes(self, node1, node2):
        newid = -len(self.Nodes)-1
        flag = 0
        if (node1.parent is not None) & (node2.parent is not None):
            if find_parentid(node1) == find_parentid(node2):
                flag = 1
            else:
                NewNode = Nodes(id = newid)
                NewNode.add_leaf(find_parentNode(node1))
                NewNode.add_leaf(find_parentNode(node2))
                find_parentNode(node1).set_parent(NewNode)
                find_parentNode(node2).set_parent(NewNode)
                self.Nodes.append(NewNode)
        if (node1.parent is not None) & (node2.parent is None):
            newid = find_parentid(node1)
            self.Nodes[np.abs(newid)-1].add_leaf(node2)
            node2.set_parent(self.Nodes[np.abs(newid)-1])
        if (node1.parent is None) & (node2.parent is not None):
            newid = find_parentid(node2)
            self.Nodes[np.abs(newid)-1].add_leaf(node1)
            node1.set_parent(self.Nodes[np.abs(newid)-1])
        if (node1.parent is None) & (node2.parent is None):
            NewNode = Nodes(id = newid)
            NewNode.add_leaf(node1)
            NewNode.add_leaf(node2)
            node1.set_parent(NewNode)
            node2.set_parent(NewNode)
            self.Nodes.append(NewNode)
        return flag
            
    
    def fit(self, x, R, A, C,l):
        """
        x:raw data, m*n
        R: minimun distance
        A: the ratio to increase R
        l: number of hash functions
        C: an int constant larger than the max(raw data)
        """
        leafs = [Leafs(vec=v, id=i) for i,v in enumerate(x)]
        distances = {}
        self.point_num, future_num = np.shape(x)  
        self.labels = [ -1 ] * self.point_num
        currentNo = self.point_num
        i = 1
        while (currentNo > 1) & (R < 20):
            k = int(future_num * C * np.sqrt(future_num)/(2 * R))+3
            if k >= (C * future_num - 3):
                k= (C * future_num - 3)
            if k <= 0:
                k = 5
            ls = LSH(k,l ,C ,d = future_num)
            h_table = ls.hash_table(x)
            r = ls.get_buckets(leafs, h_table)
            for p in range(self.point_num):
                P = ls.nn_search(p, r)
                for q in P:
                    d_key = (p, q)
                    if d_key not in distances:
                        distances[d_key] = euler_distance(leafs[p].vec, leafs[q].vec)
                    d = distances[d_key]
                    if i <= 1:
                        if d <= R:
                            flag = self.merge_nodes(leafs[p], leafs[q])
                            if flag == 0:
                                currentNo -= 1
                    else:
                        if (d <= R) & (d > R/A):
                            flag = self.merge_nodes(leafs[p], leafs[q])
                            if flag == 0:
                                currentNo -= 1
            i += 1
            R = R*A
        for i in range(self.point_num):
            self.labels[i] = find_parentid(leafs[i])

    def display_depth(self, depth):
        self.Nodes[-1].display(depth)

