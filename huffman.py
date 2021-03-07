class Base:
    def save(self, encoded, hm, name):
        with open(f"{name}_hm", "wb") as f:
            pickle.dump(hm, f)
        with open(name + '_encoded', 'wb') as file:
            encoded.tofile(file)

    def load(self, name):
        encoded = bitarray()
        with open(name + '_encoded', 'rb') as file:
            encoded.fromfile(file)
        with open(name + '_hm', 'rb') as file:
            hm = pickle.load(file)
        return encoded, hm


class Node:
    def __init__(self, parent=None, left=None, right=None, weight=0, symbol=''):
        super(Node, self).__init__()
        self.parent = parent
        self.left = left
        self.right = right
        self.weight = weight
        self.symbol = symbol


class AdaptiveHuffman(Base):
    def __init__(self):
        super(AdaptiveHuffman, self).__init__()
        self.NULL = Node(symbol="NULL")
        self.root = self.NULL
        self.nodes = []
        self.seen = {}

    def get_code_path(self, s, node, code=''):
        if self.is_leaf(node):
            if node.symbol == s:
                return code
            else:
                return ""
        else:
            result = ''
            if node.left:
                result = self.get_code_path(s, node.left, code + '0')
            if not result and node.right:
                result = self.get_code_path(s, node.right, code + '1')
            return result

    def is_leaf(self, node):
        return not node.left and not node.right

    def get_largest_weight_node(self, weight):
        for n in reversed(self.nodes):
            if n.weight == weight:
                return n

    def swap_nodes(self, n1, n2):
        i1, i2 = self.nodes.index(n1), self.nodes.index(n2)
        self.nodes[i1], self.nodes[i2] = self.nodes[i2], self.nodes[i1]

        self.swap_parents(n1, n2)

    def swap_parents(self, n1, n2):
        tmp_parent = n1.parent
        n1.parent = n2.parent
        n2.parent = tmp_parent
        if n1.parent.left is n2:
            n1.parent.left = n1
        else:
            n1.parent.right = n1
        if n2.parent.left is n1:
            n2.parent.left = n2
        else:
            n2.parent.right = n2

    def insert(self, s):
        node = self.seen[s] if s in self.seen else None

        if node is None:
            node = self.insert_new_node(node, s)

        while node is not None:
            node = self.increase_weight(node)

    def increase_weight(self, node):
        largest = self.get_largest_weight_node(node.weight)
        if (node is not largest and node is not largest.parent and
                largest is not node.parent):
            self.swap_nodes(node, largest)
        node.weight = node.weight + 1
        node = node.parent
        return node

    def insert_new_node(self, node, s):
        node_to_insert = Node(symbol=s, weight=1)
        internal = Node(symbol='', weight=1, parent=self.NULL.parent,
                        left=self.NULL, right=node_to_insert)
        node_to_insert.parent = internal
        self.NULL.parent = internal
        if internal.parent is not None:
            internal.parent.left = internal
        else:
            self.root = internal
        self.nodes.insert(0, internal)
        self.nodes.insert(0, node_to_insert)
        self.seen[s] = node_to_insert
        node = internal.parent
        return node

    def encode(self, text):
        encoding = ''

        for s in text:
            if s in self.seen:
                encoding += self.get_code_path(s, self.root)
            else:
                encoding += self.get_code_path('NULL', self.root)
                encoding += bin(ord(s))[2:].zfill(8)

            self.insert(s)

        return encoding

    def get_symbol_by_ascii(self, bin_str):
        return chr(int(bin_str, 2))

    def decode(self, text):
        result = ''

        symbol = self.get_symbol_by_ascii(text[:8])
        result += symbol
        self.insert(symbol)
        node = self.root

        i = 8
        while i < len(text):
            node = node.left if text[i] == '0' else node.right
            symbol = node.symbol

            if symbol:
                if symbol == 'NULL':
                    symbol = self.get_symbol_by_ascii(text[i + 1:i + 9])
                    i += 8

                result += symbol
                self.insert(symbol)
                node = self.root

            i += 1

        return result
