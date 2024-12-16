from dataclasses import dataclass
from typing import Optional
from text_processor.data_structures.type_definitions import Colour


# immutable (frozen=true) node class for red-black tree

# Once an instance is created, you cannot modify its attributes
# Any attempt to modify attributes will raise FrozenInstanceError

# node class for red - black tree, each node reps a word in the text and the relationship to the tree
@dataclass(frozen=True)
class Node:
    word: str
    colour: Colour
    left: Optional['Node'] = None
    right: Optional['Node'] = None


class RedBlackTree:
    """
    1. Red invariant: No red node can have a red child
    2. Black invariant: Every path from the root to empty leaf must have the same no. of black nodes
    """

    def __init__(self, root: Optional[Node] = None):
        self._root = root  # private variable (convention to start with underscore)

    def is_empty(self) -> bool:
        return self._root is None

    def root_colour(self) -> Optional[Colour]:
        return None if self.is_empty() else self._root.colour

    def _size(self) -> int:
        """Helper method to get approximate tree size"""
        if self.is_empty():
            return 0
        return 1 + self.left()._size() + self.right()._size()

    def insert(self, word: str) -> 'RedBlackTree':
        """
        Inserts a word into the tree - has to maintain tree properties
        Returns a new tree with the word inserted '->'RedBlackTree'

        Immutability - original tree remains unmodified (immutable)
        """
        new_tree = self._ins(word)
        if not new_tree.is_empty():  # why an if statement??
            # Ensure root is black
            return RedBlackTree(Node(
                word=new_tree._root.word,
                colour=Colour.BLACK,
                left=new_tree._root.left,
                right=new_tree._root.right
            ))
        return new_tree

    def _ins(self, word: str) -> 'RedBlackTree':
        """Helper method for insert, that returns a new tree
             - a red-topped tree might be returned"""
        if self.is_empty():
            return RedBlackTree(Node(word=word, colour=Colour.RED))

        # If word already exists, return current tree without modification
        if word == self._root.word:
            return self

        return RedBlackTree._balance(self.root_colour(),
                                     self.left()._ins(word) if word < self._root.word
                                     else self.left() if word > self._root.word
                                     else self,
                                     self._root.word,
                                     self.right() if word < self._root.word
                                     else self.right()._ins(word) if word > self._root.word
                                     else self)

    @staticmethod
    def _balance(colour: Colour, left: 'RedBlackTree', value: str, right: 'RedBlackTree') -> 'RedBlackTree':
        """Balance tree after insertion"""

        # Case 1: Left-Left case (black parent with red left and red left-left)
        if (colour == Colour.BLACK and  # current node is black
                not left.is_empty() and  # has left child
                left.root_colour() == Colour.RED and  # left child is red
                not left.left().is_empty() and  # left child has left child
                left.left().root_colour == Colour.RED):  # left-left grandchild is red
            return RedBlackTree(Node(
                word=left._root.word,  # move left node to root
                colour=Colour.RED,  # colour it red
                left=Node(
                    word=left.left()._root.word,  # left-left grandchild -> left child
                    colour=Colour.BLACK,  # colour it black
                    left=left.left().left()._root,  # keep its children
                    right=left.left().right()._root
                ),
                right=Node(
                    word=value,  # current value becomes right child
                    colour=Colour.BLACK,  # colour it black
                    left=left.right()._root,  # keep remaining subtrees
                    right=right._root
                )
            ))

        # Case 2: Left-Right case (black parent with red left and red left-right)
        elif (colour == Colour.BLACK and  # Current node is black
              not left.is_empty() and  # Has left child
              left.root_colour() == Colour.RED and  # Left child is red
              not left.right().is_empty() and  # Left child has right child
              left.right().root_colour() == Colour.RED):  # Left-right grandchild is red
            return RedBlackTree(Node(
                word=left.right()._root.word,  # Left-right becomes root
                colour=Colour.RED,  # Color it red
                left=Node(  # Left subtree
                    word=left._root.word,  # Original left child
                    colour=Colour.BLACK,  # Color it black
                    left=left.left()._root,  # Keep its left child
                    right=left.right().left()._root  # Get left.right's left child
                ),
                right=Node(  # Right subtree
                    word=value,  # Original root value
                    colour=Colour.BLACK,  # Color it black
                    left=left.right().right()._root,  # Get left.right's right child
                    right=right._root  # Keep original right child
                )
            ))

        # Case 3: Right-Left case (black parent with red right and red right-left)
        elif (colour == Colour.BLACK and  # Current node is black
              not right.is_empty() and  # Has right child
              right.root_colour() == Colour.RED and  # Right child is red
              not right.left().is_empty() and  # Right child has left child
              right.left().root_colour() == Colour.RED):  # Right-left grandchild is red
            return RedBlackTree(Node(
                word=right.left()._root.word,  # Right-left becomes root
                colour=Colour.RED,  # Color it red
                left=Node(  # Left subtree
                    word=value,  # Original root value
                    colour=Colour.BLACK,  # Color it black
                    left=left._root,  # Keep original left child
                    right=right.left().left()._root  # Get right.left's left child
                ),
                right=Node(  # Right subtree
                    word=right._root.word,  # Original right child
                    colour=Colour.BLACK,  # Color it black
                    left=right.left().right()._root,  # Get right.left's right child
                    right=right.right()._root  # Keep right's right child
                )
            ))

        # Case 4: Right-Right case (black parent with red right and red right-right)
        elif (colour == Colour.BLACK and  # Current node is black
              not right.is_empty() and  # Has right child
              right.root_colour() == Colour.RED and  # Right child is red
              not right.right().is_empty() and  # Right child has right child
              right.right().root_colour() == Colour.RED):  # Right-right grandchild is red
            return RedBlackTree(Node(
                word=right._root.word,  # Right child becomes root
                colour=Colour.RED,  # Color it red
                left=Node(  # Left subtree
                    word=value,  # Original root value
                    colour=Colour.BLACK,  # Color it black
                    left=left._root,  # Keep original left child
                    right=right.left()._root  # Get right.left's child
                ),
                right=Node(  # Right subtree
                    word=right.right()._root.word,  # right-right child
                    colour=Colour.BLACK,  # Color it black
                    left=right.right().left()._root,  # Get right.left's right child
                    right=right.right().right()._root  # Keep right's right child
                )
            ))

        # Default case where no restructuring is needed
        return RedBlackTree(Node(
            word=value,
            colour=colour,
            left=None if left.is_empty() else left._root,
            right=None if right.is_empty() else right._root
        ))

    def left(self) -> 'RedBlackTree':
        """Get left subtree"""
        if self.is_empty():
            return RedBlackTree()
        return RedBlackTree(self._root.left)

    def right(self) -> 'RedBlackTree':
        """Get right subtree"""
        if self.is_empty():
            return RedBlackTree()
        return RedBlackTree(self._root.right)

    def member(self, word: str) -> bool:
        """is word member of tree"""
        if self.is_empty():
            return False
        if word < self._root.word:  # searches left if word is smaller
            return self.left().member(word)
        if word > self._root.word:  # searches right if word is larger
            return self.right().member(word)
        return True

    def paint(self, colour: Colour) -> 'RedBlackTree':
        """Creates new tree identical to current one with root painted in specified colour"""
        if self.is_empty():
            return self
        return RedBlackTree(Node(
            word=self._root.word,
            colour=colour,  # only this changes
            left=self._root.left,
            right=self._root.right
        ))

    def doubled_left(self) -> bool:
        """Check red violation on the left"""
        return (not self.is_empty() and
                self.root_colour() == Colour.RED and
                not self.left().is_empty() and
                self.left().root_colour() == Colour.RED)

    def doubled_right(self) -> bool:
        """Check red violation on the right"""
        return (not self.is_empty() and
                self.root_colour() == Colour.RED and
                not self.right().is_empty() and
                self.right().root_colour() == Colour.RED)


# iterator - contains countable no. of values and can be traversed
