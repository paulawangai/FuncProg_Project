from dataclasses import dataclass
from typing import Optional, List, Tuple, Iterator
from functools import reduce
from enum import Enum, auto
import re
import sys

# increase recursion limit - due to Error: maximum recursion depth exceeded
sys.setrecursionlimit(10000)


# colour enum for red black tree
class Colour(Enum):
    RED = auto()  # get assigned integer value
    BLACK = auto()


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


# Type aliases

Wordlist = List[str]
TreeOperationResult = Tuple[Node, bool]


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

def read_file(file_path: str) -> Iterator[str]:
    """
     Reads a file and yields lines in a functional manner.

     Args:
         file_path: Path to input file

    Yields:
        Lines from the file
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        yield from file


def clean_text(text: str) -> str:
    """
    Removes punctuation marks and numbers from text in a functional way

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """

    return re.sub(r'[^\w\s]|\d', '', text.lower())


def tokenize(text: str) -> Iterator[str]:
    """
    Splits text into words in a functional manner

    Args:
        text: Input text to tokenize

    Yields:
        Individual words
    """

    yield from filter(bool, clean_text(text).split())


# Type aliases for return types
ProcessResult = Tuple[Iterator[str], int]  # words and count
TreeResult = Tuple[RedBlackTree, int]  # tree and count
WriteResult = Tuple[bool, int]  # success and count


@dataclass(frozen=True)
class ProcessStats:
    """Immutable container for process statistics"""
    word_count: int
    unique_words: int
    success: bool
    message: str


def process_file_to_words(file_path: str) -> ProcessResult:
    """
    Pure function to process file and return unique words.
    Uses function composition and immutable data structures.
    """

    def read_lines(path: str) -> Iterator[str]:
        with open(path, 'r', encoding='utf-8') as file:
            yield from file

    def tokenize_lines(lines: Iterator[str]) -> Iterator[str]:
        return (
            word for line in lines
            for word in tokenize(line)
        )

    def get_unique_words(words: Iterator[str]) -> Tuple[Iterator[str], int]:
        word_set = set(words)  # Using set for uniqueness
        return iter(sorted(word_set)), len(word_set)

    try:
        # Function composition
        lines = read_lines(file_path)
        words = tokenize_lines(lines)
        unique_words, count = get_unique_words(words)
        return unique_words, count
    except Exception:
        return iter([]), 0


def build_tree_from_words(words: Iterator[str]) -> TreeResult:
    """
    Pure function to build tree from words.
    Uses reduce for functional construction.
    """
    # Convert to list once to avoid multiple iterator passes
    word_list = list(words)

    def build_tree(word_list: List[str]) -> RedBlackTree:
        return reduce(
            lambda tree, word: tree.insert(word),
            word_list,
            RedBlackTree()
        )

    tree = build_tree(word_list)
    return tree, len(word_list)


def traverse_in_order(tree: RedBlackTree) -> Iterator[str]:
    """
    Pure function for in-order traversal.
    Uses recursive generator for functional approach.
    """

    def traverse_node(node: Optional[Node]) -> Iterator[str]:
        if node is None:
            return
        yield from traverse_node(node.left)
        yield node.word
        yield from traverse_node(node.right)

    if not tree.is_empty():
        yield from traverse_node(tree._root)


def write_words_to_file(words: Iterator[str], output_path: str) -> WriteResult:
    """
    Write words to file and return status.
    Isolates side effects and returns result.
    """
    try:
        # Convert iterator to list once
        word_list = list(words)

        with open(output_path, 'w', encoding='utf-8', buffering=8192) as f:
            f.write('\n'.join(word_list))

        return True, len(word_list)
    except Exception:
        return False, 0


def main() -> None:
    """
    Main function with integration of all processing steps:
    1. Read and tokenize text file
    2. Insert unique words into red-black tree
    3. Traverse tree for sorted list
    4. Write sorted list to output file
    """
    try:
        input_file = "war_and_peace.txt"
        output_file = "output.txt"

        # 1. Read and tokenize text file
        words, initial_count = process_file_to_words(input_file)
        if initial_count == 0:
            print("No words found in input file")
            return

        # 2. Build tree with unique words
        tree, tree_count = build_tree_from_words(words)
        if tree_count == 0:
            print("Failed to build tree")
            return

        # 3. Get sorted words through traversal
        sorted_words = traverse_in_order(tree)
        if not sorted_words:
            print("Tree traversal yielded no words")
            return

        # 4. Write sorted words to file
        success, final_count = write_words_to_file(sorted_words, output_file)

        # Report results
        if success:
            print(f"Successfully processed {initial_count} words")
            print(f"Wrote {final_count} unique words to {output_file}")
        else:
            print("Failed to write words to file")

    except Exception as e:
        print(f"Error in processing: {str(e)}")


if __name__ == "__main__":
    main()
