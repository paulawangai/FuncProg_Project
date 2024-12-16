from typing import List, Iterator, Optional, Tuple
from functools import reduce
from text_processor.data_structures.red_black_tree import RedBlackTree, Node

TreeResult = Tuple[RedBlackTree, int]


def build_tree_from_words(words: Iterator[str]) -> TreeResult:
    """Pure function to build tree from words."""
    word_list = list(words)
    tree = reduce(
        lambda tree, word: tree.insert(word),
        word_list,
        RedBlackTree()
    )
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


def build_subtree(words: List[str]) -> RedBlackTree:
    """Pure function to build a subtree"""
    return reduce(
        lambda tree, word: tree.insert(word),
        sorted(words),
        RedBlackTree()
    )


def merge_trees(tree1: RedBlackTree, tree2: RedBlackTree) -> RedBlackTree:
    return reduce(
        lambda t, word: t.insert(word),
        traverse_in_order(tree2),
        tree1
    )
