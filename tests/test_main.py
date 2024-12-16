"""
Test suite for Red-Black Tree implementation and text processing functions.
Run with: python -m unittest test_main.py
"""

import unittest
import tempfile
import os
from functools import reduce
import multiprocessing as mp


from text_processor.data_structures.red_black_tree import RedBlackTree
from text_processor.data_structures.type_definitions import Colour, SystemInfo
from text_processor.core_functionality.text_processing import _clean_text, process_text_chunk
from text_processor.core_functionality.tree_operations import traverse_in_order
from text_processor.io.file_operations import read_file_content
from text_processor.main import process_text_pure


class TestRedBlackTree(unittest.TestCase):
    def setUp(self):
        self.tree = RedBlackTree()

    def test_empty_tree(self):
        """Test empty tree properties"""
        self.assertTrue(self.tree.is_empty())
        self.assertIsNone(self.tree.root_colour())

    def test_single_insertion(self):
        """Test inserting a single word"""
        tree = self.tree.insert("hello")
        self.assertEqual(tree.root_colour(), Colour.BLACK)
        self.assertTrue(tree.member("hello"))
        self.assertFalse(tree.member("world"))

    def test_multiple_insertions(self):
        """Test inserting multiple words"""
        words = ["hello", "world", "test", "python"]
        tree = reduce(lambda t, w: t.insert(w), words, self.tree)

        for word in words:
            self.assertTrue(tree.member(word))

    def test_red_black_properties(self):
        """Test red-black tree properties after insertions"""
        words = ["a", "b", "c", "d", "e", "f", "g"]
        tree = reduce(lambda t, w: t.insert(w), words, self.tree)

        def verify_properties(node):
            if node is None:
                return 0

            # Property 1: Root must be black
            if node == tree._root:
                self.assertEqual(node.colour, Colour.BLACK)

            # Property 2: No red violation
            if node.colour == Colour.RED:
                if node.left:
                    self.assertEqual(node.left.colour, Colour.BLACK)
                if node.right:
                    self.assertEqual(node.right.colour, Colour.BLACK)

            # Property 3: Black height must be same for all paths
            left_height = verify_properties(node.left)
            right_height = verify_properties(node.right)
            self.assertEqual(left_height, right_height)

            return left_height + (1 if node.colour == Colour.BLACK else 0)

        verify_properties(tree._root)

    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        text = "Hello, World! 123 Test-Case"
        cleaned = _clean_text(text)
        self.assertEqual(cleaned, "hello world testcase")

    def test_text_chunk_processing(self):
        """Test text chunk processing"""
        text = "Hello  World\nTest"
        tokens = process_text_chunk(text)
        self.assertEqual(sorted(tokens), ["hello", "test", "world"])


class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.tree = RedBlackTree()

    def test_traverse_in_order_empty_tree(self):
        """Test traverse_in_order with an empty tree."""
        words = list(traverse_in_order(self.tree))
        self.assertEqual(words, [], "Traversing an empty tree should yield no words.")

    def test_traverse_in_order_single_node(self):
        """Test traverse_in_order with a single-node tree."""
        tree = self.tree.insert("hello")
        words = list(traverse_in_order(tree))
        self.assertEqual(words, ["hello"], "Tree with one word should yield only that word.")

    def test_traverse_in_order_balanced_tree(self):
        """Test traverse_in_order with a balanced tree."""
        words_to_insert = ["banana", "apple", "cherry", "date"]
        tree = reduce(lambda t, w: t.insert(w), words_to_insert, self.tree)
        sorted_words = list(traverse_in_order(tree))
        self.assertEqual(sorted_words, sorted(words_to_insert),
                         "In-order traversal should yield sorted words.")

    def test_traverse_in_order_duplicates(self):
        """Test traverse_in_order ignores duplicates."""
        words_to_insert = ["apple", "banana", "apple", "cherry", "banana"]
        tree = reduce(lambda t, w: t.insert(w), words_to_insert, self.tree)
        sorted_words = list(traverse_in_order(tree))
        expected_words = sorted(set(words_to_insert))
        self.assertEqual(sorted_words, expected_words,
                         "In-order traversal should remove duplicates.")


class TestFileOperations(unittest.TestCase):
    def test_read_file_empty(self):
        """Test read_file with an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp_name = tmp.name

        try:
            result = read_file_content(tmp_name)
            self.assertTrue(result.success)
            self.assertEqual(result.content, "")
            self.assertIsNone(result.error)
        finally:
            os.unlink(tmp_name)

    def test_read_file_content(self):
        """Test read_file with content."""
        content = "Line one\nLine two\nLine three"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(content)
            tmp_name = tmp.name

        try:
            result = read_file_content(tmp_name)
            self.assertTrue(result.success)
            self.assertEqual(result.content, content)
            self.assertIsNone(result.error)
        finally:
            os.unlink(tmp_name)


class TestProcessing(unittest.TestCase):
    def test_text_processing(self):
        """Test text processing pipeline"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write("Hello, World!\nTest 123\nHello again")
            tmp_name = tmp.name

        try:
            system_info = SystemInfo(
                cpu_count=mp.cpu_count(),
                chunk_size=1024 * 1024 * 4
            )

            read_result = read_file_content(tmp_name)
            self.assertTrue(read_result.success)

            result = process_text_pure(read_result.content, system_info)
            self.assertTrue(result.success)
            self.assertEqual(result.count, 4)  # unique words: hello, world, test, again
            self.assertEqual(sorted(result.words), ["again", "hello", "test", "world"])
        finally:
            os.unlink(tmp_name)


if __name__ == '__main__':
    unittest.main()

