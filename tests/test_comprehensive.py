import unittest
import tempfile
import os
from functools import reduce
import multiprocessing as mp


from text_processor.data_structures.red_black_tree import RedBlackTree
from text_processor.data_structures.type_definitions import Colour, SystemInfo
from text_processor.core_functionality.text_processing import _clean_text, process_text_chunk
from text_processor.main import process_text_pure
from text_processor.io.file_operations import read_file_content


class TestComprehensiveRedBlackTree(unittest.TestCase):

    def setUp(self):
        self.tree = RedBlackTree()
        self.system_info = SystemInfo(
            cpu_count=mp.cpu_count(),
            chunk_size=1024 * 1024 * 4
        )

    def _verify_black_height(self, tree: RedBlackTree) -> int:
        """Verify all paths have same number of black nodes."""
        if tree.is_empty():
            return 0
        left_height = self._verify_black_height(tree.left())
        right_height = self._verify_black_height(tree.right())
        self.assertEqual(left_height, right_height, "Black height violated")
        return left_height + (1 if tree.root_colour() == Colour.BLACK else 0)

    def _verify_no_red_red(self, tree: RedBlackTree) -> bool:
        """Check red nodes have black children."""
        if tree.is_empty():
            return True
        if tree.root_colour() == Colour.RED:
            if not tree.left().is_empty():
                self.assertEqual(tree.left().root_colour(), Colour.BLACK)
            if not tree.right().is_empty():
                self.assertEqual(tree.right().root_colour(), Colour.BLACK)
        return self._verify_no_red_red(tree.left()) and self._verify_no_red_red(tree.right())

    def _count_nodes(self, tree: RedBlackTree) -> int:
        """Count total nodes in the tree."""
        if tree.is_empty():
            return 0
        return 1 + self._count_nodes(tree.left()) + self._count_nodes(tree.right())

    def test_black_height_property(self):
        words = ["apple", "banana", "cherry", "date", "elderberry"]
        tree = reduce(lambda t, w: t.insert(w), words, self.tree)
        black_height = self._verify_black_height(tree)
        self.assertGreater(black_height, 0, "Tree should have positive black height")

    def test_red_node_children(self):
        words = ["dog", "cat", "mouse", "elephant", "tiger"]
        tree = reduce(lambda t, w: t.insert(w), words, self.tree)
        self.assertTrue(self._verify_no_red_red(tree))

    def test_duplicate_words_detailed(self):
        words = ["test", "test", "test"]
        tree = reduce(lambda t, w: t.insert(w), words, self.tree)
        self.assertEqual(self._count_nodes(tree), 1)
        self.assertTrue(tree.member("test"))

        words = ["apple", "banana", "apple", "cherry", "banana"]
        tree = reduce(lambda t, w: t.insert(w), words, self.tree)
        self.assertEqual(self._count_nodes(tree), 3)
        self.assertTrue(all(tree.member(w) for w in ["apple", "banana", "cherry"]))

    def test_special_characters(self):
        """Test handling of special characters and unicode."""
        words = ["hello!", "world@", "test#", "αβγ", "汉字"]
        expected_clean_words = ["hello", "world", "test", "αβγ", "汉字"]

        # Insert cleaned words instead
        tree = reduce(lambda t, w: t.insert(_clean_text(w)), words, self.tree)

        for word in expected_clean_words:
            self.assertTrue(tree.member(word), f"Word '{word}' should be in the tree")

    def test_memory_usage_simple(self):
        words = [f"word{i}" for i in range(10000)]
        tree = reduce(lambda t, w: t.insert(w), words, self.tree)
        self.assertEqual(self._count_nodes(tree), 10000)
        for i in range(10000):
            self.assertTrue(tree.member(f"word{i}"))

    def test_large_file_handling_detailed(self):
        content = [f"word{i}" for i in range(100)]
        content.extend(["duplicate"] * 50)
        file_content = '\n'.join(content)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(file_content)
            tmp_name = tmp.name

        try:
            # Read file content
            read_result = read_file_content(tmp_name)
            self.assertTrue(read_result.success)

            # Process the content
            result = process_text_pure(read_result.content, self.system_info)
            self.assertTrue(result.success)

            # Check the count of unique words
            expected_words = sorted(set(word for line in content
                                        for word in process_text_chunk(line)))
            self.assertEqual(result.count, len(expected_words),
                             f"Expected {len(expected_words)} unique words, got {result.count}")
            self.assertEqual(sorted(result.words), expected_words,
                             "Processed words don't match")

        finally:
            os.unlink(tmp_name)

    def test_tree_balance(self):
        words = [str(i) for i in range(100)]
        tree = reduce(lambda t, w: t.insert(w), words, self.tree)

        def get_height(t: RedBlackTree) -> int:
            if t.is_empty():
                return 0
            return 1 + max(get_height(t.left()), get_height(t.right()))

        height = get_height(tree)
        max_theoretical_height = 2 * (len(words) + 1).bit_length()
        self.assertLessEqual(height, max_theoretical_height)


if __name__ == '__main__':
    unittest.main()
