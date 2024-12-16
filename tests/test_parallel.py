import unittest
import tempfile
import os
import time
import multiprocessing as mp


from text_processor.data_structures.type_definitions import SystemInfo
from text_processor.main import process_text_pure
from text_processor.io.file_operations import read_file_content


class TestParallelProcessing(unittest.TestCase):
    def setUp(self):
        # Create a larger test file to make parallel processing worthwhile
        base_content = """
        The quick brown fox jumps over the lazy dog.
        Pack my box with five dozen liquor jugs.
        How vexingly quick daft zebras jump!
        The five boxing wizards jump quickly.
        Sphinx of black quartz, judge my vow.
        Two driven jocks help fax my big quiz.
        Five quacking zephyrs jolt my wax bed.
        The jay, pig, fox, zebra, and my wolves quack!
        Waltz, nymph, for quick jigs vex Bud.
        Quick zephyrs blow, vexing daft Jim.
        """
        # Repeat the content to create a larger file (about 1MB)
        self.content = base_content * 1000

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(self.content)
            self.test_file = tmp.name

        # Set up system info for processing
        self.system_info = SystemInfo(
            cpu_count=mp.cpu_count(),
            chunk_size=1024 * 1024 * 4
        )

    def tearDown(self):
        # Clean up the temporary file
        if os.path.exists(self.test_file):
            os.unlink(self.test_file)

    def test_parallel_processing_correctness(self):
        """Test that parallel processing gives same results as sequential"""
        # Read file content
        read_result = read_file_content(self.test_file)
        self.assertTrue(read_result.success)

        # Sequential processing
        chunk_size = len(read_result.content)
        seq_result = process_text_pure(read_result.content, SystemInfo(cpu_count=1, chunk_size=chunk_size))
        self.assertTrue(seq_result.success)
        seq_words = sorted(seq_result.words)

        # Parallel processing
        par_result = process_text_pure(read_result.content, self.system_info)
        self.assertTrue(par_result.success)
        par_words = sorted(par_result.words)

        # Compare results
        self.assertEqual(seq_result.count, par_result.count)
        self.assertEqual(seq_words, par_words)

    def test_parallel_performance(self):
        """Test that parallel processing performs reasonably"""
        read_result = read_file_content(self.test_file)
        self.assertTrue(read_result.success)

        # Time sequential processing
        start = time.time()
        seq_result = process_text_pure(
            read_result.content,
            SystemInfo(cpu_count=1, chunk_size=len(read_result.content))
        )
        seq_time = time.time() - start

        # Time parallel processing
        start = time.time()
        par_result = process_text_pure(read_result.content, self.system_info)
        par_time = time.time() - start

        print(f"\nSequential time: {seq_time:.3f}s")
        print(f"Parallel time: {par_time:.3f}s")

        # For larger files, parallel should not be more than 3 times slower
        self.assertLessEqual(par_time, seq_time * 3,
                             "Parallel processing is significantly slower than expected")

        # Verify both produced correct results
        self.assertTrue(seq_result.success)
        self.assertTrue(par_result.success)
        self.assertEqual(sorted(seq_result.words), sorted(par_result.words))


if __name__ == '__main__':
    unittest.main()
