import math
import multiprocessing as mp
from functools import partial
from typing import List, Tuple
from functools import reduce
from text_processor.core_functionality.text_processing import process_text_chunk
from text_processor.core_functionality.tree_operations import build_subtree, merge_trees, build_tree_from_words
from text_processor.core_functionality.tree_operations import TreeResult
from text_processor.data_structures.red_black_tree import RedBlackTree


def calculate_chunk_params(
    file_size: int,
    cpu_count: int,
    base_chunk_size: int = 1024 * 1024 * 4
) -> Tuple[int, int]:
    """Pure function to calculate optimal chunk parameters"""
    optimal_chunk_size = max(base_chunk_size, file_size // (cpu_count * 2))
    pool_size = min(math.ceil(file_size / optimal_chunk_size), cpu_count)
    return optimal_chunk_size, pool_size


def process_chunks(
    chunks: List[str],
    pool_size: int
) -> Tuple[List[str], int]:
    """Pure function to process chunks in parallel"""
    if pool_size <= 0:
        return [], 0

    with mp.Pool(pool_size) as pool:
        results = pool.map(process_text_chunk, chunks)
        all_words = frozenset().union(*[frozenset(result) for result in results])
        return sorted(all_words), len(all_words)


def build_tree_partition(
    words: List[str],
    partition_size: int
) -> List[List[str]]:
    """Pure function to partition words for parallel processing"""
    return [
        words[i:i + partition_size]
        for i in range(0, len(words), partition_size)
    ]


def build_tree_pure(
    words: List[str],
    cpu_count: int
) -> TreeResult:
    """Pure function to build tree with parallelization parameters"""
    if len(words) < 1000:
        return build_tree_from_words(iter(words))

    num_processes = min(cpu_count, max(1, len(words) // 1000))
    sorted_words = sorted(words)
    chunk_size = math.ceil(len(sorted_words) / num_processes)

    word_chunks = build_tree_partition(sorted_words, chunk_size)

    with mp.Pool(num_processes) as pool:
        partial_trees = pool.map(
            partial(build_subtree),
            word_chunks
        )

    final_tree = reduce(merge_trees, partial_trees, RedBlackTree())
    return final_tree, len(words)
