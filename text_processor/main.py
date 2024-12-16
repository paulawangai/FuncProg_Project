import sys
import time
import logging
import multiprocessing as mp
from typing import Callable

from text_processor.core_functionality.tree_operations import traverse_in_order
from text_processor.data_structures.type_definitions import SystemInfo, ProcessingResult
from text_processor.io.file_operations import read_file_content, write_file_content
from text_processor.parallel.processing import calculate_chunk_params, process_chunks, build_tree_pure

# Set recursion limit
sys.setrecursionlimit(10000)


def process_text_pure(
    content: str,
    system_info: SystemInfo
) -> ProcessingResult:
    """Pure main function that processes text without side effects"""
    try:
        # Calculate processing parameters
        chunk_size, pool_size = calculate_chunk_params(
            len(content),
            system_info.cpu_count,
            system_info.chunk_size
        )

        # Split content into chunks
        chunks = [
            content[i:i + chunk_size]
            for i in range(0, len(content), chunk_size)
        ]

        # Process chunks
        words, word_count = process_chunks(chunks, pool_size)

        # Build tree
        tree, final_count = build_tree_pure(words, system_info.cpu_count)

        # Get sorted words through traversal
        sorted_words = list(traverse_in_order(tree))

        return ProcessingResult(
            words=sorted_words,
            count=final_count,
            duration=0.0,
            success=True,
            error=None
        )
    except Exception as e:
        return ProcessingResult(
            words=[],
            count=0,
            duration=0.0,
            success=False,
            error=str(e)
        )


def main(input_path: str, output_path: str, log_fn: Callable[[str], None]) -> None:
    """Impure wrapper that handles IO and side effects"""
    system_info = SystemInfo(
        cpu_count=mp.cpu_count(),
        chunk_size=1024 * 1024 * 4
    )

    # Read input file
    start_time = time.time()
    read_result = read_file_content(input_path)
    if not read_result.success:
        log_fn(f"Error reading file: {read_result.error}")
        return

    # Process content (pure)
    result = process_text_pure(read_result.content, system_info)

    # Write output
    if result.success:
        write_result = write_file_content(
            output_path,
            '\n'.join(result.words)
        )
        duration = time.time() - start_time

        if write_result.success:
            log_fn(f"Successfully processed {result.count} words")
            log_fn(f"Total time: {duration:.2f} seconds")
        else:
            log_fn(f"Error writing file: {write_result.error}")
    else:
        log_fn(f"Error processing text: {result.error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main("war_and_peace.txt", "output.txt", logging.info)
