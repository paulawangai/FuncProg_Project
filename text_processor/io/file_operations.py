from text_processor.data_structures.type_definitions import IOResult


def read_file_content(file_path: str) -> IOResult:
    """IO function for file reading"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return IOResult(True, content, None)
    except Exception as e:
        return IOResult(False, None, str(e))


def write_file_content(file_path: str, content: str) -> IOResult:
    """IO function for file writing"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return IOResult(True, None, None)
    except Exception as e:
        return IOResult(False, None, str(e))
