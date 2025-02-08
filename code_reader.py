from llama_index.core.tools import FunctionTool
from pathlib import Path

# We can wrap any python function as a a tool that can be passed to the LLM
def code_reader_func(filename):
    """parse any source code from a input file and return the content of the file

    Args:
        filename (str): source code file

    Returns:
        str: returns the file content of the source code file
    """
    path = Path.cwd() / "data" / filename
    print("path inside code_reader.py", path)
    if not path.exists():
        return {"error": "File does not exist"}
    try:
        with open(path, "r") as f:
            content = f.read()
            return {"file_content": content}
    except Exception as e:
        return {"error": str(e)}


code_reader = FunctionTool.from_defaults(
    fn=code_reader_func,
    name="code_reader",
    description="""this tool can read the contents of code files and return 
    their results. Use this when you need to read the contents of a file""",
)