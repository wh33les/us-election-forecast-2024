import ast


def check_python_syntax(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        print(f"✅ Syntax check passed for {filename}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax Error in {filename}:")
        print(f"Line {e.lineno}: {e.text}")
        print(f"Error: {e.msg}")
        return False


check_python_syntax("src/pipeline/runner.py")
