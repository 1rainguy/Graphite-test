import os
import sys
import py_compile


def compile_dir(directory: str) -> int:
    failures = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                try:
                    py_compile.compile(path, doraise=True)
                    print(f"OK  {path}")
                except Exception as e:
                    failures += 1
                    print(f"ERR {path}: {e}")
    return failures


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    solvers_dir = os.path.join(base, 'solvers')
    if not os.path.isdir(solvers_dir):
        print("solvers directory not found")
        sys.exit(2)
    failed = compile_dir(solvers_dir)
    print(f"\nSummary: {failed} file(s) failed to compile.")
    sys.exit(1 if failed else 0)


