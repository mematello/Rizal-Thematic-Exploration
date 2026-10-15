import ast
import os
from pathlib import Path

app_dir = Path(__file__).parent.parent / "backend" / "app"

results = []

for py_path in sorted(app_dir.rglob("*.py")):
    rel_path = py_path.relative_to(app_dir.parent)
    with open(py_path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()

    try:
        tree = ast.parse(src, filename=str(py_path))
    except Exception as e:
        print(f"Error parsing {py_path}: {e}")
        continue

    for node in ast.walk(tree):
        # 1. set(...) call
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("set", "frozenset"):
                line_no = node.lineno
                line_str = lines[line_no - 1].strip()
                results.append((str(rel_path), line_no, f"{func.id}(...)", line_str))
            elif isinstance(func, ast.Attribute) and func.attr == "fromkeys":
                line_no = node.lineno
                line_str = lines[line_no - 1].strip()
                results.append((str(rel_path), line_no, "dict.fromkeys(...)", line_str))
            elif isinstance(func, ast.Attribute) and func.attr == "join":
                line_no = node.lineno
                line_str = lines[line_no - 1].strip()
                results.append((str(rel_path), line_no, ".join(...)", line_str))
        # 2. Set comprehension
        elif isinstance(node, ast.SetComp):
            line_no = node.lineno
            line_str = lines[line_no - 1].strip()
            results.append((str(rel_path), line_no, "{x for x in ...}", line_str))
        # 3. Set literal
        elif isinstance(node, ast.Set):
            line_no = node.lineno
            line_str = lines[line_no - 1].strip()
            results.append((str(rel_path), line_no, "{...}", line_str))

print(f"Total sites found: {len(results)}")
for r in results:
    print(f"{r[0]}:{r[1]} | {r[2]} | {r[3]}")
