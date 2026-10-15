import ast
from pathlib import Path

app_dir = Path(__file__).parent.parent / "backend" / "app"

# We want all set creations, set literals, set comprehensions, frozenset, dict.fromkeys, and string .join calls
sites = []

for py_path in sorted(app_dir.rglob("*.py")):
    rel_path = py_path.relative_to(app_dir.parent)
    with open(py_path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    tree = ast.parse(src, filename=str(py_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("set", "frozenset"):
                line_no = node.lineno
                line_str = lines[line_no - 1].strip()
                sites.append((str(rel_path).replace("\\", "/"), line_no, func.id, line_str))
            elif isinstance(func, ast.Attribute) and func.attr == "fromkeys":
                line_no = node.lineno
                line_str = lines[line_no - 1].strip()
                sites.append((str(rel_path).replace("\\", "/"), line_no, "dict.fromkeys", line_str))
            elif isinstance(func, ast.Attribute) and func.attr == "join":
                # ignore os.path.join
                if isinstance(func.value, ast.Attribute) and func.value.attr == "path":
                    continue
                line_no = node.lineno
                line_str = lines[line_no - 1].strip()
                sites.append((str(rel_path).replace("\\", "/"), line_no, ".join", line_str))
        elif isinstance(node, ast.SetComp):
            line_no = node.lineno
            line_str = lines[line_no - 1].strip()
            sites.append((str(rel_path).replace("\\", "/"), line_no, "set_comp", line_str))
        elif isinstance(node, ast.Set):
            line_no = node.lineno
            line_str = lines[line_no - 1].strip()
            sites.append((str(rel_path).replace("\\", "/"), line_no, "set_lit", line_str))

# Sort sites by file and line number
sites.sort(key=lambda x: (x[0], x[1]))

print(f"Total relevant sites found: {len(sites)}")
for idx, (f, l, kind, code) in enumerate(sites, 1):
    print(f"{idx:2d}. {f}:{l} | {kind} | {code}")
