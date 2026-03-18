import os

with open("docs/thesis-paper/latest-paper.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

os.makedirs("thesis", exist_ok=True)

with open("thesis/chapter1.md", "w", encoding="utf-8") as f:
    f.writelines(lines[143:456])

with open("thesis/chapter2.md", "w", encoding="utf-8") as f:
    f.writelines(lines[456:723])

with open("thesis/chapter3_old.md", "w", encoding="utf-8") as f:
    f.writelines(lines[723:2905])

print("Chapters successfully split!")
