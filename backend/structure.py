import os

EXCLUDE_DIRS = {"venv", "__pycache__", ".git", ".idea", ".pytest_cache", ".mypy_cache"}
EXCLUDE_FILES = {".DS_Store"}

def print_dir_tree(startpath, indent=''):
    try:
        for item in sorted(os.listdir(startpath)):
            if item in EXCLUDE_FILES:
                continue

            path = os.path.join(startpath, item)

            # Skip unwanted directories
            if os.path.isdir(path):
                if item in EXCLUDE_DIRS:
                    continue
                print(f"{indent}📂 {item}/")
                print_dir_tree(path, indent + "    ")
            else:
                print(f"{indent}📄 {item}")
    except PermissionError:
        print(f"{indent}🚫 [Permission denied: {startpath}]")

if __name__ == "__main__":
    root = "."  # change to your project path if needed
    print(f"📦 Directory structure of: {os.path.abspath(root)}\n")
    print_dir_tree(root)
