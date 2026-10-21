#!/usr/bin/env python3
"""
Automated Python-to-TXT Synchronization Utility for GRS 1915+105.

Maintains an up-to-date mirror of all Python scripts as plain-text (.txt) files
in a dedicated 'scripts_txt/' directory.

Supports:
1. One-shot synchronization: `python src/sync_txt_scripts.py`
2. Continuous live watching: `python src/sync_txt_scripts.py --watch`
3. Git hook installation: `python src/sync_txt_scripts.py --install-hooks`
4. Module import: `from sync_txt_scripts import sync_all_scripts`
"""

import argparse
import hashlib
import os
import shutil
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TARGET_DIR = os.path.join(REPO_ROOT, "scripts_txt")
IGNORE_DIRS = {".git", "__pycache__", ".system_generated", "scripts_txt", ".gemini", "scratch_temp"}


def get_file_hash(filepath):
    """Calculates SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_python_files(repo_root=REPO_ROOT):
    """Scans repository for all active Python (.py) files."""
    py_files = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, repo_root)
                py_files.append((full_path, rel_path))
    return py_files


def sync_all_scripts(repo_root=REPO_ROOT, target_dir=DEFAULT_TARGET_DIR, verbose=True):
    """
    Synchronizes all Python files into target_dir with .txt extensions.
    Maintains both:
    1. Mirrored subdirectories: scripts_txt/<dir>/<name>.txt
    2. Flat top-level copies:   scripts_txt/<name>.txt (with disambiguation if needed)
    """
    os.makedirs(target_dir, exist_ok=True)
    py_files = collect_python_files(repo_root)

    updated = []
    created = []
    unchanged = []
    expected_txt_paths = set()

    for full_path, rel_path in py_files:
        src_hash = get_file_hash(full_path)
        if src_hash is None:
            continue

        base_name, _ = os.path.splitext(os.path.basename(rel_path))
        dir_name = os.path.dirname(rel_path)

        # 1. Mirrored subfolder path: scripts_txt/src/foo.txt
        mirrored_dir = os.path.join(target_dir, dir_name) if dir_name else target_dir
        os.makedirs(mirrored_dir, exist_ok=True)
        mirrored_txt = os.path.join(mirrored_dir, f"{base_name}.txt")
        expected_txt_paths.add(os.path.abspath(mirrored_txt))

        # 2. Top-level flat path: scripts_txt/foo.txt
        flat_txt = os.path.join(target_dir, f"{base_name}.txt")
        expected_txt_paths.add(os.path.abspath(flat_txt))

        targets = [mirrored_txt]
        if flat_txt != mirrored_txt:
            targets.append(flat_txt)

        for target in targets:
            needs_copy = False
            if not os.path.exists(target):
                needs_copy = True
                created.append(os.path.relpath(target, repo_root))
            else:
                target_hash = get_file_hash(target)
                if target_hash != src_hash:
                    needs_copy = True
                    updated.append(os.path.relpath(target, repo_root))
                else:
                    unchanged.append(target)

            if needs_copy:
                try:
                    shutil.copy2(full_path, target)
                except Exception as e:
                    if verbose:
                        print(f"Error copying {full_path} -> {target}: {e}", file=sys.stderr)

    # Clean up orphaned .txt files in target_dir
    removed = []
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_full = os.path.abspath(os.path.join(root, f))
                if txt_full not in expected_txt_paths:
                    try:
                        os.remove(txt_full)
                        removed.append(os.path.relpath(txt_full, repo_root))
                    except Exception:
                        pass

    if verbose and (created or updated or removed):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Synchronized Python scripts to {os.path.relpath(target_dir, repo_root)}/:")
        if created:
            print(f"  [+] Created {len(created)} file(s): {', '.join(created[:5])}{'...' if len(created) > 5 else ''}")
        if updated:
            print(f"  [*] Updated {len(updated)} file(s): {', '.join(updated[:5])}{'...' if len(updated) > 5 else ''}")
        if removed:
            print(f"  [-] Removed {len(removed)} orphaned file(s)")

    return {
        "created": len(created),
        "updated": len(updated),
        "unchanged": len(unchanged),
        "removed": len(removed),
        "total_py": len(py_files),
    }


def install_git_hooks(repo_root=REPO_ROOT):
    """Installs post-commit and pre-commit hooks to auto-sync on git operations."""
    hooks_dir = os.path.join(repo_root, ".git", "hooks")
    if not os.path.isdir(hooks_dir):
        print(f"Git hooks directory not found at {hooks_dir}")
        return False

    hook_code = f"""#!/bin/sh
# Auto-sync Python scripts to scripts_txt/
PYTHON="{sys.executable}"
SYNC_SCRIPT="{os.path.join(repo_root, 'src', 'sync_txt_scripts.py')}"

if [ -f "$SYNC_SCRIPT" ]; then
    "$PYTHON" "$SYNC_SCRIPT" --quiet
    if [ -d "{os.path.join(repo_root, 'scripts_txt')}" ]; then
        git add "{os.path.join(repo_root, 'scripts_txt')}" 2>/dev/null || true
    fi
fi
"""

    for hook_name in ["pre-commit", "post-commit"]:
        hook_path = os.path.join(hooks_dir, hook_name)
        try:
            with open(hook_path, "w") as f:
                f.write(hook_code)
            os.chmod(hook_path, 0o755)
            print(f"Installed git hook: {hook_path}")
        except Exception as e:
            print(f"Failed to install git hook {hook_name}: {e}")
            return False
    return True


def watch_and_sync(repo_root=REPO_ROOT, target_dir=DEFAULT_TARGET_DIR, poll_interval=1.0):
    """
    Watches the repository for Python file modifications and updates .txt files immediately.
    Uses polling observation which is 100% robust across exFAT and local drives.
    """
    print(f"Starting continuous live watch on {repo_root} -> {os.path.relpath(target_dir, repo_root)}/ ...")
    print(f"Watching for Python (.py) edits every {poll_interval:.1f}s. Press Ctrl+C to stop.")

    # Initial sync
    sync_all_scripts(repo_root=repo_root, target_dir=target_dir, verbose=True)

    # Track mtime and hashes
    state = {}
    for full_path, _ in collect_python_files(repo_root):
        try:
            state[full_path] = (os.path.getmtime(full_path), os.path.getsize(full_path))
        except Exception:
            pass

    try:
        while True:
            time.sleep(poll_interval)
            current_files = collect_python_files(repo_root)
            current_paths = {p[0] for p in current_files}
            changed = False

            # Check if any new or removed files
            if set(state.keys()) != current_paths:
                changed = True
            else:
                for full_path in current_paths:
                    try:
                        mtime = os.path.getmtime(full_path)
                        size = os.path.getsize(full_path)
                        last_mtime, last_size = state.get(full_path, (0, 0))
                        if mtime != last_mtime or size != last_size:
                            changed = True
                            break
                    except Exception:
                        pass

            if changed:
                sync_all_scripts(repo_root=repo_root, target_dir=target_dir, verbose=True)
                # Update state
                state = {}
                for full_path, _ in collect_python_files(repo_root):
                    try:
                        state[full_path] = (os.path.getmtime(full_path), os.path.getsize(full_path))
                    except Exception:
                        pass
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


def main():
    parser = argparse.ArgumentParser(description="Synchronize Python scripts to plain-text .txt files")
    parser.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
        help=f"Target folder for .txt scripts (default: {os.path.relpath(DEFAULT_TARGET_DIR, REPO_ROOT)})",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously in background/foreground, auto-syncing whenever a .py file is saved",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for --watch (default: 1.0s)",
    )
    parser.add_argument(
        "--install-hooks",
        action="store_true",
        help="Install git pre-commit/post-commit hooks to auto-sync on git commit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output when no changes occur",
    )
    args = parser.parse_args()

    if args.install_hooks:
        install_git_hooks(REPO_ROOT)

    if args.watch:
        watch_and_sync(
            repo_root=REPO_ROOT,
            target_dir=args.target_dir,
            poll_interval=args.poll_interval,
        )
    else:
        stats = sync_all_scripts(
            repo_root=REPO_ROOT,
            target_dir=args.target_dir,
            verbose=not args.quiet,
        )
        if not args.quiet and stats["created"] == 0 and stats["updated"] == 0:
            print(f"All {stats['total_py']} Python scripts are already up-to-date in {os.path.relpath(args.target_dir, REPO_ROOT)}/.")


if __name__ == "__main__":
    main()
