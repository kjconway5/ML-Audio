#!/usr/bin/env python3
"""
Utility to extract the list of source files from filelist.json
Prints space-separated list of file paths to stdout
"""

import json
import sys
from pathlib import Path

def get_filelist(json_path="filelist.json"):
    """Read and return list of files from filelist.json"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            files = data.get("files", [])
            # Print space-separated list
            print(" ".join(files), end="")
            return files
    except FileNotFoundError:
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Support optional path argument
    json_path = sys.argv[1] if len(sys.argv) > 1 else "filelist.json"
    get_filelist(json_path)