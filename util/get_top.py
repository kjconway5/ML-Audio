#!/usr/bin/env python3
"""
Utility to extract the top module name from filelist.json
Prints the top module name to stdout
"""

import json
import sys
from pathlib import Path

def get_top(json_path="filelist.json"):
    """Read and return top module name from filelist.json"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            top = data.get("top", "")
            if not top:
                print("Error: 'top' field is missing or empty in filelist.json", file=sys.stderr)
                sys.exit(1)
            print(top, end="")
            return top
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
    get_top(json_path)