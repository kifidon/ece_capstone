"""
Generate test data: runs MoveNet on videos and writes combined CSV.
Calls generate_test_data, which uses the private _process_videos_to_csv.
Run: python scripts/generate_test_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.movenet import generate_test_data

if __name__ == "__main__":
    output_path = os.environ.get("OUTPUT_PATH", "data/poses_data.csv")
    generate_test_data(output_path)
