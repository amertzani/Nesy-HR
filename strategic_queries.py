"""
Strategic Queries Module
=========================

Provides functions for finding CSV files and loading data for strategic
and operational queries.
"""

import os
from typing import Optional

import pandas as pd


def find_csv_file_path() -> Optional[str]:
    """Find the CSV file path for the most relevant HR dataset.

    Tries a small set of known locations (HRDataset_v14.csv, HR_S.csv)
    and returns the first one that exists.
    """
    paths = [
        "/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv",
        "/Users/s20/Desktop/Gnoses/HR Data/HR_S.csv",
        os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HRDataset_v14.csv"),
        os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HR_S.csv"),
    ]

    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_csv_data(file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load CSV data from file path.

    Automatically detects common separators (comma, semicolon, tab).
    Returns a pandas DataFrame or None on failure.
    """
    if file_path is None:
        file_path = find_csv_file_path()

    if file_path is None or not os.path.exists(file_path):
        print(f"⚠️  CSV file not found: {file_path}")
        return None

    try:
        # Detect separator from first line
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
            comma_count = first_line.count(",")
            semicolon_count = first_line.count(";")
            tab_count = first_line.count("\t")

            if semicolon_count > comma_count and semicolon_count > 0:
                sep = ";"
            elif tab_count > comma_count and tab_count > 0:
                sep = "\t"
            else:
                sep = ","

        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding="utf-8",
            on_bad_lines="skip",
            engine="python",
        )

        # Fallback: if we got a single column, try semicolon explicitly
        if len(df.columns) == 1:
            df = pd.read_csv(
                file_path,
                sep=";",
                encoding="utf-8",
                on_bad_lines="skip",
                engine="python",
            )

        return df
    except Exception as e:
        print(f"⚠️  Error loading CSV: {e}")
        return None

"""
Strategic Queries Module
=========================

Provides functions for finding CSV files and loading data for strategic queries.
"""

import os
from typing import Optional


def find_csv_file_path() -> Optional[str]:
    """Find the CSV file path."""
    paths = [
        "/Users/s20/Desktop/Gnoses/HR Data/HRDataset_v14.csv",
        "/Users/s20/Desktop/Gnoses/HR Data/HR_S.csv",
        os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HRDataset_v14.csv"),
        os.path.join(os.path.expanduser("~"), "Desktop", "Gnoses", "HR Data", "HR_S.csv"),
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    return None

