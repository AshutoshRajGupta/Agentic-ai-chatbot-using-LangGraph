# logger.py

import csv
import os
from datetime import datetime

csv_path = "tool_usage_log.csv"
tool_usage_stats = {}

def log_tool_usage(
    tool_name,
    query,
    response_time,
    response="",
    query_type="",
    paper_title="",
    paper_authors="",
    paper_year="",
    joke_category="",
    success=True
):
    header = [
        "timestamp", "tool_name", "query", "response_time_sec",
        "response_size_bytes", "query_type", "paper_title",
        "paper_authors", "paper_year", "joke_category", "success"
    ]
    row = [
        datetime.now().isoformat(),
        tool_name,
        query,
        round(response_time, 3),
        len(response.encode('utf-8')) if response else 0,
        query_type,
        paper_title,
        paper_authors,
        paper_year,
        joke_category,
        success
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    tool_usage_stats[tool_name] = tool_usage_stats.get(tool_name, 0) + 1
