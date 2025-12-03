#!/usr/bin/env python3
"""
Generic crawler for Alzforum mutation tables (PSEN1, APP, etc.),
with robots.txt compliance and crawl-delay handling.

Usage (from project root):

    cd /jet/home/xwang54/GenAIProject_PSEN1

    # Default: crawl PSEN1 (https://www.alzforum.org/mutations/psen-1)
    python data/data_crawler.py

    # Crawl APP (https://www.alzforum.org/mutations/app)
    python data/data_crawler.py --protein app

    # Explicit slug
    python data/data_crawler.py --protein psen-1
"""

import argparse
import csv
import time
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


# =====================================================
# Configuration
# =====================================================

BASE_URL = "https://www.alzforum.org"
ROBOTS_URL = urljoin(BASE_URL, "/robots.txt")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# =====================================================
# Helper Functions
# =====================================================

def check_robots(target_url: str):
    """
    Check robots.txt for fetch permission and crawl delay.
    Returns:
        can_fetch (bool)
        crawl_delay (float)
    """
    rp = robotparser.RobotFileParser()
    rp.set_url(ROBOTS_URL)
    rp.read()

    can_fetch = rp.can_fetch(USER_AGENT, target_url)

    try:
        delay = rp.crawl_delay(USER_AGENT)
    except Exception:
        delay = None

    # Default delay = 10s based on Alzforum robots.txt
    if delay is None:
        delay = 10

    return can_fetch, delay


def fetch_html(url: str) -> str:
    """Fetch HTML from a URL with a custom User-Agent."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    print(f"[DEBUG] HTTP status: {resp.status_code}")
    resp.raise_for_status()
    return resp.text


def parse_table(html: str):
    """
    Parse an Alzforum mutation table into a list of dicts.

    Expected column order:
    1. Mutation
    2. Clinical Phenotype Studied
    3. Pathogenicity
    4. Neuropathology
    5. Biological Effect
    6. Genomic Region
    7. Mutation Type/Codon Change
    8. Research Models
    9. Primary Papers
    """
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table")
    if table is None:
        raise RuntimeError("Mutation table not found in the HTML.")

    rows = []
    trs = table.find_all("tr")

    if not trs:
        raise RuntimeError("Table exists but contains no rows.")

    # Skip header row
    for tr in trs[1:]:
        tds = tr.find_all("td")

        if len(tds) < 9:
            # Skip formatting or incomplete rows
            continue

        cells = [td.get_text(strip=True, separator=" ") for td in tds]

        row = {
            "Mutation": cells[0],
            "Clinical Phenotype Studied": cells[1],
            "Pathogenicity": cells[2],
            "Neuropathology": cells[3],
            "Biological Effect": cells[4],
            "Genomic Region": cells[5],
            "Mutation Type/Codon Change": cells[6],
            "Research Models": cells[7],
            "Primary Papers": cells[8],
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "Parsed 0 rows. The website structure may have changed."
        )

    return rows


def save_csv(rows, path: Path):
    """Save parsed mutation rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Mutation",
        "Clinical Phenotype Studied",
        "Pathogenicity",
        "Neuropathology",
        "Biological Effect",
        "Genomic Region",
        "Mutation Type/Codon Change",
        "Research Models",
        "Primary Papers",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Crawl the Alzforum mutation table for a given protein."
    )
    parser.add_argument(
        "--protein",
        type=str,
        default="psen-1",
        help="Protein slug (e.g. 'psen-1', 'app'). "
             "Will crawl https://www.alzforum.org/mutations/<protein>",
    )

    args = parser.parse_args()
    protein_slug = args.protein.strip()

    mutation_path = f"/mutations/{protein_slug}"
    target_url = urljoin(BASE_URL, mutation_path)

    # Generate file name: psen-1 → psen1_mutations_raw.csv
    gene_symbol = protein_slug.replace("-", "").upper()      # PSEN1, APP
    file_prefix = gene_symbol.lower()                        # psen1, app
    output_path = RAW_DIR / f"{file_prefix}_mutations_raw.csv"

    print(f"[INFO] Protein slug      : {protein_slug}")
    print(f"[INFO] Gene symbol       : {gene_symbol}")
    print(f"[INFO] Target URL        : {target_url}")
    print(f"[INFO] robots.txt URL    : {ROBOTS_URL}")
    print("[INFO] Checking robots.txt...")

    can_fetch, delay = check_robots(target_url)
    print(f"[INFO] Allowed by robots : {can_fetch}")
    print(f"[INFO] Crawl delay       : {delay} seconds")

    if not can_fetch:
        raise SystemExit(
            "robots.txt does NOT allow crawling this URL for the current User-Agent."
        )

    print(f"[INFO] Waiting {delay} seconds before fetching...")
    time.sleep(delay)

    print("[INFO] Fetching HTML...")
    html = fetch_html(target_url)

    print("[INFO] Parsing table...")
    rows = parse_table(html)
    print(f"[INFO] Parsed {len(rows)} rows.")

    print(f"[INFO] Saving CSV to {output_path}")
    save_csv(rows, output_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
