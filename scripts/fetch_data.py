#!/usr/bin/env python3
"""Download public FOMC minutes and Treasury yield data."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE = "https://www.federalreserve.gov"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def clean_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find(id="article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def fetch(session: requests.Session, url: str) -> str:
    if "fred.stlouisfed.org" in url:
        completed = subprocess.run(
            ["curl", "-L", "--max-time", "45", "-sS", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        completed = subprocess.run(
            ["curl", "-L", "--max-time", "45", "-sS", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout


def minute_links_from_page(html: str, page_url: str, start_year: int, end_year: int) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.search(r"fomcminutes\d{8}\.htm", href):
            continue
        match = re.search(r"fomcminutes(\d{8})\.htm", href)
        if not match:
            continue
        date = match.group(1)
        year = int(date[:4])
        if start_year <= year <= end_year:
            url = urljoin(page_url, href)
            if url not in seen:
                links.append((date, url))
                seen.add(url)
    return links


def download_minutes(start_year: int, end_year: int, raw_dir: Path) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "CS5100 FOMC tone academic project"})
    records: list[dict[str, str]] = []

    current_url = f"{BASE}/monetarypolicy/fomccalendars.htm"
    pages = [(None, current_url)]
    for year in range(start_year, min(end_year, 2020) + 1):
        pages.append((year, f"{BASE}/monetarypolicy/fomchistorical{year}.htm"))

    for _, page_url in pages:
        page_html = fetch(session, page_url)
        for date, minute_url in minute_links_from_page(page_html, page_url, start_year, end_year):
            out_path = raw_dir / "fomc_minutes" / f"fomcminutes{date}.txt"
            if not out_path.exists():
                minute_html = fetch(session, minute_url)
                out_path.write_text(clean_page_text(minute_html), encoding="utf-8")
            records.append(
                {
                    "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                    "year": date[:4],
                    "source_url": minute_url,
                    "local_path": str(out_path),
                }
            )

    metadata = pd.DataFrame(records).drop_duplicates("date").sort_values("date")
    return metadata


def download_fred(series_ids: list[str], raw_dir: Path) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "CS5100 FOMC tone academic project"})
    for series_id in series_ids:
        url = FRED_CSV.format(series_id=series_id)
        (raw_dir / "fred").mkdir(parents=True, exist_ok=True)
        (raw_dir / "fred" / f"{series_id}.csv").write_text(fetch(session, url), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    (raw_dir / "fomc_minutes").mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata = download_minutes(args.start_year, args.end_year, raw_dir)
    metadata.to_csv(processed_dir / "fomc_metadata.csv", index=False)
    download_fred(["DGS10", "DGS2"], raw_dir)
    print(f"Downloaded {len(metadata)} FOMC minutes and FRED series DGS10/DGS2.")


if __name__ == "__main__":
    main()
