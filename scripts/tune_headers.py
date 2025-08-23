#!/usr/bin/env python3
"""
Header tuning tool for debugging 403 responses from a website.

Usage examples:
  python3 scripts/tune_headers.py "https://moe.gov.af/some/path"
  python3 scripts/tune_headers.py "https://moe.gov.af/" --save-dir /tmp/header-tests --stop-on-success

What it does:
  - Tries a set of header presets (User-Agent, Referer, Accept-Language, Accept)
  - Performs a GET request for each preset, logs status and a short body preview
  - Saves full response bodies and headers to files in `--save-dir` for manual inspection

Notes:
  - Replace placeholder URLs with the real URL you want to test (do NOT use '...')
  - Respect robots.txt and the site's terms of service. Don't run high-volume tests.
"""

from __future__ import annotations
import argparse
import os
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List
import requests

PRESET_UAS = {
    "chrome_desktop": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "firefox_desktop": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "safari_mac": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "edge_windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "mobile_ios": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "python_requests": "python-requests/2.x",
}

COMMON_ACCEPTS = [
    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "*/*",
]

COMMON_LANGS = [
    "en-US,en;q=0.9",
    "ps-AF,ps;q=0.9,en;q=0.8",
    "en-GB,en;q=0.9",
]


def make_presets() -> List[Dict[str, str]]:
    """Create a list of header dicts to try."""
    presets = []
    
    for ua_key, ua in PRESET_UAS.items():
        for accept in COMMON_ACCEPTS:
            for lang in COMMON_LANGS:
                headers = {
                    "User-Agent": ua,
                    "Accept": accept,
                    "Accept-Language": lang,
                }
                # we'll add Referer later when URL is known
                presets.append(headers)
    # Add a minimal requests-like preset and a very plain browser one
    presets.append({"User-Agent": "python-requests/2.28.1", "Accept": "*/*"})
    return presets


def safe_filename(s: str) -> str:
    # Create a short safe filename by hashing the input and truncating
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:12]


def save_response(save_dir: str, tag: str, url: str, headers: Dict[str, str], resp: requests.Response) -> str:
    os.makedirs(save_dir, exist_ok=True)
    # Use timezone-aware UTC timestamp to avoid DeprecationWarning
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{ts}_{tag}_{safe_filename(url)}"
    body_path = os.path.join(save_dir, base + ".html")
    meta_path = os.path.join(save_dir, base + ".meta.txt")

    # Save body (binary) to preserve content; open in wb
    with open(body_path, "wb") as f:
        f.write(resp.content)

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Preset: {tag}\n")
        f.write(f"Request-Headers: {headers}\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Status: {resp.status_code}\n")
        f.write("\nResponse headers:\n")
        for k, v in resp.headers.items():
            f.write(f"{k}: {v}\n")
    return body_path


def get_referer(url: str):
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}/"
    except Exception:
        pass
    return None


def build_req_headers(preset: Dict[str, str], referer):
    req = dict(preset)  # copy to avoid mutating original
    if referer and "Referer" not in req:
        req["Referer"] = referer
    return req


def try_request(session: requests.Session, url: str, headers: Dict[str, str], timeout: float):
    try:
        resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        return resp, None
    except requests.RequestException as e:
        return None, e


def handle_response(index: int, resp: requests.Response, req_headers: Dict[str, str],
                    args, url: str, tag: str) -> bool:
    """Process and save the response. Return True if caller should stop (stop-on-success)."""
    status = resp.status_code
    print(f"[{index}] Status: {status} (UA: {req_headers.get('User-Agent')})")

    if args.verbose:
        print("Response headers:")
        for k in ("Server", "Set-Cookie", "Content-Type", "Content-Length"):
            if k in resp.headers:
                print(f"  {k}: {resp.headers.get(k)}")
        preview = resp.text[:800].replace('\n', ' ') if resp.text else ''
        print("Preview:", preview[:400])

    saved = save_response(args.save_dir, tag, url, req_headers, resp)
    if args.verbose:
        print(f"Saved body to: {saved}")

    if status != 403:
        print(f"Non-403 response observed for preset {index}: {status} -> saved to {saved}")
        if args.stop_on_success:
            print("Stopping on success as requested.")
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Header tuning tool for debugging 403 responses")
    parser.add_argument("url", help="Full URL to request (do not use '...')")
    parser.add_argument("--save-dir", default="/tmp/header-tune-results", help="Directory to save responses")
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout seconds")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests")
    parser.add_argument("--stop-on-success", action="store_true", help="Stop when a non-403 status is observed")
    parser.add_argument("--max-tries", type=int, default=200, help="Max number of presets to try (safety limit)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    url = args.url
    presets = make_presets()
    referer = get_referer(url)
    session = requests.Session()

    tried = 0
    for i, preset in enumerate(presets):
        if tried >= args.max_tries:
            break
        tried += 1
        tag = f"preset_{i}"
        req_headers = build_req_headers(preset, referer)

        if args.verbose:
            print(f"Trying preset {i} -> User-Agent: {req_headers.get('User-Agent')}")

        resp, err = try_request(session, url, req_headers, args.timeout)
        if resp is None:
            print(f"[{i}] Request failed: {err}")
            time.sleep(args.delay)
            continue

        should_stop = handle_response(i, resp, req_headers, args, url, tag)
        if should_stop:
            return

        time.sleep(args.delay)

    print("Done trying presets. Inspect files in:", args.save_dir)


if __name__ == "__main__":
    main()
