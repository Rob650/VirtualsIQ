"""
VirtualsIQ — Web scraper for off-chain signals
Twitter followers, GitHub stats, website metadata
Uses httpx for lightweight requests, Playwright for JS-heavy pages
"""

import asyncio
import logging
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = None  # Set via env GITHUB_TOKEN for higher rate limits


async def scrape_twitter_profile(twitter_url: str) -> dict:
    """
    Extract Twitter follower count and account age from profile URL.
    Uses nitter or public API fallback.
    """
    if not twitter_url:
        return {}

    # Extract handle
    handle = twitter_url.strip("/").split("/")[-1].lstrip("@")
    if not handle:
        return {}

    # Try public nitter instance for public data
    nitter_urls = [
        f"https://nitter.net/{handle}",
        f"https://nitter.privacydev.net/{handle}",
    ]

    for nitter_url in nitter_urls:
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(nitter_url, headers=HEADERS, timeout=10.0)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")

                # Extract follower count
                followers = 0
                for stat in soup.select(".profile-stat"):
                    label = stat.select_one(".profile-stat-header")
                    value = stat.select_one(".profile-stat-num")
                    if label and value and "Followers" in label.text:
                        raw = value.text.strip().replace(",", "")
                        try:
                            if "K" in raw:
                                followers = int(float(raw.replace("K", "")) * 1000)
                            elif "M" in raw:
                                followers = int(float(raw.replace("M", "")) * 1_000_000)
                            else:
                                followers = int(raw)
                        except ValueError:
                            pass
                        break

                # Extract join date for account age
                account_age_days = 0
                join_el = soup.select_one(".profile-joindate span")
                if join_el and join_el.get("title"):
                    from datetime import datetime
                    try:
                        joined = datetime.strptime(join_el["title"], "%I:%M %p - %d %b %Y")
                        account_age_days = (datetime.utcnow() - joined).days
                    except ValueError:
                        pass

                if followers > 0:
                    return {
                        "twitter_followers": followers,
                        "twitter_account_age": account_age_days,
                    }

        except Exception as e:
            logger.debug(f"Nitter scrape failed for {handle}: {e}")
            continue

    return {"twitter_followers": 0, "twitter_account_age": 0}


async def scrape_github(github_url: str) -> dict:
    """
    Extract GitHub stars, commits, contributors from repo URL.
    Falls back to GitHub API.
    """
    if not github_url:
        return {}

    # Parse owner/repo from URL
    parsed = urlparse(github_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2 or parsed.netloc not in ("github.com", "www.github.com"):
        return {}

    owner, repo = parts[0], parts[1].rstrip(".git")

    try:
        import os
        token = os.environ.get("GITHUB_TOKEN", GITHUB_TOKEN)
        headers = {**HEADERS, "Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"

        async with httpx.AsyncClient() as client:
            # Repo stats
            repo_resp = await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}",
                headers=headers, timeout=10.0
            )
            if repo_resp.status_code != 200:
                return {}
            repo_data = repo_resp.json()

            stars = repo_data.get("stargazers_count", 0)
            contributors_count = 0
            commits_30d = 0

            # Contributors count
            try:
                contrib_resp = await client.get(
                    f"{GITHUB_API}/repos/{owner}/{repo}/contributors?per_page=1&anon=false",
                    headers=headers, timeout=10.0
                )
                if contrib_resp.status_code == 200:
                    link = contrib_resp.headers.get("Link", "")
                    if 'rel="last"' in link:
                        last_page = re.search(r'page=(\d+)>; rel="last"', link)
                        contributors_count = int(last_page.group(1)) if last_page else 1
                    else:
                        contributors_count = len(contrib_resp.json())
            except Exception:
                pass

            # Recent commits (last 30 days)
            try:
                from datetime import datetime, timedelta
                since = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"
                commit_resp = await client.get(
                    f"{GITHUB_API}/repos/{owner}/{repo}/commits?since={since}&per_page=100",
                    headers=headers, timeout=10.0
                )
                if commit_resp.status_code == 200:
                    commits_30d = len(commit_resp.json())
            except Exception:
                pass

            # Last commit date
            last_commit = repo_data.get("pushed_at", "")

            return {
                "github_stars": stars,
                "github_commits_30d": commits_30d,
                "github_contributors": contributors_count,
                "github_last_commit": last_commit,
            }

    except Exception as e:
        logger.debug(f"GitHub scrape failed for {owner}/{repo}: {e}")
        return {}


async def scrape_website(website_url: str) -> dict:
    """
    Basic website metadata extraction — title, description, tech stack hints.
    """
    if not website_url:
        return {}

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(website_url, headers=HEADERS, timeout=10.0)
            if resp.status_code != 200:
                return {}

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title else ""
            desc_el = soup.find("meta", attrs={"name": "description"})
            description = desc_el.get("content", "") if desc_el else ""

            return {
                "website_title": title[:200],
                "website_description": description[:500],
            }

    except Exception as e:
        logger.debug(f"Website scrape failed for {website_url}: {e}")
        return {}


async def enrich_agent_socials(agent: dict) -> dict:
    """
    Run all social scrapes for an agent concurrently.
    Returns merged dict of social metrics.
    """
    tasks = []

    twitter_url = agent.get("linked_twitter", "")
    website_url = agent.get("linked_website", "")

    if twitter_url:
        tasks.append(scrape_twitter_profile(twitter_url))
    else:
        tasks.append(asyncio.coroutine(lambda: {})())

    if website_url and "github.com" in website_url:
        tasks.append(scrape_github(website_url))
    else:
        tasks.append(asyncio.coroutine(lambda: {})())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    merged = {}
    for r in results:
        if isinstance(r, dict):
            merged.update(r)

    return merged
