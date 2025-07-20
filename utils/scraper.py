import httpx, pandas as pd
from bs4 import BeautifulSoup

async def fetch_wikipedia_table(url: str) -> pd.DataFrame:
    """
    Fetches the first wikitable from the given URL into a pandas DataFrame.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "wikitable"})
    if table is None:
        raise ValueError("No wikitable found")
    return pd.read_html(str(table))[0]
