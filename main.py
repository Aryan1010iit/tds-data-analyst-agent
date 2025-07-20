# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
import os
import re
import pandas as pd
import numpy as np
from httpx import HTTPStatusError

from utils.scraper import fetch_wikipedia_table
from utils.plotting import make_scatterplot

# 1. Load environment
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY in .env")

app = FastAPI(title="TDS Data Analyst Agent")


@app.post("/api/")
async def analyze(task: UploadFile = File(...)):
    # 2. Read & split the incoming payload
    raw = (await task.read()).decode("utf-8")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(400, "Empty request body")

    # 3. Extract the URL from the first line
    url = lines[0]
    if "wikipedia.org" not in url.lower():
        raise HTTPException(400, f"First line is not a Wikipedia URL: {url}")

    # 4. Fetch the table from Wikipedia
    try:
        df = await fetch_wikipedia_table(url)
    except HTTPStatusError as e:
        raise HTTPException(
            400,
            f"Failed to fetch table (HTTP {e.response.status_code}).\n"
            "Make sure the URL is exactly:\n"
            "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        )

    # 5. Clean column names
    df.columns = [re.sub(r"\[.*?\]", "", col) for col in df.columns]

    # 6. Ensure required columns exist
    required = {"Worldwide gross", "Year", "Rank", "Peak"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise HTTPException(400, f"Missing columns: {missing}")

    # 7. Convert gross to numeric (coerce errors to NaN)
    gross_str = df["Worldwide gross"].str.replace(r"[\$,]", "", regex=True)
    df["gross_numeric"] = pd.to_numeric(gross_str, errors="coerce")

    # 8. Convert Year to numeric (coerce errors)
    df["Year_numeric"] = pd.to_numeric(df["Year"], errors="coerce")

    # 9. Count movies ≥ $2 bn before 2020
    count = int(
        df[(df["gross_numeric"] >= 2_000_000_000) & (df["Year_numeric"] < 2020)]
        .shape[0]
    )

    # 10. Find earliest film > $1.5 bn
    #    Determine which title column to use
    title_col = "Film" if "Film" in df.columns else ("Title" if "Title" in df.columns else None)
    if title_col is None:
        raise HTTPException(400, "Missing 'Film' or 'Title' column")

    over_1_5 = df[df["gross_numeric"] > 1_500_000_000]
    earliest = (
        over_1_5.sort_values("Year_numeric")
               .iloc[0][title_col]
        if not over_1_5.empty
        else ""
    )

    # 11. Convert Rank & Peak to numeric
    df["Rank_num"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Peak_num"] = pd.to_numeric(df["Peak"], errors="coerce")

    # 12. Prepare numeric subset for correlation/regression
    df_nr = df.dropna(subset=["Rank_num", "Peak_num"])
    if df_nr.shape[0] < 2:
        corr = 0.0
        slope = intercept = 0.0
    else:
        corr = float(df_nr["Rank_num"].corr(df_nr["Peak_num"]))
        slope, intercept = np.polyfit(df_nr["Rank_num"], df_nr["Peak_num"], 1)
        slope, intercept = float(slope), float(intercept)

    # 13. Generate the scatterplot URI
    uri = make_scatterplot(
        df_nr, "Rank_num", "Peak_num", slope, intercept
    )

    # 14. Return the four-element JSON array
    return [count, earliest, corr, uri]
