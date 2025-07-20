import duckdb

# Install and load HTTPFS so DuckDB can read s3://… URLs
duckdb.sql("INSTALL httpfs; LOAD httpfs;")

def run_duckdb_query(sql: str):
    """
    Execute any DuckDB SQL (e.g. the read_parquet(...) snippet from your instructions).
    Returns a list of dict rows.
    """
    con = duckdb.connect()
    rows = con.execute(sql).fetchall()
    cols = [c[0] for c in con.description]
    return [dict(zip(cols, row)) for row in rows]
