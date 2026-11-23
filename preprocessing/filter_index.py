import sqlite3
from datetime import datetime, timezone

class PodcastDatabase():
    """Read access to the PodcastIndex SQLite dump (one table with one row per podcast feed)."""
    def __init__(self, db_path: str, table: str):
        """Open the database at db_path and remember the table to query."""
        self._con = sqlite3.connect(db_path)
        self._cur = self._con.cursor()
        self._table = table

    def get_columns (self):
        """Return the column definitions of the table."""
        return self._cur.execute(f"PRAGMA table_info({self._table})").fetchall()       

    def filter(self, filters: dict, limit: int | None = None) -> list:
        """Return the feed URLs of all podcasts that match all filters (at most `limit` if given, only for quick tests).
        filters maps a column to its allowed values. A podcast matches a column if it matches one of the values,
        compared case-insensitively with SQL LIKE ("en%" matches en, en-us, en-GB, ...).
        newestItemPubdate and oldestItemPubdate take one year: the newest episode must be after that year started, the oldest one before."""
        query = f"SELECT url FROM {self._table} WHERE 1=1"
        params = []

        for column, values in filters.items():
            if values:
                if column == "newestItemPubdate":
                    query += f" AND {column} > (?)" # Filter out podcasts whose newest Item update is older than filter year
                    params.append(self.__date_to_utc_int(year=values[0])) # timezone object needs to be casted into a unix int for db comparison
                elif column == "oldestItemPubdate":
                    query += f" AND {column} < (?)" # Filter out podcasts whose oldest Item update is newer than filter year
                    params.append(self.__date_to_utc_int(year=values[0])) # timezone object needs to be casted into a unix int for db comparison
                else:
                    conditions = " OR ".join([f"lower({column}) LIKE ?"] * len(values))
                    query += f" AND ({conditions})"
                    params.extend(value.lower() for value in values)

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        feed_url_tupels = self._cur.execute(query, params).fetchall()
        return self.__process_to_str_list(feed_url_tupels)

    def __date_to_utc_int(self, year: int, month=1, day=1) -> int:
        """Convert a date to a Unix timestamp (UTC), the format used in the database."""
        dt = datetime(year=year, month=month, day=day, tzinfo=timezone.utc)
        return int(dt.timestamp())

    def __process_to_str_list(self, feed_url_tupels):
        """Turn the one-element row tuples returned by SQLite into a list of strings."""
        return ["".join(feed) for feed in feed_url_tupels]

    def close(self):
        """Close the database connection."""
        self._con.close()
    

    