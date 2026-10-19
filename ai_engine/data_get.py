"""
NIFTY 50 -- 5-MINUTE DATA FETCHER
-----------------------------------
Fetches up to 10 years of 5-min Nifty 50 data from Angel One SmartAPI.
Smart backfill+forward-fill: existing CSV is preserved and extended both ways.

Output: nifty_5m_data.csv
Columns: date (unix seconds), open, high, low, close, volume

Run:
  python data_get.py                    # fetch/update 10 years
  HISTORY_DAYS=3650 python data_get.py  # same, explicit

NOTE: Angel One 5-min candles typically available ~2 years back from today.
For older data (pre-2024), run once per year to accumulate history over time,
or drop a CSV named nifty_manual.csv (same columns) to merge older candles.
"""

import os, time
from datetime import datetime, timedelta

import pandas as pd
import pyotp
from SmartApi import SmartConnect
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------
CLIENT_ID  = os.getenv("ANGEL_CLIENT_ID")
API_KEY    = os.getenv("ANGEL_API_KEY")
PASSWORD   = os.getenv("ANGEL_PASSWORD")
TOTP_KEY   = os.getenv("ANGEL_TOTP_KEY")

TOKEN      = "99926000"    # Nifty 50 index token
EXCHANGE   = "NSE"
INTERVAL   = "FIVE_MINUTE"

OUTPUT_CSV    = "nifty_5m_data.csv"
MANUAL_CSV    = "nifty_manual.csv"   # optional: drop older candles here to merge
CHUNK_DAYS    = 25           # Angel One max ~30 days per request for 5-min
RESUME_FROM   = None         # Set to "YYYY-MM-DD" to resume from that date, or None to start fresh

# 10 years of history. Angel One will return what it has; older gaps stay in
# the CSV if you've accumulated them from previous runs or a manual import.
HISTORY_DAYS  = int(os.getenv("HISTORY_DAYS", "3650"))
# ----------------------------------------------------------------


def authenticate(api: SmartConnect) -> bool:
    print("[AUTH] Logging in to Angel One...")
    try:
        totp = pyotp.TOTP(TOTP_KEY).now()
        data = api.generateSession(CLIENT_ID, PASSWORD, totp)
        if data and data.get("status"):
            print("[AUTH] Login successful.")
            return True
        print(f"[AUTH] Failed: {data}")
        return False
    except Exception as e:
        print(f"[AUTH] Error: {e}")
        return False


def fetch_chunk(api: SmartConnect, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    params = {
        "exchange":    EXCHANGE,
        "symboltoken": TOKEN,
        "interval":    INTERVAL,
        "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate":      to_dt.strftime("%Y-%m-%d %H:%M"),
    }
    resp = api.getCandleData(params)
    if not resp or not resp.get("status") or not resp.get("data"):
        return pd.DataFrame()

    df = pd.DataFrame(resp["data"],
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    if not all([CLIENT_ID, API_KEY, PASSWORD, TOTP_KEY]):
        print("[ERR] Angel One credentials missing in .env")
        print("      Need: ANGEL_CLIENT_ID, ANGEL_API_KEY, ANGEL_PASSWORD, ANGEL_TOTP_KEY")
        return

    api = SmartConnect(api_key=API_KEY)
    if not authenticate(api):
        return

    end_date      = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    desired_start = end_date - timedelta(days=HISTORY_DAYS)

    # Plan ranges to fetch: backfill (older) + forward-fill (newer)
    ranges = []
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        if not existing.empty:
            csv_first = pd.to_datetime(existing["date"].min(), unit="s")
            csv_last  = pd.to_datetime(existing["date"].max(), unit="s")
            print(f"[RESUME] Found {len(existing):,} existing candles "
                  f"({csv_first.strftime('%Y-%m-%d')} -> {csv_last.strftime('%Y-%m-%d')})")

            # Backfill: data OLDER than what we have
            if desired_start < csv_first:
                bf_end = csv_first - timedelta(minutes=5)
                ranges.append(("BACKFILL", desired_start, bf_end))
                print(f"[BACKFILL] Will fetch older data: "
                      f"{desired_start.strftime('%Y-%m-%d')} -> {bf_end.strftime('%Y-%m-%d')}")

            # Forward fill: data NEWER than what we have
            ff_start = csv_last + timedelta(minutes=5)
            if ff_start < end_date:
                ranges.append(("FORWARD", ff_start, end_date))
                print(f"[FORWARD]  Will fetch newer data: "
                      f"{ff_start.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")

            if not ranges:
                print(f"[SKIP] CSV already covers desired range "
                      f"(set HISTORY_DAYS > current range to backfill).")
                return
        else:
            ranges.append(("FRESH", desired_start, end_date))
    elif RESUME_FROM:
        ranges.append(("FRESH", datetime.strptime(RESUME_FROM, "%Y-%m-%d"), end_date))
    else:
        ranges.append(("FRESH", desired_start, end_date))

    total_chunks = sum(((e - s).days // CHUNK_DAYS + 1) for _, s, e in ranges)

    print(f"\n[INFO] Fetching FIVE_MINUTE data ({HISTORY_DAYS} days target history)")
    print(f"       Total chunks across {len(ranges)} range(s): {total_chunks}\n")

    all_chunks = []
    chunk_num  = 0

    for label, range_start, range_end in ranges:
        print(f"\n[{label}] {range_start.strftime('%Y-%m-%d')} "
              f"-> {range_end.strftime('%Y-%m-%d')}")
        current = range_start
        while current < range_end:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), range_end)
            chunk_num += 1

            for attempt in range(4):
                try:
                    df = fetch_chunk(api, current, chunk_end)
                    if not df.empty:
                        all_chunks.append(df)
                        print(f"  [{chunk_num:>3}/{total_chunks}] "
                              f"{current.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')} "
                              f"| {len(df):>5} candles  [OK]")
                    else:
                        print(f"  [{chunk_num:>3}/{total_chunks}] "
                              f"{current.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')} "
                              f"| empty")
                    break   # success -- move to next chunk
                except Exception as e:
                    if "rate" in str(e).lower() or "access" in str(e).lower():
                        wait = 8 * (attempt + 1)
                        print(f"  [{chunk_num:>3}/{total_chunks}] "
                              f"{current.strftime('%Y-%m-%d')} rate-limited -- "
                              f"retrying in {wait}s (attempt {attempt+1}/4)...")
                        time.sleep(wait)
                    else:
                        print(f"  [{chunk_num:>3}/{total_chunks}] "
                              f"{current.strftime('%Y-%m-%d')} ERROR: {e}")
                        break

            current = chunk_end + timedelta(minutes=5)
            time.sleep(1.2)   # Angel One rate limit

    frames = []

    if all_chunks:
        new_df = pd.concat(all_chunks, ignore_index=True)
        new_df.drop_duplicates(subset=["timestamp"], inplace=True)
        new_df.sort_values("timestamp", inplace=True)
        new_df["date"] = new_df["timestamp"].apply(lambda x: int(x.timestamp()))
        new_df = new_df[["date", "open", "high", "low", "close", "volume"]]
        frames.append(new_df)
    else:
        print("\n[WARN] No new data fetched from Angel One "
              "(may already be up-to-date or API limit reached).")

    # Merge existing CSV
    if os.path.exists(OUTPUT_CSV):
        frames.append(pd.read_csv(OUTPUT_CSV))

    # Merge optional manual CSV (older candles provided externally)
    if os.path.exists(MANUAL_CSV):
        manual = pd.read_csv(MANUAL_CSV)
        frames.append(manual)
        print(f"[MANUAL] Merged {len(manual):,} candles from {MANUAL_CSV}")

    if not frames:
        print("[ERR] No data available at all. Run failed.")
        return

    full_df = pd.concat(frames, ignore_index=True)
    full_df.drop_duplicates(subset=["date"], inplace=True)
    full_df.sort_values("date", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    first_dt = pd.to_datetime(full_df["date"].iloc[0],  unit="s").strftime("%Y-%m-%d")
    last_dt  = pd.to_datetime(full_df["date"].iloc[-1], unit="s").strftime("%Y-%m-%d")

    full_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Candles : {len(full_df):,}")
    print(f"  Range   : {first_dt} -> {last_dt}")
    print(f"  Saved   : {OUTPUT_CSV}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
