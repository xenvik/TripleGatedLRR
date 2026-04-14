# src/loader.py
import pandas as pd
import numpy as np
import os
import re


def extract_twitter_identities(df):
    """Parses raw Twitter data to create network nodes for LRR graph."""
    df['source_user'] = df['permalink'].str.extract(r'twitter\.com/(\w+)/')
    df['rt_target']   = df['text'].str.extract(r'RT\s@(\w+)')
    df['mentions']    = df['text'].str.findall(r'(?<!RT\s)@(\w+)')
    df['mentions']    = df['mentions'].apply(lambda x: x if isinstance(x, list) else [])
    df['source_user'] = df['source_user'].fillna('unknown_user')
    return df


def calculate_cbs_metrics(tw_df):
    """
    Contradictiveness (Con) Gate Feature Engineering.

    Con = sqrt(pos * |neg|)

    Interpretation: Con is high when a tweet contains strong signals of
    BOTH positive and negative sentiment simultaneously — indicating an
    independent, non-herd perspective. Pure positive or pure negative
    tweets yield low Con scores (herd-following behaviour).
    """
    pos = tw_df.get('pos', pd.Series(0, index=tw_df.index)).astype(float)
    neg = tw_df.get('neg', pd.Series(0, index=tw_df.index)).astype(float)

    tw_df['con'] = np.sqrt(pos * np.abs(neg))
    tw_df['con'] = tw_df['con'].fillna(0.0)   # Missing data → no contribution

    # Derive sentiment if not explicitly provided
    if 'sen' not in tw_df.columns:
        tw_df['sen'] = pos - np.abs(neg)

    return tw_df


def _parse_dates_robustly(series):
    """
    Parse a date column that may use any of several common formats.
    Tries in order:
        1. pandas auto-inference (handles ISO 8601, YYYY-MM-DD, etc.)
        2. dayfirst=True  (DD/MM/YYYY, DD-MM-YYYY)
        3. dayfirst=False (MM/DD/YYYY)
        4. explicit common formats as fallback
    Returns a Series of datetime.date objects (NaT becomes NaN after .dt.date).
    """
    # Try auto first — covers ISO and unambiguous formats
    result = pd.to_datetime(series, errors='coerce')
    if result.notna().mean() > 0.9:
        return result.dt.date

    # Try day-first (European DD/MM/YYYY)
    result = pd.to_datetime(series, errors='coerce', dayfirst=True)
    if result.notna().mean() > 0.9:
        return result.dt.date

    # Try explicit formats one by one
    for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
                '%d %b %Y', '%d %B %Y', '%b %d %Y']:
        try:
            result = pd.to_datetime(series, format=fmt, errors='coerce')
            if result.notna().mean() > 0.9:
                return result.dt.date
        except Exception:
            continue

    # Last resort: coerce whatever pandas can figure out
    return pd.to_datetime(series, errors='coerce').dt.date


def load_and_clean_data(data_path):
    """
    Loads and aligns:
      - twitter.csv        : social network + sentiment data
      - btc.csv / eth.csv  : OHLCV price data
      - btc_onchain.csv / eth_onchain.csv : whale volume (optional)

    Date parsing is format-agnostic: handles DD/MM/YYYY, MM/DD/YYYY,
    YYYY-MM-DD, DD-MM-YYYY, and ISO 8601 automatically.

    Returns:
      tw     (DataFrame) : cleaned tweet-level data
      assets (dict)      : {asset_name: DataFrame with 'time' column as datetime.date}
    """
    tw_path = os.path.join(data_path, "twitter.csv")
    tw = pd.read_csv(tw_path, encoding='utf-8', encoding_errors='replace')
    tw = extract_twitter_identities(tw)
    tw = calculate_cbs_metrics(tw)
    tw['time'] = _parse_dates_robustly(tw['time'])

    assets = {}
    for asset in ['btc', 'eth']:
        p_path = os.path.join(data_path, f"{asset}.csv")
        if not os.path.exists(p_path):
            continue

        df = pd.read_csv(p_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [c.lower().strip() for c in df.columns]
        df['time'] = _parse_dates_robustly(df['time'])

        # Merge on-chain whale volume if available
        on_path = os.path.join(data_path, f"{asset}_onchain.csv")
        if os.path.exists(on_path):
            on_df = pd.read_csv(on_path)
            on_df.columns = [c.lower().strip() for c in on_df.columns]
            on_df['time'] = _parse_dates_robustly(on_df['time'])

            if 'whale_vol_usd' in on_df.columns:
                on_df['whale_vol_usd'] = (
                    on_df['whale_vol_usd'].astype(str)
                    .str.replace(',', '', regex=False)
                    .str.replace('[^0-9.]', '', regex=True)
                    .replace('', np.nan)
                    .astype(float)
                )
                # Log-normalize: whale_vol_usd is in billions, LRR scores are 0-1.
                # Raw values cause VAR coefficient instability and inflated std errors.
                # log1p keeps zero-volume days finite.
                on_df['whale_vol_log'] = np.log1p(on_df['whale_vol_usd'].fillna(0))

            merge_cols = ['time']
            for col in ['whale_vol_usd', 'whale_vol_log']:
                if col in on_df.columns:
                    merge_cols.append(col)
            df = df.merge(on_df[merge_cols], on='time', how='left')

        # Store with 'time' as a plain column (NOT index).
        # The index remains RangeIndex throughout to avoid type-mismatch
        # bugs when filtering by date in downstream modules.
        df = df.reset_index(drop=True)
        assets[asset.upper()] = df

    # -----------------------------------------------------------------------
    # Load additional assets from crypto_research_data.csv (long format)
    # Columns: time, open, high, low, close, volume, symbol
    # One row per asset per day — we split by symbol into separate DataFrames.
    # No on-chain data for these assets: Phase 11 (whale SVAR) will skip
    # automatically because 'whale_vol_log' won't exist in their DataFrames.
    # -----------------------------------------------------------------------
    multi_path = os.path.join(data_path, "crypto_research_data.csv")
    if os.path.exists(multi_path):
        try:
            # Try multiple separator + encoding combos to handle
            # BOM (Excel UTF-8-BOM), tab-sep, and comma-sep files robustly
            mdf = None
            attempts = [
                dict(sep='\t', encoding='utf-8-sig'),   # tab + BOM strip
                dict(sep=',',  encoding='utf-8-sig'),   # comma + BOM strip
                dict(sep='\t', encoding='utf-8'),       # tab plain
                dict(sep=',',  encoding='utf-8'),       # comma plain
                dict(sep='\t', encoding='latin-1'),     # tab latin
                dict(sep=',',  encoding='latin-1'),     # comma latin
            ]
            for kwargs in attempts:
                try:
                    candidate = pd.read_csv(multi_path, **kwargs)
                    # Normalise column names immediately
                    candidate.columns = [c.lower().strip().replace('\ufeff', '')
                                         for c in candidate.columns]
                    if 'time' in candidate.columns and 'close' in candidate.columns:
                        mdf = candidate
                        break
                except Exception:
                    continue

            if mdf is None:
                print("   ! crypto_research_data.csv — could not detect format. "
                      "Check column names include 'time', 'close', 'symbol'.")
            else:
                # Clean symbol column
                mdf['symbol'] = mdf['symbol'].astype(str).str.strip().str.upper()
                mdf['time']   = _parse_dates_robustly(mdf['time'])

                for symbol in sorted(mdf['symbol'].unique()):
                    if symbol in ('BTC', 'ETH'):
                        continue  # already loaded from individual CSVs

                    sym_df = (mdf[mdf['symbol'] == symbol]
                              .drop(columns=['symbol'])
                              .copy()
                              .reset_index(drop=True))

                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in sym_df.columns:
                            sym_df[col] = pd.to_numeric(sym_df[col], errors='coerce')

                    sym_df = (sym_df.dropna(subset=['time', 'close'])
                                    .sort_values('time')
                                    .reset_index(drop=True))

                    if len(sym_df) < 30:
                        print(f"   ! {symbol}: only {len(sym_df)} rows — skipping")
                        continue

                    assets[symbol] = sym_df
                    print(f"   Loaded {symbol}: {len(sym_df)} rows "
                          f"({sym_df['time'].min()} → {sym_df['time'].max()})")

        except Exception as e:
            print(f"   ! crypto_research_data.csv error: {e}")

    return tw, assets