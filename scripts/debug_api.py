#!/usr/bin/env python3
"""Quick diagnostic to test BitUnix order book and funding rate APIs."""

import asyncio
import json
import os
import sys

import aiohttp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



SPOT_BASE = "https://openapi.bitunix.com"
FUTURES_BASE = "https://fapi.bitunix.com"


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
        # Test 1: Spot depth — no precision
        url = f"{SPOT_BASE}/api/spot/v1/market/depth"
        params = {"symbol": "btcusdt"}
        print(f"\n--- Test 1: GET {url}?symbol=btcusdt (no precision) ---")
        async with s.get(url, params=params) as r:
            body = await r.json()
            print(f"Status: {r.status}")
            print(f"Code: {body.get('code')}, Msg: {body.get('msg')}")
            data = body.get("data")
            if data:
                print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                if isinstance(data, dict):
                    for k in ("bids", "b", "asks", "a"):
                        v = data.get(k)
                        if v:
                            print(f"  {k}: {len(v)} entries, first: {v[0]}")
            else:
                print(f"Data: {data}")

        # Test 2: Spot depth — with precision=2
        params = {"symbol": "btcusdt", "precision": "2"}
        print(f"\n--- Test 2: GET {url}?symbol=btcusdt&precision=2 ---")
        async with s.get(url, params=params) as r:
            body = await r.json()
            print(f"Status: {r.status}")
            print(f"Code: {body.get('code')}, Msg: {body.get('msg')}")
            data = body.get("data")
            if data and isinstance(data, dict):
                for k in ("bids", "b", "asks", "a"):
                    v = data.get(k)
                    if v:
                        print(f"  {k}: {len(v)} entries, first: {v[0]}")
            else:
                print(f"Data: {data}")

        # Test 3: Spot depth — with precision=8
        params = {"symbol": "btcusdt", "precision": "8"}
        print(f"\n--- Test 3: GET {url}?symbol=btcusdt&precision=8 ---")
        async with s.get(url, params=params) as r:
            body = await r.json()
            print(f"Status: {r.status}")
            print(f"Code: {body.get('code')}, Msg: {body.get('msg')}")
            data = body.get("data")
            if data and isinstance(data, dict):
                for k in ("bids", "b", "asks", "a"):
                    v = data.get(k)
                    if v:
                        print(f"  {k}: {len(v)} entries, first: {v[0]}")
            else:
                print(f"Data: {data}")

        # Test 4: Funding rate — single symbol
        url = f"{FUTURES_BASE}/api/v1/futures/market/funding_rate"
        params = {"symbol": "BTCUSDT"}
        print(f"\n--- Test 4: GET {url}?symbol=BTCUSDT ---")
        async with s.get(url, params=params) as r:
            body = await r.json()
            print(f"Status: {r.status}")
            print(f"Code: {body.get('code')}, Msg: {body.get('msg')}")
            data = body.get("data")
            if data and isinstance(data, list):
                print(f"Data: {len(data)} entries, first: {json.dumps(data[0], indent=2)}")
            else:
                print(f"Data: {data}")

        # Test 5: Funding rate — all symbols
        url = f"{FUTURES_BASE}/api/v1/futures/market/funding_rate"
        print(f"\n--- Test 5: GET {url} (no symbol, bulk) ---")
        async with s.get(url) as r:
            body = await r.json()
            print(f"Status: {r.status}")
            print(f"Code: {body.get('code')}, Msg: {body.get('msg')}")
            data = body.get("data")
            if data and isinstance(data, list):
                print(f"Data: {len(data)} entries, first: {json.dumps(data[0], indent=2)}")
            else:
                print(f"Data: {data}")


if __name__ == "__main__":
    asyncio.run(main())
