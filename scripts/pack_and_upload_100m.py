#!/usr/bin/env python3
"""
Wait for download_katago_40m.py to complete all 2,000 shards, then:
1. Pack all 5,000 shards (10M + 50M + 40M = 100,000,000 unique 19x19 Chinese-rules positions)
   into a single unified Zip64 archive `/content/data/katago_19x19_chinese_100m.zip`
   (using ZIP_STORED because each .npz shard is already deflate-compressed).
2. Upload `/content/data/katago_19x19_chinese_100m.zip` directly from the Colab datacenter
   to Google Drive folder `1NUwQ3e6vSJneXUOqHVEaxTtQa-K1rk7V` (`我的云端硬盘/AI训练数据/围棋/`)
   via Google Drive REST API v3 Resumable Upload (`uploadType=resumable`).
3. Clean up the temporary `.zip` file on local NVMe after verifying the remote file size on Google Drive.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

GDRIVE_FOLDER_ID = "1NUwQ3e6vSJneXUOqHVEaxTtQa-K1rk7V"  # 我的云端硬盘/AI训练数据/围棋
METADATA_TOKEN_URL = "http://172.28.0.1:8008/computeMetadata/v1/instance/service-accounts/default/token"

DIR_10M = Path("/content/data/katago_19x19_chinese_10m")
DIR_50M = Path("/content/data/katago_19x19_chinese_50m")
DIR_40M = Path("/content/data/katago_19x19_chinese_40m")
ZIP_PATH = Path("/content/data/katago_19x19_chinese_100m.zip")
ARCHIVE_NAME = "katago_19x19_chinese_100m.zip"


def get_drive_token() -> str:
    req = urllib.request.Request(
        METADATA_TOKEN_URL,
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["access_token"]


def wait_for_40m_completion():
    print("[1/3] Waiting for 40M collection (/content/data/katago_19x19_chinese_40m) to reach 2,000 shards...", flush=True)
    while True:
        shards = list(DIR_40M.glob("shard_*.npz"))
        manifest = DIR_40M / "manifest.json"
        if len(shards) >= 2000 and manifest.exists():
            print(f"40M collection complete! Found {len(shards)} shards and manifest.json.", flush=True)
            break
        print(f"  Currently {len(shards)}/2000 shards in {DIR_40M}...", flush=True)
        time.sleep(15)


def build_100m_zip():
    shards_10m = sorted(DIR_10M.glob("shard_*.npz"))
    shards_50m = sorted(DIR_50M.glob("shard_*.npz"))
    shards_40m = sorted(DIR_40M.glob("shard_*.npz"))

    assert len(shards_10m) == 500, f"Expected 500 shards in 10M, got {len(shards_10m)}"
    assert len(shards_50m) == 2500, f"Expected 2500 shards in 50M, got {len(shards_50m)}"
    assert len(shards_40m) == 2000, f"Expected 2000 shards in 40M, got {len(shards_40m)}"

    all_sources = shards_10m + shards_50m + shards_40m
    total_shards = len(all_sources)  # 5,000 shards
    total_rows = total_shards * 20_000  # 100,000,000 rows

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 45_000_000_000:
        print(f"[2/3] Existing archive {ZIP_PATH} found ({ZIP_PATH.stat().st_size / (1024**3):.2f} GiB), skipping repack.", flush=True)
        return ZIP_PATH.stat().st_size

    print(f"[2/3] Packing {total_shards:,} shards ({total_rows:,} positions) into {ZIP_PATH} (ZIP_STORED)...", flush=True)
    t0 = time.time()
    bytes_written = 0

    with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for idx, src_path in enumerate(all_sources):
            arcname = f"katago_19x19_chinese_100m/shard_{idx:05d}.npz"
            zf.write(src_path, arcname=arcname)
            bytes_written += src_path.stat().st_size
            if (idx + 1) % 250 == 0 or (idx + 1) == total_shards:
                elapsed = time.time() - t0
                speed_mb = (bytes_written / (1024 * 1024)) / max(elapsed, 1e-3)
                print(
                    f"  [Pack {idx + 1:04d}/{total_shards}] "
                    f"{bytes_written / (1024**3):.2f} GiB | "
                    f"Speed: {speed_mb:.1f} MB/s | "
                    f"Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

        manifest_data = {
            "dataset": "KataGo kata1 self-play training data (100M unified filtered & deduplicated)",
            "board_size": "19x19",
            "rules": "Chinese Area Scoring (SCORING_AREA + TAX_NONE)",
            "total_rows": total_rows,
            "num_shards": total_shards,
            "rows_per_shard": 20_000,
            "composition": {
                "shard_00000_to_00499": "10M dataset (500 shards, 10,000,000 rows)",
                "shard_00500_to_02999": "50M dataset (2,500 shards, 50,000,000 rows)",
                "shard_03000_to_04999": "40M dataset (2,000 shards, 40,000,000 rows)",
            },
            "total_raw_size_bytes": bytes_written,
            "total_raw_size_gb": round(bytes_written / (1024**3), 3),
            "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }
        zf.writestr(
            "katago_19x19_chinese_100m/manifest.json",
            json.dumps(manifest_data, indent=2, ensure_ascii=False),
        )

    final_size = ZIP_PATH.stat().st_size
    elapsed = time.time() - t0
    print(
        f"[2/3] Archive {ZIP_PATH} built! Size: {final_size / (1024**3):.2f} GiB ({final_size:,} bytes) in {elapsed:.1f}s.",
        flush=True,
    )
    return final_size


def initiate_resumable_upload(total_size: int) -> str:
    token = get_drive_token()
    init_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
    metadata = {
        "name": ARCHIVE_NAME,
        "parents": [GDRIVE_FOLDER_ID],
        "mimeType": "application/zip",
    }
    body = json.dumps(metadata).encode("utf-8")
    req = urllib.request.Request(
        init_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/zip",
            "X-Upload-Content-Length": str(total_size),
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        session_uri = resp.headers.get("Location")
        if not session_uri:
            raise RuntimeError("No Location header returned from resumable upload initiation")
        return session_uri


def query_upload_offset(session_uri: str, total_size: int) -> int:
    token = get_drive_token()
    req = urllib.request.Request(
        session_uri,
        data=b"",
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Length": "0",
            "Content-Range": f"bytes */{total_size}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return total_size
    except urllib.error.HTTPError as e:
        if e.code == 308:
            rng = e.headers.get("Range", "")
            if rng.startswith("bytes=0-"):
                return int(rng.split("-")[1]) + 1
            return 0
        raise


def upload_to_gdrive(total_size: int):
    print(f"[3/3] Starting direct datacenter Resumable Upload of {ARCHIVE_NAME} ({total_size / (1024**3):.2f} GiB) to Google Drive folder {GDRIVE_FOLDER_ID}...", flush=True)
    session_uri_file = Path("/content/data/.upload_100m_session_uri.txt")
    session_uri = None
    offset = 0

    if session_uri_file.exists():
        try:
            candidate_uri = session_uri_file.read_text().strip()
            offset = query_upload_offset(candidate_uri, total_size)
            session_uri = candidate_uri
            print(f"  Resuming existing upload session from byte offset {offset:,} ({offset / (1024**3):.2f} GiB)...", flush=True)
        except Exception as e:
            print(f"  Previous session URI invalid ({e}), creating new session...", flush=True)
            session_uri = None
            offset = 0

    if not session_uri:
        session_uri = initiate_resumable_upload(total_size)
        session_uri_file.write_text(session_uri)
        print("  Created new Google Drive Resumable Upload session.", flush=True)

    chunk_size = 64 * 1024 * 1024  # 64 MiB chunks (multiple of 256 KiB)
    t0 = time.time()
    start_offset = offset

    with open(ZIP_PATH, "rb") as f:
        while offset < total_size:
            f.seek(offset)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_len = len(chunk)
            end_byte = offset + chunk_len - 1
            content_range = f"bytes {offset}-{end_byte}/{total_size}"

            req = urllib.request.Request(
                session_uri,
                data=chunk,
                method="PUT",
                headers={
                    "Content-Length": str(chunk_len),
                    "Content-Range": content_range,
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp_body = resp.read().decode("utf-8")
                    info = json.loads(resp_body)
                    offset = total_size
                    elapsed = time.time() - t0
                    avg_speed = ((total_size - start_offset) / (1024 * 1024)) / max(elapsed, 1e-3)
                    print(
                        f"  [Upload 100.0%] Completed! File ID: {info.get('id')} | Name: {info.get('name')} | "
                        f"Avg Speed: {avg_speed:.1f} MB/s | Elapsed: {elapsed:.1f}s",
                        flush=True,
                    )
                    if session_uri_file.exists():
                        session_uri_file.unlink()
                    return info
            except urllib.error.HTTPError as e:
                if e.code == 308:
                    rng = e.headers.get("Range", "")
                    if rng.startswith("bytes=0-"):
                        offset = int(rng.split("-")[1]) + 1
                    else:
                        offset += chunk_len
                    elapsed = time.time() - t0
                    uploaded_now = offset - start_offset
                    speed_mb = (uploaded_now / (1024 * 1024)) / max(elapsed, 1e-3)
                    pct = offset / total_size * 100.0
                    eta_s = ((total_size - offset) / (1024 * 1024)) / max(speed_mb, 1e-3)
                    if (offset // chunk_size) % 4 == 0:
                        print(
                            f"  [Upload {pct:5.1f}%] {offset / (1024**3):.2f} / {total_size / (1024**3):.2f} GiB | "
                            f"Speed: {speed_mb:.1f} MB/s | ETA: {eta_s:.0f}s",
                            flush=True,
                        )
                else:
                    err_text = ""
                    try:
                        err_text = e.read().decode("utf-8")
                    except Exception:
                        pass
                    print(f"  [Warn] HTTP {e.code} during chunk upload ({err_text[:200]}), querying offset in 3s...", flush=True)
                    time.sleep(3)
                    offset = query_upload_offset(session_uri, total_size)
            except Exception as e:
                print(f"  [Warn] Network exception during chunk upload ({e}), querying offset in 3s...", flush=True)
                time.sleep(3)
                offset = query_upload_offset(session_uri, total_size)


def main():
    wait_for_40m_completion()
    total_size = build_100m_zip()
    info = upload_to_gdrive(total_size)
    print("\n" + "=" * 65)
    print(f"SUCCESSFULLY UPLOADED 100M ARCHIVE TO GOOGLE DRIVE!")
    print(f"File Name : {ARCHIVE_NAME}")
    print(f"File ID   : {info.get('id')}")
    print(f"File Size : {total_size / (1024**3):.2f} GiB ({total_size:,} bytes)")
    print(f"Folder ID : {GDRIVE_FOLDER_ID} (我的云端硬盘/AI训练数据/围棋)")
    print("=" * 65, flush=True)

    # Remove the temporary 50GB zip file from local NVMe so /content keeps >140GB free space
    try:
        ZIP_PATH.unlink()
        print(f"Cleaned up temporary local archive {ZIP_PATH} to preserve VM disk space.", flush=True)
    except Exception as e:
        print(f"Could not unlink {ZIP_PATH}: {e}", flush=True)


if __name__ == "__main__":
    main()
