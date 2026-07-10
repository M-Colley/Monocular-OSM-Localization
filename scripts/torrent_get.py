"""Minimal libtorrent file-selective downloader (no aria2c/transmission).

Usage:
    python scripts/torrent_get.py <torrent-or-magnet> <save_dir> <port> [pat1,pat2,...] [stall_min]

Only files whose path contains one of the comma-separated patterns are
downloaded (others set to priority 0). Logs progress; exits when all
SELECTED files are complete, or after `stall_min` minutes with no byte
progress (dead swarm). This is for the fleet-extension datasets (Brno, ZOD)
whose only ungated path is BitTorrent.
"""
from __future__ import annotations

import os
import sys
import time

import libtorrent as lt


def main():
    src = sys.argv[1]
    save_dir = sys.argv[2]
    port = int(sys.argv[3])
    patterns = [p for p in (sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []) if p]
    stall_min = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
    os.makedirs(save_dir, exist_ok=True)

    ses = lt.session({
        "listen_interfaces": f"0.0.0.0:{port},[::]:{port}",
        "alert_mask": lt.alert.category_t.error_notification,
    })
    for r in [("router.bittorrent.com", 6881), ("router.utorrent.com", 6881),
              ("dht.transmissionbt.com", 6881)]:
        ses.add_dht_router(*r)

    atp = (lt.parse_magnet_uri(src) if src.startswith("magnet:")
           else lt.add_torrent_params())
    if not src.startswith("magnet:"):
        atp.ti = lt.torrent_info(src)
    atp.save_path = save_dir
    # extra public trackers to widen the swarm
    for tr in ["udp://tracker.opentrackr.org:1337/announce",
               "udp://tracker.openbittorrent.com:6969/announce",
               "udp://open.demonii.com:1337/announce",
               "https://academictorrents.com/announce.php"]:
        atp.trackers.append(tr)
    h = ses.add_torrent(atp)

    print(f"[torrent] resolving metadata for {src[:60]}...", flush=True)
    while not h.status().has_metadata:
        time.sleep(1)
    info = h.torrent_file()
    fs = info.files()
    n = fs.num_files()
    prio = []
    selected_bytes = 0
    for i in range(n):
        p = fs.file_path(i)
        want = (not patterns) or any(pat in p.replace("\\", "/") for pat in patterns)
        prio.append(4 if want else 0)
        if want:
            selected_bytes += fs.file_size(i)
            print(f"[torrent] SELECT {p} ({fs.file_size(i)/1e6:.1f} MB)", flush=True)
    h.prioritize_files(prio)
    print(f"[torrent] selected {selected_bytes/1e9:.2f} GB of {info.total_size()/1e9:.2f} GB", flush=True)

    last_done = 0
    last_progress_t = time.time()
    while True:
        s = h.status()
        done = int(s.total_wanted_done)
        pct = 100.0 * done / max(s.total_wanted, 1)
        print(f"[torrent] {pct:5.1f}%  {done/1e9:.2f}/{s.total_wanted/1e9:.2f} GB  "
              f"down {s.download_rate/1e6:.2f} MB/s  peers {s.num_peers}  seeds {s.num_seeds}  "
              f"state {s.state}", flush=True)
        if s.total_wanted > 0 and done >= s.total_wanted:
            print("[torrent] DONE (selected files complete)", flush=True)
            break
        if done > last_done:
            last_done = done
            last_progress_t = time.time()
        elif time.time() - last_progress_t > stall_min * 60:
            print(f"[torrent] STALLED — no progress for {stall_min} min "
                  f"(peers {s.num_peers} seeds {s.num_seeds}); giving up.", flush=True)
            sys.exit(2)
        time.sleep(10)


if __name__ == "__main__":
    main()
