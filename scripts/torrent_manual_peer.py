"""Download a torrent by connecting DIRECTLY to seed peers obtained from a
manual HTTP tracker announce (curl), bypassing libtorrent's own announce
which fails against the Cloudflare-fronted academictorrents.com tracker.

Usage:
    python scripts/torrent_manual_peer.py <info_hex> <torrent> <save_dir> <port> <pat1,pat2> [stall_min]
"""
from __future__ import annotations

import binascii
import os
import subprocess
import sys
import time
import urllib.parse

import libtorrent as lt

TRACKERS = [
    "https://academictorrents.com/announce.php",
]


def announce(info_hex: str, my_port: int, tmp: str):
    """Manual HTTP announce via curl → list of (ip, port) peers."""
    ih = binascii.unhexlify(info_hex)
    peer_id = b"-LT2013-" + os.urandom(6).hex().encode()[:12]
    params = {"info_hash": ih, "peer_id": peer_id, "port": my_port,
              "uploaded": 0, "downloaded": 0, "left": 700000000,
              "compact": 1, "event": "started"}
    q = "&".join(f"{k}={urllib.parse.quote(v if isinstance(v, bytes) else str(v).encode())}"
                 for k, v in params.items())
    peers = []
    for base in TRACKERS:
        try:
            subprocess.run(["curl", "-sL", "--max-time", "30", "-o", tmp, f"{base}?{q}"],
                           check=False)
            data = open(tmp, "rb").read()
            dec = lt.bdecode(data)
            p = dec.get(b"peers")
            if isinstance(p, list):                      # dict model
                for d in p:
                    peers.append((d[b"ip"].decode(), int(d[b"port"])))
            elif isinstance(p, (bytes, bytearray)):      # compact model
                for k in range(0, len(p), 6):
                    ip = ".".join(str(b) for b in p[k:k + 4])
                    peers.append((ip, (p[k + 4] << 8) + p[k + 5]))
        except Exception as e:
            print(f"[announce] {base} failed: {e}", flush=True)
    return list({pp for pp in peers})


def main():
    info_hex, torrent, save_dir, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    patterns = [p for p in sys.argv[5].split(",") if p]
    stall_min = float(sys.argv[6]) if len(sys.argv) > 6 else 30.0
    os.makedirs(save_dir, exist_ok=True)
    tmp = os.path.join(save_dir, "_announce.bin")

    ses = lt.session({"listen_interfaces": f"0.0.0.0:{port}"})
    ti = lt.torrent_info(torrent)
    h = ses.add_torrent({"ti": ti, "save_path": save_dir})
    fs = ti.files()
    prio, want = [], 0
    for i in range(fs.num_files()):
        fp = fs.file_path(i).replace("\\", "/")
        keep = (not patterns) or any(pat in fp for pat in patterns)
        prio.append(4 if keep else 0)
        if keep:
            want += fs.file_size(i)
    h.prioritize_files(prio)
    print(f"[manual] selected {want/1e6:.0f} MB; announcing for peers...", flush=True)

    last_done, last_t = 0, time.time()
    while True:
        peers = announce(info_hex, port, tmp)
        for ip, pt in peers:
            try:
                h.connect_peer((ip, pt))
            except Exception:
                pass
        for _ in range(6):                               # 60 s between re-announces
            s = h.status()
            done = int(s.total_wanted_done)
            print(f"[manual] {100*done/max(s.total_wanted,1):5.1f}%  {done/1e6:.0f}/"
                  f"{s.total_wanted/1e6:.0f} MB  down {s.download_rate/1e6:.2f} MB/s  "
                  f"peers {s.num_peers}  (announce peers {len(peers)})", flush=True)
            if s.total_wanted > 0 and done >= s.total_wanted:
                print("[manual] DONE", flush=True)
                return
            if done > last_done:
                last_done, last_t = done, time.time()
            elif time.time() - last_t > stall_min * 60:
                print(f"[manual] STALLED {stall_min}min; giving up.", flush=True)
                sys.exit(2)
            time.sleep(10)


if __name__ == "__main__":
    main()
