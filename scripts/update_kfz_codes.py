"""Refresh monocular_osm/data/kfz_codes.json from Wikidata.

    python scripts/update_kfz_codes.py            # query WDQS, write the snapshot
    python scripts/update_kfz_codes.py --dry-run  # query + report, write nothing

German licence plates (Kfz-Kennzeichen) start with a 1-3 letter
registration-district code. :mod:`monocular_osm.kfz_codes` maps that code to
a geocodable district-seat name for the plate-district anchor. The table used
to be hand-curated (311 codes, self-described as not exhaustive); this script
derives it from Wikidata property P395 ("licence plate code") instead and
writes a committed, dated snapshot. The hand-curated entries are still applied
on top as overrides by ``kfz_codes.py`` (they encode verified
disambiguations such as FDB = Friedberg *in Bayern*).

Queries (public Wikidata Query Service, all texts recorded in the snapshot):

* CLASSES - the subclass closure of the entity kinds we accept: urban
  district (kreisfreie Stadt, Q22865); district (Q106658, with every state's
  subclass) plus the former-district, Prussian-district and GDR-Kreis classes;
  municipality in Germany (Q262166). (GDR *Bezirke* are excluded: their P395
  values are pre-1990 GDR-system letters, e.g. L = Bezirk Erfurt.)
* STATEMENTS - every non-deprecated P395 statement on an entity in Germany /
  West Germany / the GDR, with its end-time qualifier (P582), the entity's
  classes and dissolution date (P576).
* INFO (batched) - German label, coordinates (P625), population (P1082) and
  seat (P36: rank, end time, label, coordinates) of the entities a code may
  resolve to. Labels are read with rdfs:label (the label service occasionally
  returns a bare QID inside OPTIONAL blocks).
* STATES (batched) - the federal state of each chosen seat (P131*).

Codes must match ``[A-ZÄÖÜ]{1,3}``; malformed values are skipped and listed.
Statements of retired plate SYSTEMS are dropped (``obsolete_system``): those
ending on/before 1956-07-01 (occupation-zone codes such as "17a", "IV B") and
those ending in 1990 (the GDR's letter pairs such as Y/YO on Kreis Löbau or I
on Berlin; Y is now the Bundeswehr). Codes retired by the 1970s West German
district reforms are KEPT (flagged historic): nearly all were reissued after
the 2012 "Kennzeichenliberalisierung" even where Wikidata still records only
the old end date (EHI = Ehingen, in use again since 2013, next to our Ulm
clips).
A code is ``historic`` when no current (undissolved) entity carries it with
an open-ended statement. One seat is chosen per code, in this order:

1. a current urban district carrying the code (the city itself; with several,
   the one whose name matches the code -- L is on Leipzig *and*, as a GDR-era
   Bezirk letter, on Erfurt);
2. a district carrying it (current first, then dissolved) whose seat "matches"
   the code (first letter equal and every code letter in the seat name):
   FDB -> Friedberg (the dissolved Landkreis Friedberg) rather than Aichach
   (today's Aichach-Friedberg), LÖB -> Löbau rather than Görlitz;
3. otherwise the seat of a current district carrying it (SK -> Merseburg),
   then the seat of a dissolved one;
4. municipalities only (codes no district carries): a municipality whose
   name starts with the code (EHI -> Ehingen, CAS -> Castrop-Rauxel), else
   the most populous carrier.
Current seats are preferred over P36 statements with an end time.

The geocode string is ``"<seat label>, <state label>"`` (German labels), so
homonymous towns (two Friedbergs, two Eisenbergs, two Rotenburgs ...) resolve
in the right state.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "monocular_osm" / "data" / "kfz_codes.json"
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = ("monocular-osm-localization/kfz-code-snapshot "
              "(https://github.com/M-Colley/Monocular-OSM-Localization; "
              "scripts/update_kfz_codes.py) python-requests/" + requests.__version__)
CODE_RE = re.compile(r"^[A-ZÄÖÜ]{1,3}$")
_DISTRICT_PREFIX = re.compile(r"^(Landkreis|Kreis|Stadtkreis|Bezirksamt|Oberamt)\s+")


def obsolete_system(end: str | None) -> bool:
    """True for a P395 statement of a retired plate system (see docstring)."""
    if not end or not end[:4].isdigit():
        return False
    return end[:10] <= "1956-07-01" or end[:4] == "1990"

URBAN_ROOTS = ("Q22865",)                       # urban district / independent city
DISTRICT_ROOTS = ("Q106658",                    # district of Germany
                  "Q138509364",                 # former district of Germany
                  "Q5283531",                   # district of Prussia
                  "Q15849374")                  # Kreis of the GDR (not Bezirk)
MUNICIPALITY_ROOTS = ("Q262166",)               # municipality in Germany

QUERY_CLASSES = """\
SELECT ?cls ?root WHERE {
  VALUES ?root { %s }
  ?cls wdt:P279* ?root .
}
""" % " ".join(f"wd:{q}" for q in URBAN_ROOTS + DISTRICT_ROOTS + MUNICIPALITY_ROOTS)

QUERY_STATEMENTS = """\
SELECT ?item ?code ?end ?type ?dissolved WHERE {
  VALUES ?country { wd:Q183 wd:Q713750 wd:Q16957 }  # Germany, West Germany, GDR
  ?item p:P395 ?st ; wdt:P17 ?country .
  ?st ps:P395 ?code ; wikibase:rank ?rank .
  FILTER(?rank != wikibase:DeprecatedRank)
  OPTIONAL { ?st pq:P582 ?end }
  OPTIONAL { ?item wdt:P31 ?type }
  OPTIONAL { ?item wdt:P576 ?dissolved }
}
"""

QUERY_INFO = """\
SELECT ?item ?itemLabel ?coord ?pop ?seat ?seatLabel ?seatCoord ?seatRank ?seatEnd
WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item rdfs:label ?itemDe . FILTER(LANG(?itemDe) = "de") }
  OPTIONAL { ?item rdfs:label ?itemEn . FILTER(LANG(?itemEn) = "en") }
  BIND(COALESCE(?itemDe, ?itemEn) AS ?itemLabel)
  OPTIONAL { ?item wdt:P625 ?coord }
  OPTIONAL { ?item wdt:P1082 ?pop }
  OPTIONAL {
    ?item p:P36 ?ps . ?ps ps:P36 ?seat ; wikibase:rank ?seatRank .
    FILTER(?seatRank != wikibase:DeprecatedRank)
    OPTIONAL { ?ps pq:P582 ?seatEnd }
    OPTIONAL { ?seat rdfs:label ?seatDe . FILTER(LANG(?seatDe) = "de") }
    OPTIONAL { ?seat rdfs:label ?seatEn . FILTER(LANG(?seatEn) = "en") }
    BIND(COALESCE(?seatDe, ?seatEn) AS ?seatLabel)
    OPTIONAL { ?seat wdt:P625 ?seatCoord }
  }
}
"""

QUERY_STATES = """\
SELECT ?seat ?state ?stateLabel WHERE {
  VALUES ?seat { %s }
  ?seat wdt:P131* ?state . ?state wdt:P31 wd:Q1221156 .   # state of Germany
  ?state rdfs:label ?stateLabel . FILTER(LANG(?stateLabel) = "de")
}
"""


def _sparql(query: str, *, attempts: int = 4) -> list[dict]:
    for k in range(attempts):
        try:
            r = requests.post(ENDPOINT, data={"query": query},
                              headers={"User-Agent": USER_AGENT,
                                       "Accept": "application/sparql-results+json"},
                              timeout=300)
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            if k + 1 == attempts:
                raise
            print(f"  WDQS transfer failed ({type(e).__name__}); retrying", flush=True)
            time.sleep(15.0 * (k + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504) and k + 1 < attempts:
            wait = float(r.headers.get("Retry-After", 0) or 0) or 15.0 * (k + 1)
            print(f"  WDQS {r.status_code}; retrying in {wait:.0f} s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        break
    return [{k: v["value"] for k, v in b.items()}
            for b in r.json()["results"]["bindings"]]


def _batched(template: str, qids: list[str], batch: int = 200) -> list[dict]:
    rows: list[dict] = []
    for i in range(0, len(qids), batch):
        rows += _sparql(template % " ".join(f"wd:{q}" for q in qids[i:i + batch]))
    return rows


def _qid(uri: str | None) -> str | None:
    return uri.rsplit("/", 1)[-1] if uri else None


def _point(wkt: str | None) -> tuple[float, float] | None:
    """'Point(lon lat)' -> (lat, lon)."""
    if not wkt:
        return None
    m = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", wkt)
    return (float(m.group(2)), float(m.group(1))) if m else None


def _label_ok(label: str | None, uri: str | None) -> bool:
    # The label service falls back to the bare QID when no de/en label exists.
    return bool(label) and label != _qid(uri)


def name_matches(code: str, label: str) -> bool:
    """Code letters spell the place: same first letter, every letter present."""
    c, n = code.casefold(), label.casefold()
    return n.startswith(c[0]) and all(ch in n for ch in c[1:])


def _norm_code(raw: str) -> str | None:
    code = raw.strip().upper()
    return code if CODE_RE.fullmatch(code) else None


def classify(class_rows: list[dict], stmt_rows: list[dict]):
    """``(carriers, invalid)``: code -> qid -> {kind, dissolved, ended}."""
    roots_of: dict[str, set[str]] = defaultdict(set)
    for r in class_rows:
        roots_of[_qid(r["cls"])].add(_qid(r["root"]))
    items: dict[str, dict] = {}
    invalid: set[str] = set()
    for r in stmt_rows:
        end = r.get("end", "")
        if obsolete_system(end):
            invalid.add(f"{r['code']} (ended {end[:10]})")
            continue
        q = _qid(r["item"])
        it = items.setdefault(q, {"types": set(), "dissolved": False, "codes": {}})
        if r.get("type"):
            it["types"] |= roots_of.get(_qid(r["type"]), set())
        it["dissolved"] |= "dissolved" in r
        code = _norm_code(r["code"])
        if code is None:
            invalid.add(r["code"])
            continue
        # A code counts as "ended" on this item only if every statement has P582.
        it["codes"][code] = it["codes"].get(code, True) and "end" in r
    carriers: dict[str, dict[str, dict]] = defaultdict(dict)
    for q, it in items.items():
        t = it["types"]
        if t & set(URBAN_ROOTS):
            kind = "urban_district"
        elif t & set(DISTRICT_ROOTS):
            kind = "district"
        elif t & set(MUNICIPALITY_ROOTS):
            kind = "municipality"
        else:
            continue            # federal agencies, forests, foreign villages ...
        for code, ended in it["codes"].items():
            carriers[code][q] = {"qid": q, "kind": kind,
                                 "dissolved": it["dissolved"], "ended": ended}
    return carriers, invalid


def fetch_info(qids: list[str]) -> dict[str, dict]:
    info: dict[str, dict] = {}
    for r in _batched(QUERY_INFO, sorted(qids)):
        q = _qid(r["item"])
        d = info.setdefault(q, {"label": None, "coord": None, "pop": 0.0, "seats": {}})
        if _label_ok(r.get("itemLabel"), r["item"]):
            d["label"] = r["itemLabel"]
        d["coord"] = d["coord"] or _point(r.get("coord"))
        try:
            d["pop"] = max(d["pop"], float(r.get("pop") or 0))
        except ValueError:
            pass
        if r.get("seat") and _label_ok(r.get("seatLabel"), r["seat"]):
            s = d["seats"].setdefault(_qid(r["seat"]), {
                "qid": _qid(r["seat"]), "label": r["seatLabel"], "coord": None,
                "ended": True, "preferred": False})
            s["coord"] = s["coord"] or _point(r.get("seatCoord"))
            s["ended"] &= "seatEnd" in r
            s["preferred"] |= r.get("seatRank", "").endswith("PreferredRank")
    return info


def resolve(carriers: dict[str, dict[str, dict]], info: dict[str, dict]) -> dict[str, dict]:
    """Choose one seat per code (rules in the module docstring)."""

    def seat_of(q: str) -> dict | None:
        d = info.get(q, {})
        seats = sorted(d.get("seats", {}).values(),
                       key=lambda s: (s["ended"], not s["preferred"],
                                      s["coord"] is None, s["qid"]))
        if seats:
            return seats[0]
        if d.get("label"):
            # No P36: the district itself, named after its seat town in most
            # cases ("Landkreis Riedenburg" -> "Riedenburg").
            return {"qid": q, "label": _DISTRICT_PREFIX.sub("", d["label"]),
                    "coord": d.get("coord")}
        return None

    def self_of(q: str) -> dict | None:
        d = info.get(q, {})
        return ({"qid": q, "label": d["label"], "coord": d.get("coord")}
                if d.get("label") else None)

    def best_muni(munis: list[dict], require_prefix: bool, code: str) -> dict | None:
        c = [m for m in munis if info.get(m["qid"], {}).get("label")
             and (not require_prefix or info[m["qid"]]["label"].casefold().startswith(
                 code.casefold()))]
        c.sort(key=lambda m: (-info[m["qid"]]["pop"], info[m["qid"]]["coord"] is None,
                              m["qid"]))
        return self_of(c[0]["qid"]) if c else None

    out: dict[str, dict] = {}
    for code in sorted(carriers):
        # Open-ended statements first, then QID (determinism).
        ents = sorted(carriers[code].values(), key=lambda e: (e["ended"], e["qid"]))
        # Several cities can carry a code (Wikidata has L on Erfurt, a GDR-era
        # Bezirk letter, as well as on Leipzig): prefer the one it spells.
        urban = sorted((e for e in ents
                        if e["kind"] == "urban_district" and not e["dissolved"]),
                       key=lambda e: (e["ended"], not name_matches(
                           code, info.get(e["qid"], {}).get("label") or ""), e["qid"]))
        former = [e for e in ents if e["kind"] == "district" and e["dissolved"]]
        current = [e for e in ents if e["kind"] == "district" and not e["dissolved"]]
        munis = [e for e in ents if e["kind"] == "municipality" and not e["dissolved"]]
        chosen = via = src = None
        for e in urban:
            if (s := self_of(e["qid"])) is not None:
                chosen, via, src = s, "urban_district", e
                break
        for group, tag in ((current, "district_seat"), (former, "former_district_seat")):
            for e in group if chosen is None else ():
                s = seat_of(e["qid"])
                if s is not None and name_matches(code, s["label"]):
                    chosen, via, src = s, tag + "_name_match", e
                    break
        for group, tag in ((current, "district_seat"), (former, "former_district_seat")):
            for e in group if chosen is None else ():
                if (s := seat_of(e["qid"])) is not None:
                    chosen, via, src = s, tag, e
                    break
        if chosen is None:
            m = best_muni(munis, True, code)
            via = "municipality_name_prefix"
            if m is None:
                m, via = best_muni(munis, False, code), "municipality_largest"
            chosen = m
        if chosen is None:
            continue
        current_carrier = any(not e["dissolved"] and not e["ended"] for e in ents)
        out[code] = {
            "seat": chosen["label"],          # completed by attach_states()
            "seat_label": chosen["label"],
            "seat_qid": chosen["qid"],
            "state": None,
            "lat": round(chosen["coord"][0], 6) if chosen.get("coord") else None,
            "lon": round(chosen["coord"][1], 6) if chosen.get("coord") else None,
            "via": via,
            "entity_qid": src["qid"] if src else chosen["qid"],
            "entity_label": (info.get(src["qid"], {}).get("label") if src
                             else chosen["label"]),
            "historic": not current_carrier,
            "districts": [e["qid"] for e in ents if e["kind"] != "municipality"],
            "n_municipalities": len(munis),
        }
    return out


def attach_states(codes: dict[str, dict]) -> None:
    """Fill each record's ``state`` and make ``seat`` '<seat>, <state>'."""
    state_of: dict[str, str] = {}
    for row in _batched(QUERY_STATES, sorted({r["seat_qid"] for r in codes.values()})):
        q = _qid(row["seat"])
        if _label_ok(row.get("stateLabel"), row.get("state")):
            # Two states only via historic P131 chains; keep the first, sorted.
            state_of[q] = min(state_of.get(q, row["stateLabel"]), row["stateLabel"])
    for r in codes.values():
        st = state_of.get(r["seat_qid"])
        r["state"] = st
        if st and st != r["seat_label"]:     # Berlin, Hamburg are their own state
            r["seat"] = f"{r['seat_label']}, {st}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    retrieved = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    print("querying class closure, P395 statements ...", flush=True)
    carriers, invalid = classify(_sparql(QUERY_CLASSES), _sparql(QUERY_STATEMENTS))
    # Info for every (urban) district, and for the municipalities of the codes
    # no district carries (rule 4 candidates).
    need = set()
    for code, ents in carriers.items():
        dist = {q for q, e in ents.items() if e["kind"] != "municipality"}
        need |= dist if dist else set(ents)
    print(f"  {len(carriers)} valid codes; fetching info for {len(need)} entities ...",
          flush=True)
    info = fetch_info(sorted(need))
    codes = resolve(carriers, info)
    print("querying seat states ...", flush=True)
    attach_states(codes)

    sys.path.insert(0, str(ROOT))
    from monocular_osm.kfz_codes import MANUAL_OVERRIDES  # noqa: E402

    n_hist = sum(r["historic"] for r in codes.values())
    print(f"{len(codes)} codes ({n_hist} historic); "
          f"{len(invalid)} malformed / retired-system code statements skipped")
    vias: dict[str, int] = defaultdict(int)
    for r in codes.values():
        vias[r["via"]] += 1
    print("resolved via:", dict(sorted(vias.items())))
    missing = sorted(set(MANUAL_OVERRIDES) - set(codes))
    print(f"manual overrides not in Wikidata: {missing}")
    differ = [(c, MANUAL_OVERRIDES[c], codes[c]["seat"]) for c in sorted(MANUAL_OVERRIDES)
              if c in codes and not codes[c]["seat_label"].startswith(
                  MANUAL_OVERRIDES[c].split(",")[0].split(" (")[0])]
    print(f"{len(differ)} manual overrides name a different place than Wikidata:")
    for c, man, wd in differ:
        print(f"  {c:4s} manual={man!r:32s} wikidata={wd!r}")

    snapshot = {
        "_meta": {
            "description": ("German Kfz licence-plate district codes -> district seat, "
                            "derived from Wikidata P395. Applied under the manual "
                            "overrides in monocular_osm/kfz_codes.py."),
            "source": "Wikidata Query Service",
            "endpoint": ENDPOINT,
            "license": "CC0 1.0 (Wikidata)",
            "retrieved_utc": retrieved,
            "generator": "scripts/update_kfz_codes.py",
            "n_codes": len(codes),
            "n_historic": n_hist,
            "skipped_codes": sorted(invalid),
            "resolution": [
                "1 current urban district carrying the code -> the city (name match "
                "first when several carry it)",
                "2 district carrying it (current, then dissolved) whose seat (P36) "
                "name matches the code (same first letter, all code letters in it)",
                "3 else the seat of a current, then of a dissolved district",
                "4 codes no district carries -> municipality whose name starts with "
                "the code, else the most populous carrier",
                "seats with an end-time qualifier rank below current ones",
                "statements of retired plate systems (ended <= 1956-07-01 or in "
                "1990, i.e. occupation-zone and GDR codes) are dropped",
                "seat string = '<seat label>, <state label>' (German labels)",
            ],
            "sparql": {"classes": QUERY_CLASSES, "statements": QUERY_STATEMENTS,
                       "info": QUERY_INFO, "states": QUERY_STATES},
        },
        "codes": {c: codes[c] for c in sorted(codes)},
    }
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"wrote {args.out} ({len(codes)} codes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
