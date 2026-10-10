"""German vehicle-registration (Kfz-Kennzeichen) district codes -> geocodable
district / seat name. Used by the license-plate district anchor to turn the modal
plate prefix into a coarse location.

Two layers:

* ``data/kfz_codes.json`` - a committed, dated snapshot derived from Wikidata
  (property P395 "licence plate code" on German districts, urban districts and
  municipalities, current and historic) by ``scripts/update_kfz_codes.py``.
  Each code maps to its district seat as ``"<seat>, <state>"`` plus the seat's
  coordinates and Wikidata ids (see :data:`KFZ_SNAPSHOT`).
* :data:`MANUAL_OVERRIDES` - the original hand-curated table, applied ON TOP of
  the snapshot. It is dense around our test clips (Baden-Wurttemberg + Bavarian
  Swabia) and encodes verified disambiguations (FDB = Friedberg in Bayern vs
  FB = Friedberg in Hessen, MN = Mindelheim, EN = Schwelm, ...), so every code
  it lists resolves exactly as before the snapshot existed -- except RID, LOS
  and PM, three entries the snapshot showed to be wrong (fixed 2026-10).
  ``scripts/update_kfz_codes.py`` prints every override that names a
  different place than Wikidata, for review.

:data:`KFZ_DISTRICTS` is the merged view (snapshot, then overrides). Codes in
neither simply don't vote. Values are place names that geocode cleanly via
Nominatim (``"<name>, Germany"``).
"""

from __future__ import annotations

import json
import re
import warnings
from importlib import resources

_CODE_RE = re.compile(r"^[A-ZÄÖÜ]{1,3}$")
_SNAPSHOT = "data/kfz_codes.json"

MANUAL_OVERRIDES: dict[str, str] = {
    # --- city-states & major cities ---
    "B": "Berlin", "M": "München", "HH": "Hamburg", "K": "Köln",
    "F": "Frankfurt am Main", "S": "Stuttgart", "D": "Düsseldorf",
    "H": "Hannover", "N": "Nürnberg", "L": "Leipzig", "DD": "Dresden",
    "HB": "Bremen", "DO": "Dortmund", "E": "Essen", "DU": "Duisburg",
    "BO": "Bochum", "MA": "Mannheim", "KA": "Karlsruhe", "FR": "Freiburg",
    "HD": "Heidelberg", "WÜ": "Würzburg", "R": "Regensburg", "A": "Augsburg",
    "KS": "Kassel", "MZ": "Mainz", "W": "Wuppertal", "BN": "Bonn",
    "MS": "Münster", "AC": "Aachen", "BI": "Bielefeld", "KL": "Kaiserslautern",
    "SB": "Saarbrücken", "EF": "Erfurt", "MD": "Magdeburg", "RO": "Rosenheim",
    "KI": "Kiel", "HL": "Lübeck", "OS": "Osnabrück", "BS": "Braunschweig",
    "G": "Gera", "C": "Chemnitz", "J": "Jena", "P": "Potsdam", "FL": "Flensburg",
    "WI": "Wiesbaden", "OF": "Offenbach am Main", "DA": "Darmstadt",
    "GI": "Gießen", "MR": "Marburg", "FD": "Fulda", "HU": "Hanau",
    # --- Baden-Württemberg (our clip region) ---
    "UL": "Ulm", "BC": "Biberach an der Riß", "HDH": "Heidenheim an der Brenz",
    "GP": "Göppingen", "AA": "Aalen", "SIG": "Sigmaringen", "RV": "Ravensburg",
    "RT": "Reutlingen", "TÜ": "Tübingen", "ES": "Esslingen am Neckar",
    "BB": "Böblingen", "LB": "Ludwigsburg", "WN": "Waiblingen", "GÖ": "Göttingen",
    "HN": "Heilbronn", "KÜN": "Künzelsau", "SHA": "Schwäbisch Hall",
    "TBB": "Tauberbischofsheim", "HDB": "Eberbach", "MOS": "Mosbach",
    "PF": "Pforzheim", "CW": "Calw", "FDS": "Freudenstadt", "RW": "Rottweil",
    "VS": "Villingen-Schwenningen", "TUT": "Tuttlingen", "KN": "Konstanz",
    "ÜB": "Überlingen", "WT": "Waldshut-Tiengen", "LÖ": "Lörrach",
    "EM": "Emmendingen", "OG": "Offenburg", "BAD": "Baden-Baden",
    "RA": "Rastatt", "BL": "Balingen", "FN": "Friedrichshafen", "BT": "Bayreuth",
    # --- Bavaria, esp. Swabia (next to Ulm) ---
    "NU": "Neu-Ulm", "GZ": "Günzburg", "DLG": "Dillingen an der Donau",
    "MN": "Mindelheim", "KE": "Kempten", "OA": "Oberallgäu", "MM": "Memmingen",
    "DON": "Donauwörth", "AIC": "Aichach",
    "FDB": "Friedberg, Bayern",       # NOT the Hessian Friedberg (FB), 250 km away
    "LL": "Landsberg am Lech",
    "KF": "Kaufbeuren", "MOD": "Marktoberdorf", "SF": "Sonthofen",
    "IN": "Ingolstadt", "EI": "Eichstätt", "ND": "Neuburg an der Donau",
    "PA": "Passau", "LA": "Landshut", "DEG": "Deggendorf", "SR": "Straubing",
    "WM": "Weilheim", "GAP": "Garmisch-Partenkirchen", "MB": "Miesbach",
    "TÖL": "Bad Tölz", "FFB": "Fürstenfeldbruck", "DAH": "Dachau",
    "FS": "Freising", "ED": "Erding", "EBE": "Ebersberg", "STA": "Starnberg",
    "WÜM": "Waldmünchen", "AS": "Amberg", "SAD": "Schwandorf",
    "CHA": "Cham", "BGL": "Bad Reichenhall", "TS": "Traunstein", "MÜ": "Mühldorf am Inn",
    "AÖ": "Altötting", "FRG": "Freyung", "REG": "Regen",
    # RID was "Rottal-Inn" (whose codes are PAN/EG/GRI, ~100 km away): RID is
    # the reintroduced code of the former Landkreis Riedenburg, now in Kelheim
    # (corrected 2026-10 against the Wikidata snapshot).
    "RID": "Riedenburg",
    "KEH": "Kelheim", "NM": "Neumarkt in der Oberpfalz", "WEN": "Weiden",
    "BA": "Bamberg", "CO": "Coburg", "HO": "Hof", "KU": "Kulmbach",
    "LIF": "Lichtenfels", "FO": "Forchheim", "ER": "Erlangen",
    "FÜ": "Fürth", "AN": "Ansbach", "WUG": "Weißenburg", "RH": "Roth",
    "NEA": "Neustadt an der Aisch", "SW": "Schweinfurt", "HAS": "Haßfurt",
    "KG": "Bad Kissingen", "NES": "Bad Neustadt an der Saale", "MIL": "Miltenberg",
    "AB": "Aschaffenburg", "KT": "Kitzingen", "MSP": "Karlstadt",
    # --- NRW / Rhineland / other major districts ---
    "GE": "Gelsenkirchen", "HA": "Hagen", "HAM": "Hamm", "SO": "Soest",
    "PB": "Paderborn", "GT": "Gütersloh", "HF": "Herford", "MI": "Minden",
    "LIP": "Detmold", "HX": "Höxter", "OB": "Oberhausen", "MH": "Mülheim an der Ruhr",
    "RE": "Recklinghausen", "BOR": "Borken", "COE": "Coesfeld", "ST": "Steinfurt",
    "WAF": "Warendorf", "SU": "Siegburg", "GM": "Gummersbach", "BM": "Bergheim",
    "DN": "Düren", "EU": "Euskirchen", "HS": "Heinsberg", "VIE": "Viersen",
    "NE": "Neuss", "ME": "Mettmann", "SG": "Solingen", "RS": "Remscheid",
    "LEV": "Leverkusen", "KR": "Krefeld", "MG": "Mönchengladbach", "KLE": "Kleve",
    "WES": "Wesel", "SI": "Siegen", "OE": "Olpe", "MK": "Lüdenscheid",
    "EN": "Schwelm", "UN": "Unna", "HER": "Herne", "BOT": "Bottrop",
    # --- Hesse / RLP / Saarland ---
    "GG": "Groß-Gerau", "RÜD": "Rüdesheim am Rhein", "HG": "Bad Homburg",
    "FB": "Friedberg (Hessen)",       # NOT the Bavarian Friedberg (FDB)
    "LM": "Limburg an der Lahn", "WW": "Montabaur",
    "KO": "Koblenz", "NR": "Neuwied", "AK": "Altenkirchen", "SIM": "Simmern",
    "TR": "Trier", "WIL": "Wittlich", "BIT": "Bitburg", "DAU": "Daun",
    "LD": "Landau in der Pfalz", "SÜW": "Bad Bergzabern", "GER": "Germersheim",
    "SP": "Speyer", "FT": "Frankenthal", "LU": "Ludwigshafen", "NW": "Neustadt an der Weinstraße",
    "DÜW": "Bad Dürkheim", "KIB": "Kirchheimbolanden", "KH": "Bad Kreuznach",
    "AZ": "Alzey", "WO": "Worms", "BIR": "Birkenfeld", "NK": "Neunkirchen",
    "HOM": "Homburg", "SLS": "Saarlouis", "MZG": "Merzig", "WND": "St. Wendel",
    # --- North / East ---
    "OL": "Oldenburg", "EL": "Lingen (Ems)", "LER": "Leer", "AUR": "Aurich",
    "WHV": "Wilhelmshaven", "DEL": "Delmenhorst", "VEC": "Vechta", "CLP": "Cloppenburg",
    "DH": "Diepholz", "NI": "Nienburg", "VER": "Verden", "ROW": "Rotenburg (Wümme)",
    "STD": "Stade", "CUX": "Cuxhaven", "WL": "Winsen", "LG": "Lüneburg",
    "UE": "Uelzen", "CE": "Celle", "GF": "Gifhorn", "WOB": "Wolfsburg",
    "HI": "Hildesheim", "NOM": "Northeim", "HM": "Hameln", "HOL": "Holzminden",
    "SHG": "Stadthagen", "PE": "Peine", "SZ": "Salzgitter", "GS": "Goslar",
    "RZ": "Ratzeburg", "OD": "Bad Oldesloe", "SE": "Bad Segeberg", "PI": "Pinneberg",
    "IZ": "Itzehoe", "HEI": "Heide", "NF": "Husum", "SL": "Schleswig",
    "RD": "Rendsburg", "PLÖ": "Plön", "OH": "Eutin", "NMS": "Neumünster",
    "SN": "Schwerin", "HRO": "Rostock", "HGW": "Greifswald", "HST": "Stralsund",
    "MÜR": "Waren", "NB": "Neubrandenburg", "HWI": "Wismar",
    "HAL": "Halle", "DE": "Dessau", "WB": "Wittenberg",
    "SK": "Merseburg", "BLK": "Naumburg", "SLK": "Bernburg", "SDL": "Stendal",
    "EIC": "Heilbad Heiligenstadt", "NDH": "Nordhausen", "SÖM": "Sömmerda",
    "GTH": "Gotha", "SM": "Schmalkalden", "SHL": "Suhl", "SON": "Sonneberg",
    "SLF": "Saalfeld", "GRZ": "Greiz", "ABG": "Altenburg", "Z": "Zwickau",
    "V": "Plauen", "FG": "Freiberg", "PIR": "Pirna", "GR": "Görlitz",
    "BZ": "Bautzen", "MEI": "Meißen", "TDO": "Torgau", "LÖB": "Löbau",
    "BAR": "Eberswalde", "MOL": "Strausberg",
    # LOS / PM named the neighbouring kreisfreie Staedte Frankfurt (Oder) (FF)
    # and Brandenburg an der Havel (BRB), which lie OUTSIDE those districts;
    # use the district seats (Oder-Spree: Beeskow, Potsdam-Mittelmark: Bad
    # Belzig; corrected 2026-10 against the Wikidata snapshot).
    "LOS": "Beeskow", "PM": "Bad Belzig",
    "TF": "Luckenwalde", "HVL": "Rathenow",
    "OPR": "Neuruppin", "PR": "Perleberg", "UM": "Prenzlau", "CB": "Cottbus",
    "SPN": "Spremberg", "EE": "Herzberg", "OHV": "Oranienburg",
}


def _load_snapshot() -> tuple[dict[str, dict], dict]:
    """``(codes, meta)`` from the packaged Wikidata snapshot; empty (with a
    warning) if it is missing or unreadable, so the overrides still work."""
    try:
        raw = resources.files("monocular_osm").joinpath(
            *_SNAPSHOT.split("/")).read_text(encoding="utf-8")
        blob = json.loads(raw)
        codes = {c: r for c, r in blob["codes"].items()
                 if _CODE_RE.fullmatch(c) and isinstance(r, dict) and r.get("seat")}
        return codes, blob.get("_meta", {})
    except (OSError, ValueError, KeyError, TypeError) as e:  # pragma: no cover
        warnings.warn(f"Kfz snapshot {_SNAPSHOT} unavailable ({e}); using the "
                      f"{len(MANUAL_OVERRIDES)} manual codes only", RuntimeWarning,
                      stacklevel=2)
        return {}, {}


#: Wikidata snapshot records keyed by code (seat, state, lat/lon, QIDs, historic).
KFZ_SNAPSHOT, KFZ_SNAPSHOT_META = _load_snapshot()

#: Merged code -> geocodable seat name: Wikidata snapshot, manual overrides win.
KFZ_DISTRICTS: dict[str, str] = {
    **{c: r["seat"] for c, r in KFZ_SNAPSHOT.items()},
    **MANUAL_OVERRIDES,
}
