"""Text normalisation shared by blocking and features.

Every rule applies to every record regardless of its country label, so an unseen
country (France appears only in test) goes through exactly the same path.
The maps below are hand-written spelling/abbreviation rules, not lookups of any
business identity.
"""
import re
import unicodedata

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[\W_]+")
_APOS = re.compile(r"['’`´]")

# Multi-word phrases rewritten before tokenising (longest first).
PHRASES = {
    # legal forms
    "limited liability company": "llc",
    "limited liability partnership": "llp",
    "private limited": "pvt ltd",
    "public limited company": "plc",
    "doing business as": "dba",
    "trading as": "dba",
    "and company": "co",
    "and co": "co",
    "and sons": "sons",
    "societe anonyme": "sa",
    "societe par actions simplifiee": "sas",
    "societe a responsabilite limitee": "sarl",
    # US states
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "south carolina": "sc",
    "south dakota": "sd", "west virginia": "wv", "rhode island": "ri",
    "district of columbia": "dc",
    # Indian states / UTs
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "himachal pradesh": "hp",
    "madhya pradesh": "mp", "uttar pradesh": "up", "tamil nadu": "tn",
    "west bengal": "wb", "jammu and kashmir": "jk", "new delhi": "delhi",
    "post office": "po", "post box": "po box",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
IN_STATES = {
    "assam": "as", "bihar": "br", "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "telangana": "ts", "tripura": "tr",
    "uttarakhand": "uk", "uttaranchal": "uk", "chandigarh": "chd", "puducherry": "py",
    "pondicherry": "py", "maharashtra": "mh",
}
# Old/new or transliterated spellings of the same city.
CITY_ALIASES = {
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurgaon": "gurugram", "poona": "pune", "trivandrum": "thiruvananthapuram",
    "mysore": "mysuru", "baroda": "vadodara", "cochin": "kochi", "benares": "varanasi",
    "banaras": "varanasi", "mangalore": "mangaluru", "belgaum": "belagavi",
    "hubli": "hubballi", "vizag": "visakhapatnam", "allahabad": "prayagraj",
}

NAME_MAP = {
    "private": "pvt", "prvt": "pvt", "pvt": "pvt",
    "limited": "ltd", "ltd": "ltd",
    "corporation": "corp", "corpn": "corp", "corp": "corp",
    "company": "co", "comp": "co", "cmpny": "co", "compagnie": "co", "cie": "co",
    "incorporated": "inc", "incorp": "inc",
    "enterprises": "ent", "enterprise": "ent", "entp": "ent", "enterprize": "ent",
    "industries": "ind", "industry": "ind", "inds": "ind", "indus": "ind",
    "brothers": "bros", "bro": "bros",
    "international": "intl", "intnl": "intl", "internat": "intl",
    "technologies": "tech", "technology": "tech", "techno": "tech",
    "services": "svc", "service": "svc", "svcs": "svc", "servs": "svc",
    "solutions": "soln", "solution": "soln", "solns": "soln",
    "manufacturing": "mfg", "manufacturers": "mfg", "manufacturer": "mfg", "mfrs": "mfg",
    "associates": "assoc", "associate": "assoc", "asso": "assoc", "assocs": "assoc",
    "center": "ctr", "centre": "ctr", "cntr": "ctr",
    "hospital": "hosp", "hospitals": "hosp",
    "department": "dept",
    "saint": "st", "sainte": "ste", "mount": "mt",
    "university": "univ", "institute": "inst", "national": "natl",
    "group": "grp", "holdings": "hldg", "holding": "hldg", "hldgs": "hldg",
    "systems": "sys", "system": "sys",
    "laboratories": "lab", "laboratory": "lab", "labs": "lab",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma",
    "engineering": "engg", "engineers": "engg",
    "jewellers": "jewel", "jewelers": "jewel", "jewellery": "jewel", "jewelry": "jewel",
    "chemicals": "chem", "chemical": "chem",
    "electricals": "elec", "electrical": "elec",
    "etablissements": "ets",
    "et": "and",
}
LEGAL = {
    "pvt", "ltd", "llc", "llp", "inc", "corp", "co", "plc", "lp", "lllp", "pllc", "pc",
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "gmbh", "ag", "pte", "pty", "opc",
    "ltda", "bv", "nv",
}
NAME_DROP = {"the", "and", "of", "ms", "messrs", "de", "du", "des", "la", "le", "les", "d", "l"}

ADDR_MAP = {
    "street": "st", "str": "st", "strt": "st", "saint": "st",
    "road": "rd", "avenue": "ave", "av": "ave", "aven": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "drive": "dr", "drv": "dr",
    "lane": "ln", "place": "pl", "square": "sq", "court": "ct", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "expressway": "expy", "terrace": "ter",
    "trail": "trl", "suite": "ste", "apartment": "apt", "building": "bldg", "bld": "bldg",
    "batiment": "bldg", "bat": "bldg", "floor": "fl", "flr": "fl", "etage": "fl",
    "room": "rm", "number": "no", "num": "no",
    "near": "near", "nr": "near", "opposite": "opp", "opst": "opp", "bh": "behind",
    "bhnd": "behind", "north": "n", "nth": "n", "south": "s", "sth": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "nager": "nagar", "ngr": "nagar", "sector": "sec", "sect": "sec", "phase": "ph",
    "colony": "colony", "clny": "colony", "extension": "ext", "extn": "ext",
    "industrial": "indl", "estate": "est", "district": "dist", "distt": "dist",
    "tehsil": "teh", "village": "vill", "vlg": "vill", "bazaar": "bazar", "bzr": "bazar",
    "mohala": "mohalla", "chemin": "chemin", "chem": "chemin", "impasse": "imp",
    "allee": "allee", "route": "rte", "faubourg": "fbg", "residence": "res",
}
ADDR_MAP.update(US_STATES)
ADDR_MAP.update(IN_STATES)
ADDR_MAP.update(CITY_ALIASES)
ADDR_DROP = {"cedex", "the", "of", "de", "du", "des", "la", "le", "les", "d", "l"}
LANDMARK = {"near", "opp", "behind", "beside", "besides", "next", "adjacent", "above",
            "below", "landmark", "facing", "front"}

_PHRASE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(PHRASES, key=len, reverse=True)) + r")\b"
)
_PIN_SPLIT = re.compile(r"\b(\d{3}) (\d{3})$")
_DIGITS = re.compile(r"\d")


def fold(text: str) -> str:
    """Lowercase, strip Latin accents (é->e), map &->and, drop punctuation."""
    text = unicodedata.normalize("NFKD", _APOS.sub("", text or ""))
    out, prev_ascii = [], False
    for ch in text:
        if prev_ascii and unicodedata.combining(ch):
            continue
        out.append(ch)
        prev_ascii = ch.isascii()
    text = "".join(out).lower().replace("&", " and ").replace("@", " at ")
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _join_initials(tokens):
    """'l l c' -> 'llc', 'i b m' -> 'ibm' (abbreviations written with dots)."""
    out, buf = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf))
    return out


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t


def _tokens(text: str, mapping: dict) -> list:
    text = _PHRASE_RE.sub(lambda m: PHRASES[m.group(1)], fold(text))
    return [_stem(mapping.get(t, t)) for t in _join_initials(text.split())]


def normalize_name(raw: str) -> dict:
    toks = _tokens(raw, NAME_MAP)
    legal = [t for t in toks if t in LEGAL]
    core = [t for t in toks if t not in LEGAL and t not in NAME_DROP] or toks
    # DBA / trade names: keep each side as an alternative name.
    alts = [" ".join(core)]
    if "dba" in core:
        i = core.index("dba")
        alts = [" ".join(p) for p in (core[:i], core[i + 1:]) if p] + [" ".join(core[:i] + core[i + 1:])]
    core_str = " ".join(t for t in core if t != "dba")
    return {
        "name_full": " ".join(toks),
        "name_core": core_str,
        "name_legal": " ".join(sorted(set(legal))),
        "name_alts": alts,
        "name_initials": "".join(t[0] for t in core if t != "dba" and t[0].isalpha()),
        "name_compact": core_str.replace(" ", ""),
    }


def normalize_address(raw: str) -> dict:
    folded = _PIN_SPLIT.sub(r"\1\2", fold(raw))  # "400 001" -> "400001"
    text = _PHRASE_RE.sub(lambda m: PHRASES[m.group(1)], folded)
    toks = [_stem(ADDR_MAP.get(t, t)) for t in _join_initials(text.split())]
    toks = [t for t in toks if t not in ADDR_DROP]
    postal = ""
    for t in reversed(toks):
        if t.isdigit() and len(t) in (5, 6):
            postal = t
            break
    nums = [t for t in toks if _DIGITS.search(t) and t != postal]
    words = [t for t in toks if not _DIGITS.search(t)]
    return {
        "addr_full": " ".join(toks),
        "addr_words": " ".join(words),
        "addr_postal": postal,
        "addr_nums": " ".join(nums),
        "addr_house": nums[0] if nums else "",
        "addr_landmark": int(any(t in LANDMARK for t in toks)),
    }


def normalize_country(raw: str) -> str:
    return fold(raw)
