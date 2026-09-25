"""Hand-written normalisation rules (abbreviations, legal forms, state names) used by prep.py.

Every rule applies to every record regardless of its country label, so an unseen
country (France appears only in test) goes through exactly the same path.
The maps below are small hand-written abbreviation rules (allowed per the organisers'
Q&A); no gazetteer, postal or business data is bundled or looked up.
"""
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
ADDR_DROP = {"cedex", "the", "of", "de", "du", "des", "la", "le", "les", "d", "l"}
LANDMARK = {"near", "opp", "behind", "beside", "besides", "next", "adjacent", "above",
            "below", "landmark", "facing", "front"}
