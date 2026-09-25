"""SYNTHETIC smoke-test data in the challenge's file layout. It only proves the pipeline
runs end to end and writes valid files. It is not used for training and its scores mean
nothing about the real leaderboard.

    python tools/make_synthetic.py --out student_resource_synthetic/dataset
"""
import argparse
import random
from pathlib import Path

R = random.Random(7)
SPEC = {
    "US": dict(
        a=["Blue", "Golden", "Eagle", "Summit", "Pioneer", "Liberty", "Silver", "Oak", "River", "Maple",
           "Northern", "Lakeside", "Harbor", "Prime", "Evergreen", "Johnson", "Miller", "Garcia"],
        b=["Bakery", "Auto Repair", "Dental Care", "Hardware", "Consulting", "Logistics", "Pharmacy",
           "Diner", "Fitness", "Plumbing", "Coffee House", "Electric", "Books", "Insurance"],
        legal=["LLC", "Inc", "Incorporated", "Corporation", "Corp", "Co", ""],
        street=["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Lake", "Hill", "Park"],
        stype=[("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr")],
        city=[("Austin", "Texas", "TX"), ("Denver", "Colorado", "CO"), ("Seattle", "Washington", "WA"),
              ("Columbus", "Ohio", "OH"), ("Phoenix", "Arizona", "AZ"), ("Portland", "Oregon", "OR")],
        postal=5),
    "India": dict(
        a=["Shree", "Laxmi", "Ganesh", "Sai", "Om", "Balaji", "Krishna", "Durga", "Sharma", "Patel",
           "Gupta", "Mehta", "Agarwal", "Reddy", "Iyer", "Bharat"],
        b=["Traders", "Enterprises", "Textiles", "Jewellers", "Electricals", "Medical Store", "Sweets",
           "Hardware", "Motors", "Industries", "Steel", "Agencies"],
        legal=["Pvt Ltd", "Private Limited", "LLP", ""],
        street=["MG", "Station", "Gandhi", "Nehru", "Ring", "Temple", "Market", "Link"],
        stype=[("Road", "Rd"), ("Marg", "Marg"), ("Street", "St")],
        city=[("Mumbai", "Maharashtra", "MH"), ("Bengaluru", "Karnataka", "KA"),
              ("Chennai", "Tamil Nadu", "TN"), ("Jaipur", "Rajasthan", "RJ"), ("Pune", "Maharashtra", "MH")],
        postal=6),
    "France": dict(
        a=["Boulangerie", "Café", "Pharmacie", "Garage", "Librairie", "Hôtel", "Brasserie", "Atelier"],
        b=["du Parc", "de la Gare", "Saint-Michel", "Lumière", "des Arts", "Martin", "Dubois", "Lefèvre"],
        legal=["SARL", "SAS", "SA", ""],
        street=["Victor Hugo", "de la République", "Jean Jaurès", "Pasteur", "de Verdun", "Saint-Denis"],
        stype=[("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Place", "Pl.")],
        city=[("Paris", "Île-de-France", "IDF"), ("Lyon", "Rhône", "69"), ("Marseille", "Provence", "13")],
        postal=5),
}
LANDMARKS = ["Near SBI ATM", "Opp. Bus Stand", "Behind City Mall", "Near Railway Station"]


def entity(country):
    s = SPEC[country]
    name = f"{R.choice(s['a'])} {R.choice(s['b'])}"
    city, state, st_abbr = R.choice(s["city"])
    stype = R.choice(s["stype"])
    num = str(R.randint(1, 999))
    postal = "".join(R.choice("0123456789") for _ in range(s["postal"]))
    if country == "France":
        addr = [f"{num} {stype[0]} {R.choice(s['street'])}", f"{postal} {city}"]
    elif country == "India":
        addr = [f"{num}, {R.choice(s['street'])} {stype[0]}", f"{R.choice(['Shastri', 'Rajaji', 'Anna'])} Nagar",
                city, f"{state} {postal}"]
    else:
        addr = [f"{num} {R.choice(s['street'])} {stype[0]}", city, f"{state} {postal}"]
    return dict(name=name, legal=R.choice(s["legal"]), addr=addr, stype=stype, st_abbr=st_abbr,
                state=state, country=country)


def typo(w):
    if len(w) < 4:
        return w
    k = R.randrange(1, len(w) - 1)
    return R.choice([w[:k] + w[k + 1:], w[:k] + w[k + 1] + w[k] + w[k + 2:], w[:k] + w[k] + w[k:]])


def render(e, noisy):
    name, legal, addr = e["name"], e["legal"], list(e["addr"])
    if noisy:
        words = name.split()
        if R.random() < 0.3:
            i = R.randrange(len(words))
            words[i] = typo(words[i])
        if R.random() < 0.15 and len(words) > 1:
            words[0], words[1] = words[1], words[0]
        name = " ".join(words)
        if R.random() < 0.4:
            legal = R.choice(SPEC[e["country"]]["legal"])
        name = name.replace(" and ", " & ") if R.random() < 0.5 else name
        long_t, short_t = e["stype"]
        addr[0] = addr[0].replace(long_t, short_t) if R.random() < 0.5 else addr[0]
        if R.random() < 0.3:
            addr = [a for a in addr if not any(c.isdigit() for c in a) or a is addr[0]]
        if R.random() < 0.3:
            addr = [a.replace(e["state"], e["st_abbr"]) for a in addr]
        if e["country"] == "India" and R.random() < 0.3:
            addr.insert(1, R.choice(LANDMARKS))
        if R.random() < 0.2:
            name = name.upper()
    return (f"{name} {legal}".strip(), ", ".join(addr))


def split(n_s1, countries, prefix_off):
    s1, s2, s3, gt = [], [], [], []
    ents = [entity(R.choice(countries)) for _ in range(n_s1)]
    for k, e in enumerate(ents):
        sid = f"S1-{prefix_off + k:05d}"
        n, a = render(e, noisy=False)
        s1.append((sid, n, a, e["country"]))
        matches = []
        for _ in range(R.choices([0, 1, 2, 3], [0.35, 0.35, 0.2, 0.1])[0]):
            tgt = R.choice([s2, s3])
            rid = f"{'S2' if tgt is s2 else 'S3'}-{len(tgt) + prefix_off:05d}"
            tgt.append((rid, *render(e, noisy=True), e["country"]))
            matches.append(rid)
        gt.append((sid, ",".join(matches)))
        if R.random() < 0.15:  # a different branch of the same chain: same name, other address
            other = dict(entity(e["country"]), name=e["name"], legal=e["legal"])
            tgt = R.choice([s2, s3])
            tgt.append((f"{'S2' if tgt is s2 else 'S3'}-{len(tgt) + prefix_off:05d}", *render(other, True), e["country"]))
    for _ in range(n_s1 // 3):  # records with no S1 counterpart
        e = entity(R.choice(countries))
        tgt = R.choice([s2, s3])
        tgt.append((f"{'S2' if tgt is s2 else 'S3'}-{len(tgt) + prefix_off:05d}", *render(e, True), e["country"]))
    for lst in (s2, s3):
        R.shuffle(lst)
    return s1, s2, s3, gt


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="student_resource_synthetic/dataset")
    args = ap.parse_args()
    out = Path(args.out)
    cols = ["entity_id", "business_name", "business_address", "country"]
    for name, n, countries, off in (("train", 2000, ["US", "India"], 0), ("test", 900, ["US", "India", "France"], 50000)):
        s1, s2, s3, gt = split(n, countries, off)
        for k, rows in ((1, s1), (2, s2), (3, s3)):
            write(out / name / f"{name}_source{k}.tsv", cols, rows)
        if name == "train":
            write(out / name / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    print(f"wrote synthetic data to {out}")


if __name__ == "__main__":
    main()
