"""
mlb_ml_trainer.py  -  MLB MONEYLINE model, Elo-based (same backbone as the Dota 2 / CS2 apps)

What it rates, the way the esports models do:
  team Elo        run-margin scaled, regressed between seasons, strength of schedule
  pitcher Elo     every starting pitcher, updated on runs allowed per out against the offence faced
  batter Elo      every hitter, updated on wOBA per plate appearance against the pitcher faced
  bullpen Elo     each team's relief corps, with a fatigue term for outs thrown in the last three days
  defence Elo     team run prevention with the starter's own contribution taken out

On top of that, everything baseball-specific that can actually be fetched:
  park run factor and its handedness split, altitude, roof
  weather at first pitch: temperature, wind component out to centre field, air density
  head to head, and each hitter's own history against the pitcher he is facing
  each pitcher's history against that specific opponent
  starter rest days, team rest days, bullpen fatigue, travel distance and time-zone shift
  platoon edges in the posted lineup, day/night, doubleheaders, series game number
  season win %, games back, last-10 and last-30 form

Data (all free, no keys):
  MLB Stats API   schedule, box scores (lineups, batting lines, pitchers, umpires), venues, people
  Open-Meteo      historical hourly weather at each park

Setup:  pip install requests numpy
Run:    python mlb_ml_trainer.py                                 # 2019-2026, everything
        python mlb_ml_trainer.py --seasons 2023 2024 2025 2026
        python mlb_ml_trainer.py --market odds_mlb.csv           # also fit the market stack

First full run fetches one box score per game (about 2,400 a season) so give it an hour or two.
Everything lands in cache_mlb_ml/, so every run after that only picks up new games.
"""

import argparse, csv, json, math, os, sys, time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

import numpy as np
import requests

API = "https://statsapi.mlb.com/api/v1"
CACHE = "cache_mlb_ml"
OUT = "model.json"

# ---- Elo constants. Baseball is the highest-variance major sport, so K is small and the
# ---- single-game signal is weak; the ratings earn their accuracy over a full season.
BASE = 1500.0
K_TEAM = 6.0            # team Elo, per game, before margin scaling
K_SP = 16.0             # starting pitcher
K_BAT = 7.0             # hitter
K_PEN = 5.0             # bullpen
K_DEF = 4.0             # team run prevention
CARRY = 0.70            # fraction of a team's Elo above base carried into the next season
CARRY_P = 0.80          # same for players, who move teams but keep their ability
HOME_ELO = 24.0         # home field, in Elo points, used only inside the Elo expectation
SOS_N = 30              # rolling window for strength of schedule
FORM_N = 10
FORM_N2 = 30
TEST_FRACTION = 0.2

WOBA = {"bb": 0.69, "hbp": 0.72, "1b": 0.88, "2b": 1.27, "3b": 1.62, "hr": 2.10}
LG_WOBA = 0.315

# bearing in degrees from home plate toward centre field, and park elevation in metres.
PARK_GEO = {"Angel Stadium": (50, 50), "Busch Stadium": (60, 140), "Chase Field": (0, 330), "Citi Field": (30, 3),
            "Citizens Bank Park": (10, 10), "Comerica Park": (150, 180), "Coors Field": (0, 1580),
            "Daikin Park": (345, 15), "Minute Maid Park": (345, 15), "Dodger Stadium": (25, 80),
            "Fenway Park": (45, 6), "Nationals Park": (30, 5), "George M. Steinbrenner Field": (20, 10),
            "Yankee Stadium": (75, 16), "Yankee Stadium III": (75, 16), "Globe Life Field": (25, 170),
            "Great American Ball Park": (120, 150), "Guaranteed Rate Field": (130, 180), "Rate Field": (130, 180),
            "Kauffman Stadium": (45, 230), "loanDepot park": (40, 3), "Oracle Park": (85, 3),
            "Oriole Park at Camden Yards": (30, 10), "PNC Park": (120, 220), "Petco Park": (0, 20),
            "Progressive Field": (0, 200), "American Family Field": (130, 200), "Rogers Centre": (345, 90),
            "Sutter Health Park": (30, 10), "T-Mobile Park": (60, 5), "Target Field": (100, 250),
            "Truist Park": (150, 320), "Wrigley Field": (30, 180)}


# ============================================================== fetching
def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60, headers={"User-Agent": "MLBMoneylineTrainer/1.0"})
            if r.status_code == 429:
                time.sleep(10 * (i + 1)); continue
            r.raise_for_status(); return r.json()
        except requests.RequestException:
            if i == tries - 1: raise
            time.sleep(3)


def jload(name, default):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p): return default
    try: return json.load(open(p, encoding="utf-8"))
    except Exception: return default


def jsave(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    json.dump(obj, open(os.path.join(CACHE, name), "w", encoding="utf-8"))


def ip_to_outs(s):
    try:
        w, f = (str(s).split(".") + ["0"])[:2]
        return int(w) * 3 + int(f)
    except Exception:
        return 0


def woba_of(b):
    """wOBA numerator/denominator from one batting line."""
    den = (b["ab"] + b["bb"] + b["hbp"] + b["sf"])
    num = (WOBA["bb"] * b["bb"] + WOBA["hbp"] * b["hbp"] + WOBA["1b"] * b["1b"]
           + WOBA["2b"] * b["2b"] + WOBA["3b"] * b["3b"] + WOBA["hr"] * b["hr"])
    return num, den


def fetch_season(season):
    have = {g["pk"]: g for g in jload(f"games_{season}.json", [])}
    sched = get(f"{API}/schedule", {"sportId": 1, "season": season, "gameType": "R,F,D,L,W",
                                    "hydrate": "probablePitcher,linescore,venue",
                                    "startDate": f"{season}-03-01", "endDate": f"{season}-11-30"})
    games = [g for d in sched.get("dates", []) for g in d.get("games", [])
             if g.get("status", {}).get("codedGameState") == "F"]
    todo = [g for g in games if g["gamePk"] not in have]
    print(f"{season}: {len(games)} completed games, {len(have)} cached, {len(todo)} to fetch")
    for i, g in enumerate(todo):
        try:
            have[g["gamePk"]] = slim_game(g, get(f"{API}/game/{g['gamePk']}/boxscore"))
        except Exception as e:
            print(f"  game {g['gamePk']} failed: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)}")
            jsave(f"games_{season}.json", list(have.values()))
        time.sleep(0.04)
    out = sorted((have[g["gamePk"]] for g in games if g["gamePk"] in have), key=lambda x: x["ts"])
    jsave(f"games_{season}.json", out)
    return out


def slim_game(g, box):
    out = {"pk": g["gamePk"], "date": g["gameDate"][:10],
           "ts": int(datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")).timestamp()),
           "season": int(g.get("season") or g["gameDate"][:4]),
           "venue": (g.get("venue") or {}).get("id"), "venue_name": (g.get("venue") or {}).get("name", ""),
           "night": g.get("dayNight") == "night", "type": g.get("gameType", "R"),
           "dh": g.get("doubleHeader", "N") != "N", "series_game": g.get("seriesGameNumber") or 1,
           "innings": len((g.get("linescore") or {}).get("innings", []))}
    for side in ("home", "away"):
        t = g["teams"][side]; bt = box["teams"][side]; players = bt.get("players", {})
        pitchers = bt.get("pitchers", []); plist = []
        for pid in pitchers:
            p = players.get(f"ID{pid}", {}); st = p.get("stats", {}).get("pitching", {})
            plist.append({"id": pid, "outs": ip_to_outs(st.get("inningsPitched", "0")),
                          "r": st.get("runs", 0) or 0, "er": st.get("earnedRuns", 0) or 0,
                          "k": st.get("strikeOuts", 0) or 0, "bb": st.get("baseOnBalls", 0) or 0,
                          "bf": st.get("battersFaced", 0) or 0,
                          "pc": st.get("pitchesThrown") or st.get("numberOfPitches") or 0,
                          "name": p.get("person", {}).get("fullName", "")})
        bats = []
        for key, p in players.items():
            bo = p.get("battingOrder")
            if not bo: continue
            b = p.get("stats", {}).get("batting", {})
            if not b: continue
            h = b.get("hits", 0) or 0; d2 = b.get("doubles", 0) or 0
            d3 = b.get("triples", 0) or 0; hr = b.get("homeRuns", 0) or 0
            bats.append({"id": p.get("person", {}).get("id"), "order": int(bo),
                         "ab": b.get("atBats", 0) or 0, "bb": b.get("baseOnBalls", 0) or 0,
                         "hbp": b.get("hitByPitch", 0) or 0, "sf": b.get("sacFlies", 0) or 0,
                         "1b": h - d2 - d3 - hr, "2b": d2, "3b": d3, "hr": hr,
                         "name": p.get("person", {}).get("fullName", "")})
        bats.sort(key=lambda x: x["order"])
        out[side] = {"id": t["team"]["id"], "name": t["team"]["name"],
                     "abbr": bt.get("team", {}).get("abbreviation", ""),
                     "runs": t.get("score", 0) or 0, "starter": pitchers[0] if pitchers else None,
                     "pitchers": plist, "lineup": [b["id"] for b in bats if b["order"] % 100 == 0][:9],
                     "bats": bats}
    out["win"] = 1 if out["home"]["runs"] > out["away"]["runs"] else 0
    return out


def fetch_people(ids):
    have = jload("people.json", {})
    todo = [i for i in ids if i and str(i) not in have]
    print(f"players: {len(have)} cached, {len(todo)} to fetch")
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        try:
            for p in get(f"{API}/people", {"personIds": ",".join(map(str, chunk))}).get("people", []):
                have[str(p["id"])] = {"bat": (p.get("batSide") or {}).get("code", "R"),
                                      "pitch": (p.get("pitchHand") or {}).get("code", "R"),
                                      "pos": ((p.get("primaryPosition") or {}).get("abbreviation") or ""),
                                      "name": p.get("fullName", "")}
        except Exception as e:
            print("  people batch failed:", e)
        time.sleep(0.1)
    jsave("people.json", have); return have


def fetch_venues(ids):
    have = jload("venues.json", {})
    for v in ids:
        if not v or str(v) in have: continue
        try:
            d = get(f"{API}/venues/{v}", {"hydrate": "location,fieldInfo,timezone"})["venues"][0]
            loc = d.get("location", {}); co = loc.get("defaultCoordinates", {})
            name = d.get("name", "")
            have[str(v)] = {"name": name, "lat": co.get("latitude"), "lon": co.get("longitude"),
                            "tz": (d.get("timeZone") or {}).get("offset", -5),
                            "roof": ((d.get("fieldInfo") or {}).get("roofType") or "Open"),
                            "bearing": PARK_GEO.get(name, (0, 0))[0], "elev": PARK_GEO.get(name, (0, 0))[1]}
        except Exception as e:
            print(f"  venue {v} failed: {e}")
            have[str(v)] = {"name": "", "lat": None, "lon": None, "tz": -5, "roof": "Open", "bearing": 0, "elev": 0}
        time.sleep(0.1)
    jsave("venues.json", have); return have


def fetch_weather(games, venues):
    have = jload("weather.json", {})
    by_vs = defaultdict(list)
    for g in games:
        if str(g["pk"]) in have: continue
        v = venues.get(str(g["venue"]), {})
        if not v.get("lat") or "dome" in str(v.get("roof", "Open")).lower():
            have[str(g["pk"])] = {"temp": 72.0, "wind": 0.0, "dir": 0.0, "hum": 50.0, "pres": 1013.0, "indoor": True}
            continue
        by_vs[(g["venue"], g["date"][:4])].append(g)
    if by_vs: print(f"weather: {len(by_vs)} venue-seasons to fetch")
    for (vid, yr), gs in by_vs.items():
        v = venues[str(vid)]; d0 = min(g["date"] for g in gs); d1 = max(g["date"] for g in gs)
        try:
            w = get("https://archive-api.open-meteo.com/v1/archive",
                    {"latitude": v["lat"], "longitude": v["lon"], "start_date": d0, "end_date": d1,
                     "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m,surface_pressure",
                     "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC"})
            H = w["hourly"]; hrs = H["time"]
            arrs = {k: H.get(k) or [None] * len(hrs) for k in
                    ("temperature_2m", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m", "surface_pressure")}
            idx = {h: i for i, h in enumerate(hrs)}
            for g in gs:
                key = datetime.fromtimestamp(g["ts"], tz=timezone.utc).strftime("%Y-%m-%dT%H:00")
                i = idx.get(key)
                pick = lambda k, d: (float(arrs[k][i]) if (i is not None and arrs[k][i] is not None) else d)
                have[str(g["pk"])] = {"temp": pick("temperature_2m", 72.0), "wind": pick("wind_speed_10m", 5.0),
                                      "dir": pick("wind_direction_10m", 0.0), "hum": pick("relative_humidity_2m", 50.0),
                                      "pres": pick("surface_pressure", 1013.0), "indoor": False}
        except Exception as e:
            print(f"  weather {vid} {yr} failed: {e}")
            for g in gs:
                have[str(g["pk"])] = {"temp": 72.0, "wind": 5.0, "dir": 0.0, "hum": 50.0, "pres": 1013.0, "indoor": False}
        time.sleep(0.3)
    jsave("weather.json", have); return have


def fetch_upcoming(days=8):
    """Scheduled games from today forward, with the probable starters."""
    d0 = datetime.now(timezone.utc).date() - timedelta(days=1)
    d1 = d0 + timedelta(days=days + 1)
    try:
        sched = get(f"{API}/schedule", {"sportId": 1, "gameType": "R,F,D,L,W",
                                        "hydrate": "probablePitcher,venue,team",
                                        "startDate": d0.isoformat(), "endDate": d1.isoformat()})
    except Exception as e:
        print("upcoming failed:", e); return []
    out = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            state = (g.get("status", {}) or {}).get("abstractGameState", "")
            if state == "Final": continue
            pp = lambda s: ((g["teams"][s].get("probablePitcher") or {}) or {})
            out.append({"pk": g["gamePk"], "date": g["gameDate"][:10],
                        "ts": int(datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")).timestamp()),
                        "venue": (g.get("venue") or {}).get("id"),
                        "venue_name": (g.get("venue") or {}).get("name", ""),
                        "night": g.get("dayNight") == "night", "dh": g.get("doubleHeader", "N") != "N",
                        "series_game": g.get("seriesGameNumber") or 1,
                        "type": g.get("gameType", "R"), "state": state,
                        "home": {"id": g["teams"]["home"]["team"]["id"], "name": g["teams"]["home"]["team"]["name"],
                                 "starter": pp("home").get("id")},
                        "away": {"id": g["teams"]["away"]["team"]["id"], "name": g["teams"]["away"]["team"]["name"],
                                 "starter": pp("away").get("id")}})
    print(f"upcoming: {len(out)} scheduled games in the next {days} days")
    return out


# ============================================================== helpers
def haversine(a, b):
    if not a or not b or a[0] is None or b[0] is None: return 0.0
    R = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1; dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def elo_exp(a, b): return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


# the ordered feature list. Python builds it here and the app rebuilds the identical
# vector in JavaScript, which is what lets you swap a starter or a hitter in the app
# and get a real number back instead of a cached one.
FEATURES = ["elo", "sp", "lineup", "pen", "def", "sp_rest_h", "sp_rest_a", "pen_fatigue",
            "platoon", "form10", "form30", "sos", "winpct", "rest", "travel", "tz",
            "park", "temp", "wind_out", "air", "night", "dh", "series", "h2h",
            "bat_vs_sp", "sp_vs_team", "missing"]


def build_vector(d):
    """d is a flat dict of scalars; both Python and the app call this with the same keys."""
    return [
        (d["elo_h"] - d["elo_a"]) / 100.0,
        (d["sp_h"] - d["sp_a"]) / 100.0,
        (d["lu_h"] - d["lu_a"]) / 100.0,
        (d["pen_h"] - d["pen_a"]) / 100.0,
        (d["def_h"] - d["def_a"]) / 100.0,
        min(d["sp_rest_h"], 8.0) / 5.0,
        min(d["sp_rest_a"], 8.0) / 5.0,
        (d["fat_h"] - d["fat_a"]) / 9.0,
        (d["plat_h"] - d["plat_a"]) / 9.0,
        d["form10_h"] - d["form10_a"],
        d["form30_h"] - d["form30_a"],
        (d["sos_h"] - d["sos_a"]) / 100.0,
        d["wp_h"] - d["wp_a"],
        (min(d["rest_h"], 4.0) - min(d["rest_a"], 4.0)) / 4.0,
        min(d["travel_a"], 4000.0) / 4000.0,
        abs(d["tz_a"]) / 3.0,
        d["park"] - 1.0,
        (d["temp"] - 72.0) / 20.0,
        d["wind_out"] / 15.0,
        d["air"] - 1.0,
        1.0 if d["night"] else 0.0,
        1.0 if d["dh"] else 0.0,
        min(d["series"], 4.0) / 4.0,
        d["h2h"],
        d["bat_vs_sp_h"] - d["bat_vs_sp_a"],
        d["sp_vs_team_h"] - d["sp_vs_team_a"],
        d["miss_a"] - d["miss_h"],
    ]


# ============================================================== the rating state
class State:
    def __init__(self, people, venues, weather):
        self.people = people; self.venues = venues; self.weather = weather
        self.elo = defaultdict(lambda: BASE)          # team
        self.sp = defaultdict(lambda: BASE)           # starting pitcher
        self.bat = defaultdict(lambda: BASE)          # hitter
        self.pen = defaultdict(lambda: BASE)          # bullpen, by team
        self.defn = defaultdict(lambda: BASE)         # run prevention, by team
        self.form = defaultdict(lambda: deque(maxlen=FORM_N2))
        self.sos = defaultdict(lambda: deque(maxlen=SOS_N))
        self.rec = defaultdict(lambda: [0, 0])        # season win/loss
        self.last_ts = {}
        self.last_venue = {}
        self.sp_last = {}                             # pitcher -> ts of last start
        self.sp_starts = defaultdict(int)
        self.sp_outs = defaultdict(int)
        self.sp_k = defaultdict(int); self.sp_bb = defaultdict(int); self.sp_bf = defaultdict(int)
        self.bat_pa = defaultdict(int)
        self.bat_woba = defaultdict(lambda: [0.0, 0])  # num, den
        self.relief = defaultdict(list)               # team -> [(ts, outs)]
        self.h2h = defaultdict(lambda: [0, 0])        # (a,b) -> [wins_a, games]
        self.bvp = defaultdict(lambda: [0.0, 0])      # (batter, pitcher) -> woba num, den
        self.pvt = defaultdict(lambda: [0, 0])        # (pitcher, opp team) -> outs, runs
        self.park = defaultdict(lambda: [0.0, 0])     # venue -> runs, games
        self.lg_w = deque(maxlen=4000)                # rolling league wOBA, keeps hitter Elo from drifting
        self.team_players = defaultdict(lambda: defaultdict(lambda: {"g": 0, "last": 0, "order": 0}))
        self.team_pitchers = defaultdict(lambda: defaultdict(lambda: {"g": 0, "last": 0}))
        self.lg_runs = deque(maxlen=2000)
        self.season = None
        self.n_games = 0

    # ---------- read-only accessors used by both features() and the export
    def lg_rpg(self): return (sum(self.lg_runs) / len(self.lg_runs)) if self.lg_runs else 4.5

    def lg_woba(self):
        """the actual league wOBA in this era, not a constant. A fixed 0.315 makes every hitter
        gain or lose every single game and the whole hitter pool drifts away from 1500."""
        return (sum(self.lg_w) / len(self.lg_w)) if len(self.lg_w) >= 200 else LG_WOBA

    def park_factor(self, vid):
        r, n = self.park[vid]
        lg = self.lg_rpg() * 2.0
        if n < 20 or lg <= 0: return 1.0
        return max(0.80, min(1.25, ((r + 30 * lg) / (n + 30)) / lg))

    def form_of(self, t, n):
        f = list(self.form[t])[-n:]
        return (sum(f) / len(f)) if f else 0.5

    def sos_of(self, t):
        s = self.sos[t]
        return (sum(s) / len(s)) if s else BASE

    def winpct(self, t):
        w, l = self.rec[t]
        return (w + 10 * 0.5) / (w + l + 10)

    def rest(self, t, ts):
        return min((ts - self.last_ts.get(t, ts - 86400)) / 86400.0, 6.0)

    def sp_rest(self, pid, ts):
        if not pid or pid not in self.sp_last: return 5.0
        return min((ts - self.sp_last[pid]) / 86400.0, 12.0)

    def fatigue(self, t, ts):
        return sum(o for (tt, o) in self.relief[t] if 0 <= ts - tt <= 3 * 86400) / 3.0

    def travel(self, t, vid):
        prev = self.last_venue.get(t)
        if not prev or prev == vid: return 0.0
        a = self.venues.get(str(prev), {}); b = self.venues.get(str(vid), {})
        return haversine((a.get("lat"), a.get("lon")), (b.get("lat"), b.get("lon")))

    def tzshift(self, t, vid):
        prev = self.last_venue.get(t)
        if not prev: return 0.0
        a = self.venues.get(str(prev), {}); b = self.venues.get(str(vid), {})
        try: return float(b.get("tz", -5)) - float(a.get("tz", -5))
        except Exception: return 0.0

    def hand(self, pid): return (self.people.get(str(pid), {}) or {}).get("pitch", "R")

    def batside(self, pid): return (self.people.get(str(pid), {}) or {}).get("bat", "R")

    def lineup_elo(self, lineup):
        """batting-order weighted average; the top of the order gets more plate appearances."""
        if not lineup: return BASE
        w = [1.16, 1.12, 1.09, 1.05, 1.00, 0.95, 0.91, 0.87, 0.85]
        tot = 0.0; wt = 0.0
        for i, b in enumerate(lineup[:9]):
            k = w[i] if i < 9 else 1.0
            tot += k * self.bat[b]; wt += k
        return tot / wt if wt else BASE

    def platoon(self, lineup, opp_hand):
        """how many of the nine have the handedness edge against the man on the mound."""
        n = 0
        for b in (lineup or [])[:9]:
            s = self.batside(b)
            if s == "S": n += 1
            elif opp_hand == "L" and s == "R": n += 1
            elif opp_hand == "R" and s == "L": n += 1
        return float(n)

    def bat_vs_sp(self, lineup, pid):
        """the posted lineup's own history against this pitcher, shrunk hard toward league."""
        if not pid: return 0.0
        num = 0.0; den = 0
        for b in (lineup or [])[:9]:
            n, d = self.bvp[(b, pid)]
            num += n; den += d
        if den < 25: return 0.0
        w = min(den / 150.0, 1.0)
        return w * ((num / den) - self.lg_woba()) / 0.05

    def sp_vs_team(self, pid, opp):
        """this pitcher's career record against this specific opponent."""
        if not pid: return 0.0
        outs, runs = self.pvt[(pid, opp)]
        if outs < 45: return 0.0
        ra9 = runs * 27.0 / outs
        w = min(outs / 200.0, 1.0)
        return w * (self.lg_rpg() - ra9) / 2.0

    def regulars(self, t):
        """the nine who play most. Sorted with an explicit tie-break on id so that the app,
        sorting the same list in JavaScript, lands on exactly the same nine."""
        return [p for p, _ in sorted(self.team_players[t].items(),
                                     key=lambda kv: (-kv[1]["g"], kv[0]))[:9]]

    def missing(self, t, lineup):
        """regulars not in today's lineup, as a share of the nine."""
        reg = set(self.regulars(t))
        if not reg: return 0.0
        have = set((lineup or [])[:9])
        return len(reg - have) / 9.0

    def h2h_of(self, h, a):
        w, n = self.h2h[(h, a)]
        if n < 4: return 0.0
        return (w / n - 0.5) * min(n / 12.0, 1.0)

    # ---------- the feature dict for one game
    def context(self, g, sp_h, sp_a, lu_h, lu_a):
        v = self.venues.get(str(g["venue"]), {})
        w = self.weather.get(str(g["pk"]), {"temp": 72.0, "wind": 5.0, "dir": 0.0, "hum": 50.0,
                                            "pres": 1013.0, "indoor": True})
        h = g["home"]["id"]; a = g["away"]["id"]
        # wind component along the home-plate -> centre-field line: positive means blowing out
        bearing = float(v.get("bearing", 0) or 0)
        if w.get("indoor"):
            wind_out = 0.0
        else:
            wind_out = float(w.get("wind", 0)) * math.cos(math.radians(float(w.get("dir", 0)) - bearing + 180.0))
        elev = float(v.get("elev", 0) or 0)
        air = (float(w.get("pres", 1013.0)) / 1013.0) * (288.0 / (273.0 + (float(w.get("temp", 72.0)) - 32.0) * 5.0 / 9.0)) \
            * math.exp(-elev / 8500.0)
        return {
            "elo_h": self.elo[h], "elo_a": self.elo[a],
            "sp_h": self.sp[sp_h] if sp_h else BASE, "sp_a": self.sp[sp_a] if sp_a else BASE,
            "lu_h": self.lineup_elo(lu_h), "lu_a": self.lineup_elo(lu_a),
            "pen_h": self.pen[h], "pen_a": self.pen[a],
            "def_h": self.defn[h], "def_a": self.defn[a],
            "sp_rest_h": self.sp_rest(sp_h, g["ts"]), "sp_rest_a": self.sp_rest(sp_a, g["ts"]),
            "fat_h": self.fatigue(h, g["ts"]), "fat_a": self.fatigue(a, g["ts"]),
            "plat_h": self.platoon(lu_h, self.hand(sp_a)), "plat_a": self.platoon(lu_a, self.hand(sp_h)),
            "form10_h": self.form_of(h, FORM_N), "form10_a": self.form_of(a, FORM_N),
            "form30_h": self.form_of(h, FORM_N2), "form30_a": self.form_of(a, FORM_N2),
            "sos_h": self.sos_of(h), "sos_a": self.sos_of(a),
            "wp_h": self.winpct(h), "wp_a": self.winpct(a),
            "rest_h": self.rest(h, g["ts"]), "rest_a": self.rest(a, g["ts"]),
            "travel_a": self.travel(a, g["venue"]), "tz_a": self.tzshift(a, g["venue"]),
            "park": self.park_factor(g["venue"]),
            "temp": float(w.get("temp", 72.0)), "wind_out": wind_out, "air": air,
            "night": bool(g.get("night")), "dh": bool(g.get("dh")), "series": float(g.get("series_game") or 1),
            "h2h": self.h2h_of(h, a),
            "bat_vs_sp_h": self.bat_vs_sp(lu_h, sp_a), "bat_vs_sp_a": self.bat_vs_sp(lu_a, sp_h),
            "sp_vs_team_h": self.sp_vs_team(sp_h, a), "sp_vs_team_a": self.sp_vs_team(sp_a, h),
            "miss_h": self.missing(h, lu_h), "miss_a": self.missing(a, lu_a),
        }

    def recentre(self):
        """Elo is only meaningful relative to the pool. These updates are not strictly zero-sum
        (a hitter can beat expectation without the pitcher being charged the same amount), so
        each pool is pulled back so its active members average 1500. Without this the hitter
        ratings climb every season and a 1700 hitter in 2019 is not a 1700 hitter in 2026."""
        for d, active in ((self.bat, [b for b in self.bat_pa if self.bat_pa[b] >= 100]),
                          (self.sp, [p for p in self.sp_starts if self.sp_starts[p] >= 3]),
                          (self.pen, list(self.pen.keys())), (self.defn, list(self.defn.keys())),
                          (self.elo, list(self.elo.keys()))):
            if len(active) < 10: continue
            off = sum(d[k] for k in active) / len(active) - BASE
            if abs(off) < 0.5: continue
            for k in list(d.keys()): d[k] -= off

    # ---------- carry ratings into a new season
    def new_season(self, season):
        if self.season is None:
            self.season = season; return
        if season == self.season: return
        self.recentre()
        self.season = season
        for d, c in ((self.elo, CARRY), (self.pen, CARRY), (self.defn, CARRY),
                     (self.sp, CARRY_P), (self.bat, CARRY_P)):
            for k in list(d.keys()):
                d[k] = BASE + (d[k] - BASE) * c
        for t in list(self.rec.keys()): self.rec[t] = [0, 0]
        self.form.clear(); self.relief.clear()

    # ---------- apply one finished game
    def update(self, g):
        h = g["home"]["id"]; a = g["away"]["id"]
        rh = g["home"]["runs"]; ra = g["away"]["runs"]
        win = g["win"]
        lg = self.lg_rpg()

        # --- team Elo, margin scaled the way the esports models scale by round or game length
        exp = elo_exp(self.elo[h] + HOME_ELO, self.elo[a])
        diff = abs(rh - ra)
        mov = math.log(1.0 + diff) * (2.2 / (abs(self.elo[h] - self.elo[a]) * 0.001 + 2.2))
        d = K_TEAM * mov * (win - exp)
        self.elo[h] += d; self.elo[a] -= d

        # --- starting pitchers
        for side, opp, runs_allowed in (("home", a, ra), ("away", h, rh)):
            pl = g[side]["pitchers"]
            if not pl: continue
            sp = pl[0]
            pid = sp["id"]; outs = sp["outs"]
            if outs >= 6:
                opp_off = self.lineup_elo(g["away" if side == "home" else "home"]["lineup"])
                # what we expected this pitcher to give up, before the game
                exp_ra9 = lg * (10 ** (-(self.sp[pid] - BASE) / 900.0)) * (10 ** ((opp_off - BASE) / 1400.0)) \
                    / max(0.80, min(1.25, self.park_factor(g["venue"])))
                act_ra9 = sp["r"] * 27.0 / outs
                perf = max(-1.2, min(1.2, (exp_ra9 - act_ra9) / max(1.5, exp_ra9)))
                self.sp[pid] += K_SP * perf * min(outs / 18.0, 1.3)
                self.sp_last[pid] = g["ts"]
                self.sp_starts[pid] += 1; self.sp_outs[pid] += outs
                self.sp_k[pid] += sp["k"]; self.sp_bb[pid] += sp["bb"]; self.sp_bf[pid] += sp["bf"]
                self.team_pitchers[g[side]["id"]][pid]["g"] += 1
                self.team_pitchers[g[side]["id"]][pid]["last"] = g["ts"]
                self.pvt[(pid, opp)][0] += outs; self.pvt[(pid, opp)][1] += sp["r"]
            # --- bullpen: everything after the starter
            ro = sum(p["outs"] for p in pl[1:]); rr = sum(p["r"] for p in pl[1:])
            t = g[side]["id"]
            if ro >= 3:
                opp_off0 = self.lineup_elo(g["away" if side == "home" else "home"]["lineup"])
                exp_pen = lg * (10 ** (-(self.pen[t] - BASE) / 900.0)) * (10 ** ((opp_off0 - BASE) / 1400.0))
                act = rr * 27.0 / ro
                perf = max(-1.2, min(1.2, (exp_pen - act) / max(1.5, exp_pen)))
                self.pen[t] += K_PEN * perf * min(ro / 9.0, 1.2)
            if ro: self.relief[t].append((g["ts"], ro))
            # --- team run prevention overall. The expectation has to carry the quality of the
            # --- offence faced, or a team that drew weak lineups all month rates as a great defence.
            opp_off = self.lineup_elo(g["away" if side == "home" else "home"]["lineup"])
            off_adj = 10 ** ((opp_off - BASE) / 1400.0)
            exp_def = lg * (10 ** (-(self.defn[t] - BASE) / 900.0)) * off_adj
            perf = max(-1.2, min(1.2, (exp_def - runs_allowed) / max(1.5, exp_def)))
            self.defn[t] += K_DEF * perf

        # --- hitters
        for side in ("home", "away"):
            opp_pl = g["away" if side == "home" else "home"]["pitchers"]
            opp_sp = opp_pl[0]["id"] if opp_pl else None
            opp_sp_elo = self.sp[opp_sp] if opp_sp else BASE
            for b in g[side]["bats"]:
                num, den = woba_of(b)
                if den <= 0: continue
                bid = b["id"]
                lgw = self.lg_woba()
                exp_w = lgw * (1.0 - (opp_sp_elo - BASE) / 3500.0) * (1.0 + (self.bat[bid] - BASE) / 2600.0)
                perf = max(-1.2, min(1.2, ((num / den) - exp_w) / 0.16))
                self.lg_w.append(num / den)
                self.bat[bid] += K_BAT * perf * min(den / 4.0, 1.3)
                self.bat_pa[bid] += den
                self.bat_woba[bid][0] += num; self.bat_woba[bid][1] += den
                if opp_sp:
                    self.bvp[(bid, opp_sp)][0] += num; self.bvp[(bid, opp_sp)][1] += den
                tp = self.team_players[g[side]["id"]][bid]
                tp["g"] += 1; tp["last"] = g["ts"]
                if b["order"] % 100 == 0: tp["order"] = b["order"] // 100

        # --- bookkeeping
        self.form[h].append(1.0 if win else 0.0); self.form[a].append(0.0 if win else 1.0)
        self.sos[h].append(self.elo[a]); self.sos[a].append(self.elo[h])
        self.rec[h][0 if win else 1] += 1; self.rec[a][1 if win else 0] += 1
        self.h2h[(h, a)][0] += win; self.h2h[(h, a)][1] += 1
        self.h2h[(a, h)][0] += (1 - win); self.h2h[(a, h)][1] += 1
        self.last_ts[h] = g["ts"]; self.last_ts[a] = g["ts"]
        self.last_venue[h] = g["venue"]; self.last_venue[a] = g["venue"]
        self.park[g["venue"]][0] += rh + ra; self.park[g["venue"]][1] += 1
        self.lg_runs.append((rh + ra) / 2.0)
        self.n_games += 1
        if self.n_games % 750 == 0: self.recentre()


# ============================================================== logistic
def train_logistic(X, y, l2=1.0, iters=40):
    """Ridge logistic by IRLS (Newton). With a couple of dozen features this solves exactly in
    a handful of steps, so there is no learning rate to guess wrong and no half-trained fit."""
    mu = X.mean(0); sd = X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Z = (X - mu) / sd
    n, d = Z.shape
    A = np.column_stack([Z, np.ones(n)])          # intercept as the last column
    pen = np.eye(d + 1) * l2; pen[d, d] = 0.0     # never penalise the intercept
    beta = np.zeros(d + 1)
    prev = None
    for _ in range(iters):
        eta = np.clip(A @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-6, None)
        g = A.T @ (y - p) - pen @ beta
        H = (A.T * W) @ A + pen
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta + step
        ll = float(np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
        if prev is not None and abs(ll - prev) < 1e-9: break
        prev = ll
    return {"w": beta[:d], "b": float(beta[d]), "mu": mu, "sd": sd}


def predict_logistic(m, X):
    return 1.0 / (1.0 + np.exp(-(((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"])))


def choose_l2(X, y, grid=(1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000), blocks=5):
    """Pick the ridge strength by out-of-fold log loss, in time order. We bet the probability,
    not the argmax, so log loss is the thing to minimise - a model can gain accuracy while
    getting worse at the only number that decides stake size."""
    n = len(X); best = (None, 1e9)
    for l2 in grid:
        tot = 0.0; cnt = 0
        for bi in range(1, blocks):                # always train on the past, test on the future
            cut = n * bi // blocks; hi = n * (bi + 1) // blocks
            m = train_logistic(X[:cut], y[:cut], l2=l2)
            p = np.clip(predict_logistic(m, X[cut:hi]), 1e-9, 1 - 1e-9)
            yy = y[cut:hi]
            tot += float(-np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p))); cnt += len(yy)
        ll = tot / max(cnt, 1)
        print(f"    l2={l2:<6} out-of-fold log loss {ll:.4f}")
        if ll < best[1]: best = (l2, ll)
    print(f"  chose l2={best[0]}")
    if best[0] == grid[-1]:
        print("  WARNING: that is the largest value tried, so the real optimum is probably higher still.")
        print("  Heavy shrinkage means the features carry little signal beyond the base rate - read the")
        print("  holdout numbers with that in mind rather than trusting the probabilities' spread.")
    return best[0]


def fit_temperature(X, y, l2, blocks=5):
    """One parameter, fitted only on predictions the model never trained on: p' = sigmoid(t * logit(p)).
    t < 1 pulls extreme calls back toward the middle. A model that says 87% and wins 77% is not
    slightly wrong, it is wrong in the direction that empties a bankroll fastest, because that is
    exactly where Kelly tells you to bet the most."""
    n = len(X); L = []; Y = []
    for bi in range(1, blocks):
        cut = n * bi // blocks; hi = n * (bi + 1) // blocks
        m = train_logistic(X[:cut], y[:cut], l2=l2)
        p = np.clip(predict_logistic(m, X[cut:hi]), 1e-6, 1 - 1e-6)
        L.append(np.log(p / (1 - p))); Y.append(y[cut:hi])
    L = np.concatenate(L); Y = np.concatenate(Y)
    best = (1.0, 1e9)
    for t in np.arange(0.40, 1.41, 0.02):
        q = np.clip(1.0 / (1.0 + np.exp(-t * L)), 1e-9, 1 - 1e-9)
        ll = float(-np.mean(Y * np.log(q) + (1 - Y) * np.log(1 - q)))
        if ll < best[1]: best = (float(t), ll)
    print(f"  calibration temperature {best[0]:.2f} (1.00 = already calibrated, "
          f"lower = the raw model is overconfident)")
    return best[0]


def calibration(p, y, bins=8):
    """Does a 60% prediction actually win 60% of the time? If not, every stake is wrong."""
    print("  calibration: predicted -> actual (n)")
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if m.sum() < 10: continue
        print(f"    {p[m].mean()*100:5.1f}% -> {y[m].mean()*100:5.1f}%  ({int(m.sum())})")


def evaluate(name, p, y):
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    ll = float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))
    br = float(np.mean((p - y) ** 2))
    print(f"  {name:<28} accuracy {acc*100:5.1f}%   log loss {ll:.4f}   Brier {br:.4f}")
    return acc, ll, br


# ============================================================== market
def load_market(path):
    """CSV with columns: date,home,away,home_price,away_price  (decimal odds).
    Optional. Without it the app falls back to a fixed 35/65 model/market blend."""
    if not path or not os.path.exists(path): return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["date"][:10], row["home"].strip().lower(), row["away"].strip().lower())
                out[key] = (float(row["home_price"]), float(row["away_price"]))
            except Exception:
                continue
    print(f"market: {len(out)} priced games loaded from {path}")
    return out


def devig(ph, pa):
    ih, ia = 1.0 / ph, 1.0 / pa
    s = ih + ia
    return ih / s if s > 0 else 0.5


# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=[2019, 2021, 2022, 2023, 2024, 2025, 2026],
                    help="2020 is left out by default: 60 games, empty parks, no travel")
    ap.add_argument("--market", default="", help="optional CSV of closing moneyline prices")
    ap.add_argument("--days", type=int, default=8, help="how far ahead to list scheduled games")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # ---------------- fetch
    games = []
    for s in sorted(args.seasons):
        games.extend(fetch_season(s))
    games.sort(key=lambda g: g["ts"])
    if len(games) < 500:
        sys.exit("not enough games fetched - check your connection and try again")
    print(f"\n{len(games)} completed games from {games[0]['date']} to {games[-1]['date']}")

    pids = set()
    for g in games:
        for side in ("home", "away"):
            for p in g[side]["pitchers"]: pids.add(p["id"])
            for b in g[side]["bats"]: pids.add(b["id"])
    upcoming = fetch_upcoming(args.days)
    for u in upcoming:
        for side in ("home", "away"):
            if u[side].get("starter"): pids.add(u[side]["starter"])

    people = fetch_people(sorted(pids))
    venues = fetch_venues(sorted({g["venue"] for g in games} | {u["venue"] for u in upcoming}))
    weather = fetch_weather(games, venues)
    market = load_market(args.market)

    # ---------------- walk history forward, building rows as we go
    st = State(people, venues, weather)
    rows = []
    for g in games:
        st.new_season(g["season"])
        sp_h = g["home"]["pitchers"][0]["id"] if g["home"]["pitchers"] else None
        sp_a = g["away"]["pitchers"][0]["id"] if g["away"]["pitchers"] else None
        if sp_h and sp_a and g["home"]["lineup"] and g["away"]["lineup"]:
            ctx = st.context(g, sp_h, sp_a, g["home"]["lineup"], g["away"]["lineup"])
            rows.append({"x": build_vector(ctx), "y": g["win"], "ts": g["ts"], "date": g["date"],
                         "home": g["home"]["name"], "away": g["away"]["name"]})
        st.update(g)

    X = np.array([r["x"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)
    print(f"\n{len(rows)} usable rows, {len(FEATURES)} features. Home teams won {y.mean()*100:.1f}%.")

    cut = int(len(rows) * (1 - TEST_FRACTION))
    print("\nchoosing regularisation on the training years only:")
    l2 = choose_l2(X[:cut], y[:cut])
    temp = fit_temperature(X[:cut], y[:cut], l2)
    model = train_logistic(X[:cut], y[:cut], l2=l2)
    print("\nHOLDOUT (the most recent %d games, never seen in training)" % (len(rows) - cut))
    raw_test = predict_logistic(model, X[cut:])
    p_test = np.clip(1.0 / (1.0 + np.exp(-temp * np.log(np.clip(raw_test, 1e-9, 1 - 1e-9)
                                                        / (1 - np.clip(raw_test, 1e-9, 1 - 1e-9))))), 1e-9, 1 - 1e-9)
    evaluate("model, raw", raw_test, y[cut:])
    acc, ll, br = evaluate("model, calibrated", p_test, y[cut:])
    evaluate("home team always", np.full(len(y) - cut, y[:cut].mean()), y[cut:])
    calibration(p_test, y[cut:])

    # feature weights, biggest first
    print("\nWHAT THE MODEL LEANS ON (standardised weight, + favours the home team)")
    order = np.argsort(-np.abs(model["w"]))
    for i in order:
        print(f"  {FEATURES[i]:<14} {model['w'][i]:+.3f}")

    # ---------------- market stack, fitted out of fold so it is honest
    stack = None
    if market:
        mp, mx, my = [], [], []
        for r in rows:
            k = (r["date"], r["home"].strip().lower(), r["away"].strip().lower())
            if k in market:
                ph, pa = market[k]
                mp.append(devig(ph, pa)); mx.append(r["x"]); my.append(r["y"])
        if len(mp) > 400:
            MX = np.array(mx); MY = np.array(my); MP = np.array(mp)
            blocks = 6; n = len(MX); oof = np.zeros(n)
            for bi in range(blocks):
                lo, hi = n * bi // blocks, n * (bi + 1) // blocks
                idx = np.ones(n, bool); idx[lo:hi] = False
                m = train_logistic(MX[idx], MY[idx])
                oof[lo:hi] = predict_logistic(m, MX[lo:hi])
            Z = np.column_stack([np.array([logit(v) for v in oof]), np.array([logit(v) for v in MP])])
            sm = train_logistic(Z, MY, l2=0.5)
            a_, b_ = sm["w"] / sm["sd"]
            c_ = float(sm["b"] - (sm["w"] / sm["sd"] * sm["mu"]).sum())
            stack = {"a": float(a_), "b": float(b_), "c": c_}
            print(f"\nSTACKED model+market on {len(mp)} priced games: "
                  f"{stack['a']:.2f}*logit(model) + {stack['b']:.2f}*logit(market) + {stack['c']:.2f}")
            evaluate("market alone", MP, MY)
            evaluate("stacked", np.array([sigmoid(stack['a'] * logit(o) + stack['b'] * logit(m) + stack['c'])
                                          for o, m in zip(oof, MP)]), MY)
        else:
            print("\nnot enough priced games to fit a stack - the app will use the default 35/65 blend")
    else:
        print("\nno stack yet - pass --market odds_mlb.csv once you have closing prices logged")

    # ---------------- export
    print("\nbuilding model.json ...")
    # ranks
    bat_ids = [b for b in st.bat_pa if st.bat_pa[b] >= 150]
    sp_ids = [p for p in st.sp_starts if st.sp_starts[p] >= 5]
    bat_rank = {b: i + 1 for i, b in enumerate(sorted(bat_ids, key=lambda b: -st.bat[b]))}
    sp_rank = {p: i + 1 for i, p in enumerate(sorted(sp_ids, key=lambda p: -st.sp[p]))}

    teams = {}
    for t in st.elo:
        nm = ""
        for g in reversed(games):
            if g["home"]["id"] == t: nm = g["home"]["name"]; break
            if g["away"]["id"] == t: nm = g["away"]["name"]; break
        pool = sorted(st.team_players[t].items(), key=lambda kv: (-kv[1]["last"], -kv[1]["g"]))[:24]
        pitchers = sorted(st.team_pitchers[t].items(), key=lambda kv: (-kv[1]["last"], -kv[1]["g"]))[:10]
        teams[str(t)] = {
            "name": nm, "elo": round(st.elo[t], 1), "pen": round(st.pen[t], 1), "def": round(st.defn[t], 1),
            "form10": round(st.form_of(t, FORM_N), 3), "form30": round(st.form_of(t, FORM_N2), 3),
            "sos": round(st.sos_of(t), 1), "winpct": round(st.winpct(t), 3),
            "rec": st.rec[t],
            "pool": [{"id": p, "g": v["g"], "order": v["order"]} for p, v in pool],
            "regulars": st.regulars(t),
            "rot": [{"id": p, "g": v["g"]} for p, v in pitchers],
        }

    players = {}
    for b in bat_ids:
        num, den = st.bat_woba[b]
        players[str(b)] = {"t": "bat", "elo": round(st.bat[b], 1), "rank": bat_rank[b],
                           "name": people.get(str(b), {}).get("name", str(b)),
                           "hand": st.batside(b), "pa": st.bat_pa[b],
                           "woba": round(num / den, 3) if den else None}
    for p in sp_ids:
        bf = max(st.sp_bf[p], 1)
        players[str(p)] = {"t": "sp", "elo": round(st.sp[p], 1), "rank": sp_rank[p],
                           "name": people.get(str(p), {}).get("name", str(p)),
                           "hand": st.hand(p), "starts": st.sp_starts[p],
                           "ip": round(st.sp_outs[p] / 3.0, 1),
                           "k_pct": round(st.sp_k[p] / bf, 3), "bb_pct": round(st.sp_bb[p] / bf, 3),
                           "ipgs": round(st.sp_outs[p] / 3.0 / max(st.sp_starts[p], 1), 1),
                           "last": st.sp_last.get(p, 0)}   # lets the app work out rest for a swapped-in starter

    # upcoming games, with the context the app needs to recompute after you edit a lineup
    ups = []
    for u in upcoming:
        h = u["home"]["id"]; a = u["away"]["id"]
        if h not in st.elo or a not in st.elo: continue
        # expected lineup = the nine who most recently batted in an order slot for that team
        def expect(t):
            pool = [(p, v) for p, v in st.team_players[t].items() if v["order"]]
            pool.sort(key=lambda kv: (-kv[1]["last"], kv[1]["order"] or 9))
            seen = []; slots = {}
            for p, v in pool:
                o = v["order"]
                if o and o not in slots and p not in seen:
                    slots[o] = p; seen.append(p)
                if len(slots) == 9: break
            return [slots[o] for o in sorted(slots)]
        lu_h = expect(h); lu_a = expect(a)
        sp_h = u["home"].get("starter"); sp_a = u["away"].get("starter")
        gg = dict(u); gg["home"] = {"id": h}; gg["away"] = {"id": a}
        # weather for a future game is left to the app, which pulls the forecast live
        st.weather[str(u["pk"])] = st.weather.get(str(u["pk"]),
                                                  {"temp": 72.0, "wind": 5.0, "dir": 0.0, "hum": 50.0,
                                                   "pres": 1013.0,
                                                   "indoor": "dome" in str(venues.get(str(u["venue"]), {}).get("roof", "")).lower()})
        ctx = st.context(gg, sp_h, sp_a, lu_h, lu_a)

        # Everything the app needs to recompute after you edit a lineup or swap a starter.
        # bvp: how each of this team's available hitters has actually done against the man
        # scheduled to start for the other side. spvt: how each arm in this team's rotation has
        # done against this particular opponent.
        def bvp_map(t, opp_sp):
            if not opp_sp: return {}
            m = {}
            for p, _ in st.team_players[t].items():
                num, den = st.bvp[(p, opp_sp)]
                if den > 0: m[str(p)] = [round(num, 3), den]
            return m

        def spvt_map(t, opp):
            m = {}
            for p, _ in st.team_pitchers[t].items():
                v = st.sp_vs_team(p, opp)
                if v: m[str(p)] = round(v, 4)
            return m

        ups.append({
            "pk": u["pk"], "ts": u["ts"], "date": u["date"], "state": u["state"],
            "venue": u["venue"], "venue_name": u["venue_name"],
            "home": {"id": h, "name": u["home"]["name"], "starter": sp_h, "lineup": lu_h,
                     "bvp": bvp_map(h, sp_a), "spvt": spvt_map(h, a)},
            "away": {"id": a, "name": u["away"]["name"], "starter": sp_a, "lineup": lu_a,
                     "bvp": bvp_map(a, sp_h), "spvt": spvt_map(a, h)},
            "night": u["night"], "dh": u["dh"], "series": u["series_game"],
            "ctx": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ctx.items()},
            "park_lat": venues.get(str(u["venue"]), {}).get("lat"),
            "park_lon": venues.get(str(u["venue"]), {}).get("lon"),
            "park_bearing": venues.get(str(u["venue"]), {}).get("bearing", 0),
            "park_elev": venues.get(str(u["venue"]), {}).get("elev", 0),
            "roof": venues.get(str(u["venue"]), {}).get("roof", "Open"),
        })
    ups.sort(key=lambda u: u["ts"])

    out = {
        "game": "mlb", "market": "moneyline",
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "seasons": [int(s) for s in sorted(args.seasons)],
        "games_used": len(rows), "teams_n": len(teams),
        "features": FEATURES,
        "w": [float(v) for v in model["w"]], "b": float(model["b"]),
        "mu": [float(v) for v in model["mu"]], "sd": [float(v) for v in model["sd"]],
        "holdout": {"n": len(rows) - cut, "accuracy": round(acc, 4), "logloss": round(ll, 4), "brier": round(br, 4)},
        "base_rate": round(float(y.mean()), 4),
        "lg_rpg": round(st.lg_rpg(), 3), "lg_woba": round(st.lg_woba(), 4),
        "order_w": [1.16, 1.12, 1.09, 1.05, 1.00, 0.95, 0.91, 0.87, 0.85], "base": BASE,
        "l2": float(l2), "temp": float(temp), "stack": stack,
        "teams": teams, "players": players, "upcoming": ups,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), separators=(",", ":"))
    print(f"wrote {args.out}: {len(teams)} teams, {len(players)} rated players, {len(ups)} scheduled games")
    print("next:  python build.py https://cdn.jsdelivr.net/gh/<you>/mlb-model@main/model.js")


if __name__ == "__main__":
    main()
