"""
build.py - bake model.json into the MLB app.

Run:   python build.py                                   # offline build, no online updates
       python build.py https://cdn.jsdelivr.net/gh/<you>/mlb-model@main/model.js

Makes: app.html            open this to check it
       apk-upload.zip      upload this to WebIntoApp
       model.js            publish this daily; the installed app picks it up with no new APK
"""
import json, os, sys, zipfile

here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, "app.template.html"), encoding="utf-8").read()
mp = os.path.join(here, "model.json")
if not os.path.exists(mp):
    sys.exit("model.json not found - run mlb_ml_trainer.py first")

model = json.load(open(mp, encoding="utf-8"))
blob = json.dumps(model, separators=(",", ":")).replace("</", "<\\/")
url = os.environ.get("MODEL_URL", "")
if len(sys.argv) > 1:
    url = sys.argv[1]

out = tpl.replace("__MODEL_JSON__", blob).replace("__MODEL_URL__", url)
open(os.path.join(here, "model.js"), "w", encoding="utf-8").write("window.MODEL_REMOTE=" + blob + ";")
open(os.path.join(here, "app.html"), "w", encoding="utf-8").write(out)
with zipfile.ZipFile(os.path.join(here, "apk-upload.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("index.html", out)

print("also wrote model.js - publish this file daily and the app picks it up without a new APK"
      if url else
      "NO MODEL_URL SET: this build can never update itself online.\n"
      "  Rebuild with:  python build.py https://cdn.jsdelivr.net/gh/<you>/mlb-model@main/model.js")
print(f"wrote app.html and apk-upload.zip ({len(out)//1024} KB): "
      f"{len(model['teams'])} teams, {len(model.get('upcoming', []))} scheduled games, "
      f"{len(model['players'])} rated players, as of {model['as_of']}")
