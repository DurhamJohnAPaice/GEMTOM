import requests
import zipfile
import io
import pandas as pd
import os


## Get TNS token
print("Loading dotenv...")
from dotenv import load_dotenv, dotenv_values
load_dotenv()
print(dotenv_values())
print("Dotenv loaded.")

TNS                 = "www.wis-tns.org"
url_tns_api         = "https://" + TNS + "/api/get"

TNS_BOT_ID          = "187806"
TNS_BOT_NAME        = "BotGEM"
TNS_API_KEY         = os.getenv('TNS_API_TOKEN', 'TNS_API_TOKEN not set')


# Example: replace with the real TNS API endpoint & your key
url = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"
headers = {
    'User-Agent': 'tns_marker{"tns_id":' + TNS_BOT_ID + ',"type":"bot","name":"' + TNS_BOT_NAME + '"}',
    'Authorization': 'ApiKey ' + TNS_API_KEY + ':' + TNS_BOT_NAME
}

# request the zipped JSON
response = requests.post(url, headers=headers)

# open zipfile from response content
z = zipfile.ZipFile(io.BytesIO(response.content))

with z.open(z.namelist()[0]) as f:
    df_tns = pd.read_csv(f, skiprows=1)

df_tns = df_tns.sort_values(by=['discoverydate']).reset_index(drop=True)

## Find contributed sources
df_bgem_contrib = df_tns[df_tns["reporting_group"] != "BlackGEM"]
df_bgem_contrib = df_bgem_contrib.dropna(subset=['internal_names'])
df_bgem_contrib = df_bgem_contrib[df_bgem_contrib["internal_names"].str.contains("BGEM")]

df_bgem_discoveries = df_tns[df_tns["reporting_group"] == "BlackGEM"]
df_bgem_sn_discoveries = df_bgem_discoveries[df_bgem_discoveries["name_prefix"] == "SN"]

# save to CSV
# df.to_csv("../Documents/Data/BG_files/tns_latest.csv", index=False)
df_tns.to_csv("./GEMTOM/GEMTOM/data/tns_latest.csv", index=False)
df_bgem_contrib.to_csv("./GEMTOM/GEMTOM/data/bgem_tns_contributions.csv", index=False)
df_bgem_discoveries.to_csv("./GEMTOM/GEMTOM/data/bgem_tns_discoveries.csv", index=False)
df_bgem_sn_discoveries.to_csv("./GEMTOM/GEMTOM/data/bgem_tns_sn_discoveries.csv", index=False)
