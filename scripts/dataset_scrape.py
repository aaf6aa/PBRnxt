import io
import shutil
import cv2
import os
import pathlib
import random
import requests
import zipfile


ASSETS_URL = "https://3dassets.one/api/v2/assets?type=pbr-material&creator=ambientcg,polyhaven,cgbookcase"
AMBIENTCG_URL = "https://ambientcg.com/api/v2/full_json?include=downloadData&id="
POLYHAVEN_URL = "https://api.polyhaven.com/files/"
CGBOOKCASE_URL = "https://cgbookcase-volume.b-cdn.net/t/"

def get_assets():
    limit = 500
    offset = 0
    assets = []
    while limit > 0:
        url = ASSETS_URL + "&limit=" + str(limit) + "&offset=" + str(offset)

        response = requests.get(url).json()
        assets += response["assets"]
        if len(assets) == 0 or response["nextCollection"] is None:
            break
        
        limit = response["nextCollection"]["limit"]
        offset = response["nextCollection"]["offset"]

    # deduplicate assets
    ids = []
    for a in assets:
        if a["id"] not in ids:
            ids.append(a["id"])
        else:
            assets.remove(a)

    return assets    


def ambientcg(url, output_dir):
    asset_id = url.split("/")[-1]
    output_name = "ambientcg_" + asset_id
    output_dir = os.path.join(output_dir, output_name)

    url = AMBIENTCG_URL + asset_id

    downloads = requests.get(url).json()
    downloads = downloads["foundAssets"][0]["downloadFolders"]["default"]["downloadFiletypeCategories"]

    # reject files that don't have a zip file as they are likely to be HDRIs or Substance Painter files and etc.
    if "zip" not in downloads:
        return
    
    # sort by resolution in descending order and only accept JPGs
    downloads = downloads["zip"]["downloads"]
    downloads.sort(key=lambda x: x["attribute"], reverse=True)
    downloads = [d for d in downloads if d["attribute"].endswith("JPG")]

    # reject 1K textures
    if len(downloads) == 0 or downloads[0]["attribute"].startswith("1K"):
        return

    url = downloads[0]["downloadLink"]
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # download the zip file in-memory and extract it to the output directory
    print("Downloading " + output_name + "...")
    r = requests.get(url)

    with zipfile.ZipFile(io.BytesIO(r.content), "r") as zip_file:
        zip_file.extractall(output_dir)
    
    # rename the textures to their respective names and delete extra files
    b, n, r, d = False, False, False, False
    for f in os.listdir(output_dir):
        filename, extension = os.path.splitext(f)
        f = os.path.join(output_dir, f)

        if filename.endswith("AmbientOcclusion"):
            os.replace(f, os.path.join(output_dir, "ao" + extension))
        elif filename.endswith("Color"):
            os.replace(f, os.path.join(output_dir, "base" + extension))
            b = True
        elif filename.endswith("Displacement"):
            os.replace(f, os.path.join(output_dir, "disp" + extension))
            d = True
        elif filename.endswith("NormalGL"):
            os.replace(f, os.path.join(output_dir, "normal" + extension))
            n = True
        elif filename.endswith("Roughness"):
            os.replace(f, os.path.join(output_dir, "rough" + extension))
            r = True
        else:
            os.remove(f)
    
    if not (b and n and r and d):
        shutil.rmtree(output_dir)
    
    return


def polyhaven(url, output_dir):
    asset_id = url.split("/")[-1]
    output_name = "polyhaven_" + asset_id.replace("_", "")
    output_dir = os.path.join(output_dir, output_name)

    url = POLYHAVEN_URL + asset_id

    json = requests.get(url).json()

    # get the highest resolution textures available, and lower if not available
    res_keys = ["8k", "4k", "2k", "1k"]

    while len(res_keys):
        res_key = res_keys.pop(0)
        if res_key in json["Diffuse"]:
            break

    # reject 1K textures
    if len(res_keys) == 0:
        return
    
    textures = {
        "ao.jpg": json["AO"][res_key]["jpg"]["url"],
        "base.jpg": json["Diffuse"][res_key]["jpg"]["url"],
        "disp.jpg": json["Displacement"][res_key]["jpg"]["url"],
        "normal.jpg": json["nor_gl"][res_key]["jpg"]["url"],
        "rough.jpg": json["Rough"][res_key]["jpg"]["url"]
    }

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # download the textures to the output directory
    print("Downloading " + output_name + "...")

    for filename in textures.keys():
        url = textures[filename]
        r = requests.get(url)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(r.content)
    

def cgbookcase(url, output_dir):
    asset_id = url.split("/")[-1].split("?")[0].replace("-", " ").title().replace(" ", "")
    output_name = "cgbookcase_" + asset_id.replace("_", "").lower()
    output_dir = os.path.join(output_dir, output_name)

    for res in ["8K", "4K", "2K"]:
        url = CGBOOKCASE_URL + asset_id + f"_MR_{res}.zip"
        r = requests.head(url)
        if r.status_code == 200:
            break

    # reject 1K textures
    if r.status_code != 200:
        return
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # download the zip file in-memory and extract it to the output directory
    print("Downloading " + output_name + "...")
    r = requests.get(url)

    with zipfile.ZipFile(io.BytesIO(r.content), "r") as zip_file:
        zip_file.extractall(output_dir)

    base = None

    # rename the textures to their respective names and delete extra files
    b, n, r, d = False, False, False, False

    for f in os.listdir(output_dir):
        filename, extension = os.path.splitext(f)
        f = os.path.join(output_dir, f)

        if filename.endswith("AO"):
            os.replace(f, os.path.join(output_dir, "ao" + extension))
        elif filename.endswith("BaseColor"):
            base = os.path.join(output_dir, "base" + extension)
            os.replace(f, base)
            b = True
        elif filename.endswith("Displacement") or filename.endswith("Height"):
            os.replace(f, os.path.join(output_dir, "disp" + extension))
            d = True
        elif filename.endswith("Normal"):
            # flip the green channel of the normal map as it is inverted
            normal = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            normal[:, :, 1] = 255 - normal[:, :, 1]
            cv2.imwrite(f, normal, [cv2.IMWRITE_JPEG_QUALITY, 100])
            os.replace(f, os.path.join(output_dir, "normal" + extension))
            n = True
        elif filename.endswith("Roughness"):
            os.replace(f, os.path.join(output_dir, "rough" + extension))
            r = True
        else:
            os.remove(f)
    
    if not (b and n and r and d):
        shutil.rmtree(output_dir)
    
    return

def download(url, output_dir):
    try:
        if "ambientcg" in url:
            ambientcg(url, output_dir)
        elif "polyhaven" in url:
            polyhaven(url, output_dir)
            pass
        elif "cgbookcase" in url:
            cgbookcase(url, output_dir)
    except:
        print("Failed to download " + url)

from joblib import Parallel, delayed

def main():
    output_dir = r"F:\datasets\pbr\src"

    assets = get_assets()
    print("Found " + str(len(assets)) + " assets")

    random.shuffle(assets)
    #Parallel(n_jobs=6)(delayed(download)(asset["url"].lower(), output_dir) for asset in assets)
    for asset in assets:
        download(asset["url"].lower(), output_dir)
    

if __name__ == '__main__':
    main()