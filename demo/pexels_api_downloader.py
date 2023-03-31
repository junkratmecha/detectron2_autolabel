import requests
import os
import json
from dotenv import load_dotenv

# APIキーを設定
load_dotenv()
api_key = os.environ.get("PEXELS_API_KEY")

# 画像を検索する関数
def search_photos(query, per_page=50):
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}"
    headers = {"Authorization": api_key}
    response = requests.get(url, headers=headers)
    return response.json()

# 画像をダウンロードする関数
def download_photos(photos, keyword, folder="./pexels"):
    folder = os.path.join(folder, keyword)
    os.makedirs(folder, exist_ok=True)

    for photo in photos:
        url = photo["src"]["large"]
        response = requests.get(url)
        file_name = os.path.join(folder, f"{photo['id']}.jpg")

        with open(file_name, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")

# メイン処理
def main():
    keyword = "machine human"
    search_result = search_photos(keyword)
    download_photos(search_result["photos"], keyword)

if __name__ == "__main__":
    main()