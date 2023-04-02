import requests
import os
import json
from dotenv import load_dotenv

# APIキーを設定
load_dotenv()
api_key = os.environ.get("PIXABAY_API_KEY")

# 画像を検索する関数
def search_photos_pixabay(query, per_page=50):
    url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo&per_page={per_page}"
    response = requests.get(url)
    return response.json()

# 画像をダウンロードする関数
def download_photos_pixabay(photos, keyword, folder="./pixabay"):
    folder = os.path.join(folder, keyword)
    os.makedirs(folder, exist_ok=True)

    for photo in photos:
        url = photo["largeImageURL"]
        response = requests.get(url)
        file_name = os.path.join(folder, f"{photo['id']}.jpg")

        with open(file_name, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")

# メイン処理
def main():
    keyword = "humanoid"
    search_result = search_photos_pixabay(keyword)
    download_photos_pixabay(search_result["hits"], keyword)

if __name__ == "__main__":
    main()
