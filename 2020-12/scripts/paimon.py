import os
from os.path import join, dirname
import time
import tweepy 
from dotenv import load_dotenv
import urllib.request, urllib.error

dotenv_path = join(dirname('../'+__file__), '.env')
load_dotenv(dotenv_path)

# 環境変数
API_KEY             = os.environ.get('API_key')
API_SECRET_KEY      = os.environ.get('API_secret_key')
ACCESS_TOKEN        = os.environ.get('Access_token')
ACCESS_TOKEN_SECRET = os.environ.get('Access_token_secret')

# 検索オプション
SEARCH_PAGES_NUMBER = 1000 # 読み込むページ数
PER_PAGE_NUMBER = 100 # ページごとに返されるツイートの数（最大100）

auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

def search_illust(search_word, lang, max_id):
    url_list = []

    if max_id:
        search_results = api.search(q=search_word, lang=lang, rpp=PER_PAGE_NUMBER, max_id=max_id)
    else:
        search_results = api.search(q=search_word, lang=lang, rpp=PER_PAGE_NUMBER)
    
    for result in search_results:
        if 'media' not in result.entities:
            continue
        for media_url in result.entities['media']:
            url = media_url['media_url_https']
            if url not in url_list:
                url_list.append(url)
    max_id = result.id

    return url_list, max_id

def download_illust(url):
    url_org = url
    # url_org = '%s:orig' % url # オリジナル画像のサイズで欲しいならコメント外す
    save_path = 'G:/Image/paimon/' + url.split('/')[-1]
    
    try:
        response = urllib.request.urlopen(url=url_org)
        with open(save_path, "wb") as f:
            f.write(response.read())
    except Exception as e:
        print('error')

def main(max_id=None):
    search_word = input('search_word >')
    search_lang = input('search_lang >')

    for page in range(SEARCH_PAGES_NUMBER):
        # 進行状況
        if page % 10 == 0:
            print(page)

        # 検索
        url_list, max_id = search_illust(search_word, search_lang, max_id)

        # ダウンロード
        for url in url_list:
            download_illust(url)

        time.sleep(0.5)

if __name__ == "__main__":
    main()