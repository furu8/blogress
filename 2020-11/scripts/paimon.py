import os
from os.path import join, dirname
from requests_oauthlib import OAuth1Session
import requests
import json, datetime, time, pytz, re, sys,traceback
from collections import defaultdict
from dotenv import load_dotenv

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

oauth = OAuth1Session(API_KEY, API_SECRET_KEY,ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

def get_tweet_data(search_word, max_id):
    url = 'https://api.twitter.com/1.1/search/tweets.json'
    params = {
            'q': search_word,
            'count': PER_PAGE_NUMBER,
    }
    # max_idの指定があれば設定する
    if max_id != -1:
        params['max_id'] = max_id

    req = oauth.get(url, params=params)   # Tweetデータの取得

    # 取得したデータの分解
    if req.status_code == 200: # 成功した場合
        timeline = json.loads(req.text)
        metadata = timeline['search_metadata']
        statuses = timeline['statuses']
        limit = req.headers['x-rate-limit-remaining'] if 'x-rate-limit-remaining' in req.headers else 0
        reset = req.headers['x-rate-limit-reset'] if 'x-rate-limit-reset' in req.headers else 0              
        return {
            'result': True, 
            'metadata': metadata, 
            'statuses': statuses, 
            'limit':limit, 
            'reset_time': datetime.datetime.fromtimestamp(float(reset)), 
            'reset_time_unix': reset 
        }
    else: # 失敗した場合
        print ('Error: %d' % req.status_code)
        return { 
            'result': False, 
            'status_code': req.status_code
        }


def search_illust(res):
    url_list = []

    tweet_list = res['statuses']
    for tweet in tweet_list:
        if 'extended_entities' in tweet:
            for media in tweet['extended_entities']['media']:
                media_type = media['type']
                url = media['media_url']
                if media_type == 'photo':
                    url_list.append(url)

    return url_list


def download_illust(url_list):
    for url in url_list:
        url_org = url
        # url_org = '%s:orig' % url # オリジナル画像のサイズで欲しいならコメント外す
        save_path = 'D:/Illust/Paimon/raw/' + os.path.basename(url)
        # save_path = 'G:/Image/paimon/raw/' + os.path.basename(url)

        try:
            response = requests.get(url_org)
            response.raise_for_status()
            # DL済みの画像かどうか判定
            if not os.path.exists(save_path):
                with open(save_path, 'wb') as f:
                    # print('save: ' + save_path)
                    f.write(response.content)
        except Exception as e:
            print('download error')
            break

def sleep_limit(res):
    # 待ち時間の計算. リミット+5秒後に再開する
    diff_sec = int(res['reset_time_unix']) - _now_unix_time()
    print('sleep %d sec.' % (diff_sec + 5))

    if diff_sec > 0:
        time.sleep(diff_sec + 5)


# 現在時刻をUNIX Timeで返す
def _now_unix_time():
    return time.mktime(datetime.datetime.now().timetuple())


def main():
    max_id = -1
    search_word = input('search_word >') # パイモン
    # search_lang = input('search_lang >') # ja

    for page in range(SEARCH_PAGES_NUMBER):
        # データ取得
        res = get_tweet_data(search_word, max_id)

        # 失敗したら終了する
        if res['result'] == False:    
            print('status_code', res['status_code'])
            break
        else:
            max_id = res['statuses'][-1]['id'] # 次のmax_idを記録

        # 回数制限
        if int(res['limit']) == 0:    
            sleep_limit(res)
        # 回数制限でなければダウンロード処理実行
        else:
            if len(res['statuses']) == 0:
                sys.stdout.write('statuses is none.')
            elif 'next_results' in res['metadata']:
                # 検索
                url_list = search_illust(res)
                # ダウンロード
                download_illust(url_list)
                # 進行状況
                if page % 10 == 0:
                    print(page)
            else:
                sys.stdout.write('next is none. finished.')
                break

if __name__ == '__main__':
    main()