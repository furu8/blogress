##はじめに
Ryze Telloでプログラミングしたときのセンサデータ収集方法についてまとめます。言語はPythonでやりました。SDKとして、DJIの公式SDK「Tello-Python」を使っています。

https://github.com/dji-sdk/Tello-Python

本ブログでは、上記のサンプルコードを理解、実装済みであることを想定しています。まだの方はgit cloneして動かしてみてください。
以下のサイトが参考になると思います。

https://qiita.com/hsgucci/items/3327cc29ddf10a321f3c

## 開発環境

- macOS Catalina バージョン10.15.3（windows10でも動作確認済み)
- Python3.7.6
- Tello EDU （Tello SDK 2.0）

## 目的

今回やりたいことはプログラムでTelloを飛行させつつ、Tello内に内蔵されているセンサの情報を取得し、CSVファイルに保存することです。Telloには内部にデータを保存する機構がないため、取得してあるデータは操作しているPCに送る必要があります。

簡単に言うと、Telloとの通信には3種類あるようです。その中にTelloの機体ステータスを受信し続ける8890ポートがあります。PC側でポート8890を指定してUDPサーバーを立てると、Telloが自身の状態、つまりセンサデータを送信し続けてくれます。

データの取得はポート8890で良いですが、こちらからコマンドを送り、Telloに飛行命令を出すのはポート8889になります。この通信は双方向通信で、PCからコマンドを送ると、Tello側から応答が帰ってきます。

今回のプログラムではカメラは使わないため、ポート8890と8889を使います。ポート8890に関するサンプルプログラムはtello_state.py、ポート8889に関するサンプルプログラムがSingle_Tello_Testフォルダ内にある、tello_test.py、tello.py、stats.pyです。

- [tello_state.py解説](https://qiita.com/hsgucci/items/7067e356eda5ba2d8e73)
- [Single_Tello_Test解説](https://qiita.com/hsgucci/items/a199e021bb55572bb43d)

上記のサイトが「Tello-Python」の中にあるtello_state.pyとSingle_Tello_Testのサンプルプログラムを詳しく解説してくださっています。 

## 前準備
今回のプログラムはPython3環境で実装していますが、「Tello-Python」はPython2環境で動かすプログラムとなっています。そのため、そのままPython3環境で実行すると、しょうもないエラーが出てしまいます。まずはそれをどうにかしましょう。

https://qiita.com/coffiego/items/54c8bb553394590787f9

と言ってもリンクを貼るだけですが。

上記のサイトではSingle_Tello_TestのサンプルプログラムをPython3で動くように修正してくださっています。tello_test.pyを手動でコマンドを手入力で実行できるように変更していますが、今回はSDKのサンプルプログラム通り、コマンドがあらかじめ書かれたテキストを読み込んで動くようにプログラムを実装しています。自分がやりたい方で実装してみてください。

## センサデータ収集
Tellloのセンサデータを収集するのに使う主なプログラムは以下の４つです。

- main.py
- move_Tello.py
- tello.py
- stats.py

ディレクトリ構成は以下のようになっています。
```
data
  |--raw
  |    |--実行した日付.csv
notebooks
  |--visualize.ipynb
scripts
  |--data
  |    |--Tello
  |    |    |--FlightPlan
  |    |    |    |--command.txt
  |    |    |--log
  |    |    |    |--実行した日付.txt
  |    |    |--main.py
  |    |    |--move_Tello.py
  |    |    |--tello.py
  |    |    |--stats.py
```

### main.py
move_Tello.pyを実行しているだけです。なくてもいいですが、ここで、手動でコマンドを打つか、飛行計画からコマンドを実行するかを分岐させると良いかもしれません。

```python
from move_Tello import MoveTello

def move_tello():
    move = MoveTello()
    # 飛行計画から動かす
    move.auto_move()

if __name__ == "__main__":
    print('started')
    move_tello()
```

### move_Tello.py

- post_commandメソッド

    このコードでTelloに実行命令を出しています。
    飛行計画にdelayが合った場合、何秒delayしたかを標準出力します。

- read_FlightPlanメソッド

    command.txtに書いてあるコマンドを読み込んで、リストとして保存しています。command.txtの中身は以下のとおりです。飛行計画はお好みで設定してみてください。下記のコマンドは離陸 ｰ> 4m前進 ｰ> 180度時計回り ｰ> 4m前進 ｰ> 着陸の順に実行されます。

```
command
takeoff
delay 3
forward 400
delay 3
cw 180
delay 3
forward 400
delay 3
land
```

- outputメソッド
    Single_Tello_Testのtello_test.pyのログ出力部分をメソッド化しているだけです。正直センサデータを記録しているので、今回このログはあまり重要ではありません。実行する度にログファイルが作られるため、筆者はずっとコメントアウトしてました。

- auto_moveメソッド

    read_FlightPlanから入手したコマンドを順番にpost_commandに渡し、コマンドを制御しています。for文の中にsleepを入れているのは、Telloに一気にコマンドを渡さないためです。現状、Telloにコマンドを渡し、そのコマンドの挙動を実行している間に次のコマンドを送っても無視されるため、それをなくしたいというアプローチです。Telloが確実に前のコマンドの挙動を終えている5秒後に次のコマンドを送るようにしています。

``` python
from tello import Tello
import sys
from datetime import datetime
import time

class MoveTello:
    def __init__(self):
        # ログ記録用時刻
        self.name = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        # 強制着陸用時刻
        self.t1 = time.time()
        # インスタンス
        self.tello = Tello()
        
    # 飛行計画から動かす
    def auto_move(self):
        # 飛行計画読み込み
        commands = self.read_FlightPlan()
        # コマンドを判定して投げる
        for command in commands:
            self.post_command(command)
            time.sleep(5)
        # ログ出力
        self.output() #邪魔ならここをコメントアウト

    # logと標準出力
    def output(self):
        log = self.tello.get_log()
        out = open('log/' + self.name + '.txt', 'w')
        for stat in log:
            stat.print_stats()
            str = stat.return_stats()
            out.write(str)

    # 飛行計画読み込み
    def read_FlightPlan(self):
        with open('FlightPlan/command.txt', 'r') as f:
            commands = f.readlines()
        return commands
    
    # コマンド判定して投げる
    def post_command(self, command):
        # 空文字かつ改行
        if command != '' and command != '\n':
            # 末尾の空白を削除
            command = command.rstrip() 
            # commandの中にdelayがない
            if command.find('delay') != -1: 
                # 秒数抽出
                sec = float(command.partition('delay')[2])
                print('delay %s' % sec)
                time.sleep(sec)
            else:
                self.tello.send_command(command)

```

### tello.py

- イニシャライザに追記

    ほぼSingle_Tello_Testのtello.pyそのままなので、主な変更点だけ述べます。  
    \__init__.pyにいくつか変数を追記しています。CSVファイルに記録するために必要なヘッダー、フラグ、記録日、コマンドを変数として持ちます。Telloのステータスを受信するためにlocal_postは8890に変更し、threading.Threadの引数targetをget_tello_sensorに変更して、別スレッドとしてセンサデータの収集とCSV出力を行います。

- get_tello_sensorメソッド

    このコードはtello_state.pyにいくつか変更を加えたプログラムです。主な変更分はデータの整形になるため、少々複雑です（くそコードとも言います）。ポート8890から得られるステータスの出力例は以下のようになっています。

    [f:id:Noleff:20200501011644p:plain]

    socket.recvfrom(1024)から返ってくるreseponseはbytes型なので、splitするためにstring型にしています。responseのセンサデータ部分だけをsplitで抽出し、sensor_listに保存します。mpryは同じsplit方法で抽出できないので別途splitしています。  
    CSVファイルに保存するのはwrite_csvメソッドです。フラグはヘッダを最初の一回だけ保存するために用いています。

```python
import socket
import threading
import time
from datetime import datetime
import csv
from stats import Stats

class Tello:
    def __init__(self):
        self.INTERVAL = 0.1

        self.header = ['datetime', 'status',
                'mid', 'x', 'y', 'z',
                'mpry1', 'mpry2', 'mpry3',
                'pitch', 'roll', 'yaw', 
                'agx', 'agy', 'agz',
                'vgx', 'vgy', 'vgz', 
                'templ', 'temph', 'tof', 'h', 
                'bat', 'baro', 'time']
        self.flag = True
        self.name = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.status = None

        # ステータス受信用のUDPサーバの設定
        self.local_ip = ''
        self.local_port = 8890
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket.bind((self.local_ip, self.local_port))

       # コマンド送信用の設定
        self.receive_thread = threading.Thread(target=self.get_tello_sensor)
        # self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        self.tello_ip = '192.168.10.1'
        self.tello_port = 8889
        self.tello_address = (self.tello_ip, self.tello_port)
        self.log = []

        self.MAX_TIME_OUT = 15.0

    def send_command(self, command):
        """
        Send a command to the ip address. Will be blocked until
        the last command receives an 'OK'.
        If the command fails (either b/c time out or error),
        will try to resend the command
        :param command: (str) the command to send
        :param ip: (str) the ip of Tello
        :return: The latest command response
        """
        self.status = command
        self.log.append(Stats(command, len(self.log)))

        self.socket.sendto(command.encode('utf-8'), self.tello_address)
        print('sending command: %s to %s' % (command, self.tello_ip))

        start = time.time()
        while not self.log[-1].got_response():
            now = time.time()
            diff = now - start
            if diff > self.MAX_TIME_OUT:
                print ('Max timeout exceeded... command %s' % command)
                # TODO: is timeout considered failure or next command still get executed
                # now, next one got executed
                return
        print('Done!!! sent command: %s to %s' % (command, self.tello_ip))

    def _receive_thread(self):
        """Listen to responses from the Tello.

        Runs as a thread, sets self.response to whatever the Tello last returned.

        """
        while True:
            try:
                self.response, ip = self.socket.recvfrom(1024)
                print('from %s: %s' % (ip, self.response))
                self.log[-1].add_response(self.response)
            except socket.error as exc:
                print("Caught exception socket.error : %s" % exc)

    def get_tello_sensor(self):
        while True:
            index = 0                                
            try:
                index += 1
                time.sleep(self.INTERVAL) # 一定時間待つ
                now = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f'))
                self.response, ip = self.socket.recvfrom(1024) # 受信は最大1024バイトまで
                self.response = str(self.response)
                
                # print('from %s: %s' % (ip, self.response))
                self.log[-1].add_response(self.response)
                
                if self.response == "b'ok'":
                    continue
                
                # 受信データに手を加える
                self.response = self.response.split(';')[:21] # センサ関係だけ抽出
                sensor_list = []
                for sensor in self.response:

                    s = sensor.split(':')
                    if s[0] == 'mpry':
                        mpry_list = s[1].split(',')
                    else:
                        sensor_list.append(s[1])
                    
                sensor_dict = {
                    'datetime':now, 'status':self.status,
                    'mid':sensor_list[0], 'x':sensor_list[1], 'y':sensor_list[2], 'z':sensor_list[3],
                    'mpry1':mpry_list[0], 'mpry2':mpry_list[1], 'mpry3':mpry_list[2],
                    'pitch':sensor_list[4], 'roll':sensor_list[5], 'yaw':sensor_list[6],
                    'agx':sensor_list[17], 'agy':sensor_list[18], 'agz':sensor_list[19],
                    'vgx':sensor_list[7], 'vgy':sensor_list[8], 'vgz':sensor_list[9],
                    'templ':sensor_list[10], 'temph':sensor_list[11], 'tof':sensor_list[12], 'h':sensor_list[13],
                    'bat':sensor_list[14], 'baro':sensor_list[15], 'time':sensor_list[16]
                }
                
                self.write_csv('../../../data/raw/'+ self.name + '.csv', self.header, sensor_dict, self.flag)

                if self.flag:
                    self.flag = False
                
            except socket.error as exc:
                print("Caught exception socket.error : %s" % exc)

    def write_csv(self, filename, header, value, flag):
        with open(filename, mode='a', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if flag:
                writer.writeheader()
            writer.writerow(value)

    def on_close(self):
        # for ip in self.tello_ip_list:
        self.socket.sendto('land'.encode('utf-8'), self.tello_address)
        self.socket.close()

    def get_log(self):
        return self.log
```

### stats.py
Single_Tello_Testのstats.pyをpython3環境で実行できるようにしただけです。説明は省略します。

```python
from datetime import datetime

class Stats:
    def __init__(self, command, id):
        self.command = command
        self.response = None
        self.id = id

        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None

    def add_response(self, response):
        self.response = response
        self.end_time = datetime.now()
        self.duration = self.get_duration()
        # self.print_stats()

    def get_duration(self):
        diff = self.end_time - self.start_time
        return diff.total_seconds()

    def print_stats(self):
        print('\nid: %s' % self.id)
        print('command: %s' % self.command)
        print('response: %s' % self.response)
        print('start time: %s' % self.start_time)
        print('end_time: %s' % self.end_time)
        print('duration: %s\n' % self.duration)

    def got_response(self):
        if self.response is None:
            return False
        else:
            return True

    def return_stats(self):
        str = ''
        str +=  '\nid: %s\n' % self.id
        str += 'command: %s\n' % self.command
        str += 'response: %s\n' % self.response
        str += 'start time: %s\n' % self.start_time
        str += 'end_time: %s\n' % self.end_time
        str += 'duration: %s\n' % self.duration
        return str
```

## データの描画

### コード
実際に取得したセンサをグラフ化しました。こちらはnotebook形式でコーディングしています。
'mid'、 'x'、 'y'、 'z'、'mpry1'、'mpry2'、'mpry3'に関しては、ミッションパッド使用時にしか値に変化が現れないとのことなのでグラフとして出力はしていません。

```python
import pandas as pd
import matplotlib.pyplot as plt

# センサデータ取得
def get_TelloSensor_data(filename):
    df = pd.read_csv('../data/raw/'+ filename +'.csv')
    return df

# datetimeから時間の差を追加
def differ_datetime(df):
    # dateimeから時間の差を割り当てる
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S:%f')
    df['dif_sec'] = df['datetime'].diff().dt.total_seconds()
    df['dif_sec'] = df['dif_sec'].fillna(0)
    df['cum_sec'] = df['dif_sec'].cumsum()
    return df

# main
sensor_list = ['pitch', 'roll', 'yaw', 
            'agx', 'agy', 'agz',
            'vgx', 'vgy', 'vgz', 
            'templ', 'temph', 'tof', 'h', 
            'bat', 'baro']

for sensor in sensor_list:
    # tello
    tel_df = get_TelloSensor_data(test)
    tel_dif_df = differ_datetime(tel_df)
    # visualize
    fig = plt.figure(figsize=(16,4))
    plt.plot(tel_dif_df['cum_sec'], tel_dif_df[sensor], label='tello')
    plt.xlabel('cum_sec')
    plt.ylabel(sensor)
    plt.legend()
    plt.show()
```

### グラフ
データとして挙動がわかる部分のセンサだけをピックアップしました。
ソースは見つかりませんでしたが、Telloの進行方向（カメラがついている方向）がy軸、横方向がx軸、垂直方向がz軸です。センサデータが収集できるスマートフォンと同じ挙動をさせたときに同じグラフになるかどうかで検証しました。

#### vgy

速度のvgy(y軸)では、Telloが前進したときに速度変化が現れているのがわかると思います。なぜ、往路の前進ではマイナス値が出ているのかは不明ですが……。

[f:id:Noleff:20200501014416p:plain]

#### pitch

ジャイロのpitch(x軸の回転)では、とてもわかりやすい値が出ました。前進するときTelloは少しだけ前側に傾きます。その後4m先で停止するために後ろ側に傾きます。これより、谷ができた後に山ができているというグラフが描画されたと考えられます。なお、[SDK2.0公式ドキュメント](https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf)には回転の命令は1~360の範囲となっていますが、なぜかジャイロセンサの出力値は-180~180の範囲で出力されます。

[f:id:Noleff:20200501014437p:plain]

#### yaw

ジャイロのyaw(z軸の回転)では、4m前進した後の時計回りに180度回転するときに値がきちんと変わりました。-125~50なので、ほぼ180度変化があったとみていいでしょう。

[f:id:Noleff:20200501023146p:plain]

#### tof

距離センサのtof(Time Of Flight)では、Telloの高さが出力されます。Telloををひっくり返すと赤く光っているやつだと思います。おおむね正しく高さが検出できていると思います。

[f:id:Noleff:20200501014514p:plain]

## 今後の課題

今回はTelloにテキストファイルから飛行計画を読み込んで命令を与えていました。しかし、現状の方法は一つのコマンド命令を実行 ｰ> 命令が終わるとホバリング(sleep中) ｰ> 次のコマンド命令を実行……、というふうにTelloの挙動はスムーズではありません。今後はスムーズな飛行できるように改良していこうと思います。