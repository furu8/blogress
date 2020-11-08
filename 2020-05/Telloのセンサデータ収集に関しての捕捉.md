# 前回

前回の記事で[Telloのセンサデータ収集方法](https://noleff.hatenablog.com)について書きました。今回はその記事の補足となります。良かったら、まず先にそちらを読んでいただければなと思います。

# 今回の記事内容

今回の記事は前回の内容データ収集部分に関する捕捉です。具体的にはtello.pyの中のget_tello_sensorメソッドの話になります。以下コードです。

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

ちょっと長いですが見るのはget_tello_sensorメソッドだけで良いです。このメソッド内の5行目で、time.sleep(self.INTERVAL)と記述しています。これは取得したセンサデータをCSVに出力するタイミングを操作しています。今回のコードではイニシャライザでself.INTERVAL = 0.1としていますので、0.1秒ごとにセンサデータが書き込まれます。

## 問題点

さて、何が問題なのか説明します。このself.INTERVALは、このコードを作る上で参考にしたサンプルプログラムがあります。
ここでは、self.INTERVALは0.2秒となっていますが、本プログラムでこれを0.2秒にすると、センサデータが間延びした形になるという問題が発生してしまいます。

[https://github.com/dji-sdk/Tello-Python/blob/master/tello_state.py:embed:cite]


## 検証方法

検証にはTelloとiPhone、二つのプログラムを同時に実行させ、同じ挙動をさせたときのセンサ値を見ます。今回はroll回転(Y軸)を180度回転しては戻す操作を繰り返しました。
動作のフローとしては、プログラム実行 ｰ> 10秒停止 ｰ> 約20秒間Roll回転 ｰ> 10秒放置 ｰ> プログラム停止の順番です。

## 検証結果

結果、Telloのグラフだけ、間延びしていることがわかると思います。iPhoneの方はself.INTERVALの秒数を増やすごとに、データ数は減っていますが、ほぼ同じグラフをプロットしています。
上から順に、self.INTERVALが0.1秒、0.2秒、0.3秒だったときのグラフになります。

0.1秒のとき、TelloとiPhoneが微妙にずれているのは、両手にTelloとiPhoneを持って回転させたため、逆位相になっているからです（ここは本当にミスりました）。  
0.2秒のとき、TelloのセンサデータのRoll値の変化が現れる時間がiPhoneより10秒以上遅くなっています。最初はこのずれは、Telloのセンサデータを送るときのラグだと思っていましたが、検証したことからtime.sleepが悪さしていることがわかりました。  
0.3秒のとき、もう明らかに間延びしています。間延びしてしまう原因は不明ですが、0.1秒以外は信用できないと言えるでしょう。  

- 0.1秒
[f:id:Noleff:20200606141952p:plain]

- 0.2秒
[f:id:Noleff:20200606142011p:plain]

- 0.3秒
[f:id:Noleff:20200606142024p:plain]

また、0.1秒が問題ない理由としては、Telloのセンサデータを送る仕組みが、PC側でUDPサーバを立てて、TelloがPCにひたすらセンサデータを送り続けてくるからだと思います。この秒数がおそらく0.1秒なため、0.1秒以上にすると内部的におかしなことが起きているのかもしれません。  
実はself.INTERVALを入れなくてもプログラムは動きます。下のグラフが入れなかったときのグラフです。0.1秒のときと大差ないことがわかると思います。

- 0秒
[f:id:Noleff:20200606155353p:plain]


# まとめ

今回の記事ではTelloのセンサデータが間延びしていることを検証しました。検証方法としては、同じ挙動をさせたiPhoneのセンサデータと比較するというシンプルなものです。
結果、time.sleepを0.1秒より大きな値を入れると、間延びしていることがわかりました。原因はわかりませんが、TelloはiPhoneと異なり通信してデータをCSVに溜め込んでいることが原因な気がします。

# Appnedix

iPhoneのプログラムはpythonista3という環境で作成しました。以下コードです。説明は省略します。

```python
import ui, location, csv, datetime, time, motion, sound
from datetime import datetime
import os

def csv_writer(filename, header, value, flag):
	with open(filename, mode="a") as f:
		writer = csv.DictWriter(f, fieldnames=header)
		if flag:
			writer.writeheader()
		writer.writerow(value)

# motion header
header = ['datetime', 'latitude', 'longitude', 'altitude', 'timestamp', 'horizontal_accuracy', 'vertical_accuracy', 'speed', 'course', 'pitch', 'roll', 'yaw', 'agx', 'agy', 'agz', 'gra_x', 'gra_y', 'gra_z', 'uacc_x', 'uacc_y', 'uacc_z', 'com_x', 'com_y', 'com_z', 'accuracy']

flag = True
name = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

while True:	
	# get motion
	motion.start_updates()
	# get GPSdata
	location.start_updates() # updata GPSdata 
	time.sleep(0.1)
	gravity = motion.get_gravity()
	user_accele = motion.get_user_acceleration()
	attitude = motion.get_attitude()
	magnetic = motion.get_magnetic_field()
	gps = location.get_location() # get GPSdata
	motion.stop_updates()
	location.stop_updates() 
	
	# get realtime
	now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f'))
	# user accele
	accele = [a + g for (a, g) in zip(user_accele, gravity)]
	
	# dict
	sensor_dict = {'datetime':now, 'latitude':gps['latitude'], 'longitude':gps['longitude'], 'altitude':gps['altitude'], 'timestamp':gps['timestamp'], 'horizontal_accuracy':gps['horizontal_accuracy'], 'vertical_accuracy':gps['vertical_accuracy'], 'speed':gps['speed'], 'course':gps['course'], 'pitch':attitude[1], 'roll':attitude[0], 'yaw':attitude[2], 'agx':accele[0], 'agy':accele[1], 'agz':accele[2], 'gra_x':gravity[0], 'gra_y':gravity[1], 'gra_z':gravity[2], 'uacc_x':user_accele[0], 'uacc_y':user_accele[1], 'uacc_z':user_accele[2], 'com_x':magnetic[0], 'com_y':magnetic[1], 'com_z':magnetic[2], 'accuracy':magnetic[3]}
	
	# Open csv file and write motion and GPS dictionary data
	csv_writer("/private/var/mobile/Library/Mobile Documents/iCloud~com~omz-software~Pythonista3/Documents/csvData/" + name + ".csv", header, sensor_dict, flag)
	
	if flag:
		flag = False

```