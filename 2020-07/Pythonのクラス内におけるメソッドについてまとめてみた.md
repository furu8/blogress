# 目的

備忘録用です。普段Pythonでプログラミングするとき、めんどくさいのであまりクラスを使ってプログラミングしないことが多いです（jupyter notebookのベタ書き脳死コーディングのせい）。  
ですが、最近はちょっとこのままでは良くないなと、あまり使う必要ないと思う場面でもわざわざクラスにしてPythonのコードを書いてます。  

今回の記事は二番煎じ感満載かもしれませんが、ご了承ください。  

# メソッド

クラス内のメソッドには以下の３種類があります。

- インスタンスメソッド
- クラスメソッド
- スタティックメソッド

## インスタンスメソッド

普通のメソッドがこれです。簡単に例題プログラムを書きます。

```python
class Student:
    
    def __init__(self, name, s_id):
        self.name = name
        self.s_id = s_id
        
    def reply_name(self):
        print("私は" + self.name + "と言います")
        
    def reply_id(self):
        print("私の学籍番号は" + str(self.s_id) + "です")

if __name__ == '__main__':
    st = Student('山田太郎', '20A999')
    st.reply_name()
    st.reply_id()
```

```
山田太郎と言います
学籍番号は20A999です
```

学生を表すStudentクラスを作りました。結果は上記のようになります。  
クラス内のインスタンスメソッドであるreply_nameとreply_idはどちらもpublicなメソッドになります。  
privateなメソッドも追加してみましょう。

```python
class Student:

    def __init__(self, name, s_id):
        self.name = name
        self.s_id = s_id
        # 成績は順番に「秀、優、良、可、不可」
        self.school_credit = {}

    def reply_name(self):
        print(self.name + "と言います")
        
    def reply_id(self):
        print("学籍番号は" + str(self.s_id) + "です")

    def reply_gpa(self):
        print("GPAは" + str(self.__calc_GPA()) + "です")

    def input_credit(self, credit_dict):
        self.school_credit = credit_dict

    def __calc_GPA(self):
        gpa = 0
        gpa_list = ['不可', '可', '良', '優', '秀']

        # 取得した単位はすべて1とする
        for sc in self.school_credit.values():
            if sc in gpa_list:
                gpa += gpa_list.index(sc)
               
        gpa /= len(self.school_credit)
        
        return gpa

if __name__ == '__main__':
    st = Student('山田太郎', '20A999')
    st.reply_name()
    st.reply_id()
    st.input_credit({'英語':'可', '実験':'優', 'プログラミング':'秀', 'DB':'良'})
    st.reply_gpa()
```

```
山田太郎と言います
学籍番号は20A999です
GPAは2.5です
```

GPAの計算式は雑ですがこんな感じで。計算式間違ってたらごめんなさい。  
__calc_GPAメソッドがprivateなので、次のようにするとエラーを吐きます。

```python
if __name__ == '__main__':
    st = Student('山田太郎', '20A999')
    st.reply_name()
    st.reply_id()
    st.input_credit({'英語':'可', '実験':'優', 'プログラミング':'秀', 'DB':'良'})
    print(st.__calc_GPA())
```

```
山田太郎と言います
学籍番号は20A999です
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
 in 
      4     st.reply_id()
      5     st.input_credit({'英語':'可', '実験':'優', 'プログラミング':'秀', 'DB':'良'})
----> 6     print(st.__calc_GPA())

AttributeError: 'Student' object has no attribute '__calc_GPA'
```

privateなので外からアクセスできません。  
しかし、実は「インスタンス._クラス名__変数名」でアクセスできます。推奨されないので覚えなくてもいいですが……。ちなみに変数も同様です。

```
if __name__ == '__main__':
    st = Student('山田太郎', '20A999')
    st.reply_name()
    st.reply_id()
    st.input_credit({'英語':'可', '実験':'優', 'プログラミング':'秀', 'DB':'良'})
    print(st._Student__calc_GPA())
```

```
山田太郎と言います
学籍番号は20A999です
2.5
```

## クラスメソッド

### インスタンスメソッドとの違い

ここから個人的には本題です。  
インスタンスメソッドとの違いを挙げるとすれば、まずはクラスから直接呼び出せることだと思います。メソッドの頭に@classmethodとデコレートし、慣例的に引数をclsにします。selfはインスタンスなのに対し、こちらはクラスを意味します。

```python
class Student:

    def __init__(self, name, s_id):
        self.name = name
        self.s_id = s_id

    def reply_name(self):
        print(self.name + "と言います")
        
    @classmethod
    def reply_greeting(cls):
        print('こんにちは')

if __name__ == '__main__':
    st = Student('山田太郎', '20A999')
    st.reply_greeting()
    Student.reply_greeting()
```

```
こんにちは
こんにちは
```

### クラスメソッドの使い所

これだけの例だと、いまいち利点や使い所がわかりません。  
例えば外部ファイル（JSONなど）から学生の情報を取得するとします（普通はDBでしょうが）。今までと同様に書けば以下のようになります。

```python
import pandas as pd

class Student:

    def __init__(self, name, s_id):
        self.name = name
        self.s_id = s_id

    def reply_name(self):
        print(self.name + "と言います")

if __name__ == '__main__':
    info_df = pd.read_json('student.json', encoding='UTF-8')
    st = Student(info_df['name'].values[0], info_df['id'].values[0])
    st.reply_name()
```

関数にするなら以下です。

```python
import pandas as pd

class Student:

    def __init__(self, name, s_id):
        self.name = name
        self.s_id = s_id

    def reply_name(self):
        print(self.name + "と言います")

def get_student():
    info_df = pd.read_json('student.json', encoding='UTF-8')
    return Student(info_df['name'].values[0], info_df['id'].values[0])

if __name__ == '__main__':
    st = get_student()
    st.reply_name()
```

では、続いてクラスメソッドで書きます。

```python
import pandas as pd

class Student:

    def __init__(self, name, s_id):
        self.name = name
        self.s_id = s_id

    def reply_name(self):
        print(self.name + "と言います")

    @classmethod
    def get_student(cls):
        info_df = pd.read_json('student.json', encoding='UTF-8')
        return cls(info_df['name'].values[0], info_df['id'].values[0])

if __name__ == '__main__':
    st = Student.get_student()
    st.reply_name()
```

違いはなんとなくわかるかもしれませんが、クラスメソッドの利点がわかりにくいので解説します。  
最大のうまみは**クラスの中にインスタンスを作るメソッドを書ける**ことです。  

関数にした場合とクラスメソッドにした場合を比較するとわかりやすいですが、get_studentをクラスメソッドではStudentクラス内でまとめて管理できます。  
MainクラスからStudentクラスをimportするときなど、まとめられている方が使い勝手が良いです。  
クラスに依存したメソッドはクラスメソッドで定義するほうが好ましいと思います。  
  
また、実装方法によって一概には言えませんが、クラスメソッドを使わない場合、毎回

```python
info_df = pd.read_json('student.json', encoding='UTF-8')
```

を書かなければなりません。非常に面倒です。  

## スタティックメソッド

最後にスタティックメソッドとクラスメソッドの違いをまとめます。  
スタティックメソッドもクラスメソッド同様クラスからもインスタンスからも呼び出せます。  
しかし、いくつか違いあり、1つ目は@staticmethodでデコレートすること。2つ目は引数になにも受け取らない実装が可能なことです。  
インスタンスメソッドではself、クラスメソッドではclsを暗黙的に引数として書く必要がありますが、スタティックメソッドは書く必要はありません。そのため、**クラスに依存しないメソッド**と明示的に記述することができます。

```python
lass Student:

    def __init__(self, name, s_id):
        self.name = name
        self.s_id = s_id

    def reply_name(self):
        print(self.name + "と言います")

    @staticmethod
    def reply_greeting():
        print('こんにちは')

if __name__ == '__main__':
    Student.reply_greeting()
    st = Student('山田太郎', '20E999')
    st.reply_name()
```

```
こんにちは
山田太郎と言います
```

replay_greetingはこんにちはと出力するだけなため、まったくクラスに依存しません。このようなメソッドはクラスメソッドではなく、スタティックメソッドにすることが好ましいです。

しかし、クラスメソッドがあれば、スタティックメソッドがなくても実装上は問題ありません。  
最初にクラスメソッドのプログラム例で書いたように、スタティックメソッドを使って書けるコードはクラスメソッドでも書けるからです（reply_greetingの話）。  

Pythonにスタティックメソッドが必要なのかという[記事](https://atsuoishimoto.hatenablog.com/entry/20100807/1281169026)もあるため、スタティックメソッドを使うのはある意味自己満足の世界な気もします。

# 参考文献

[https://djangobrothers.com/blogs/class_instance_staticmethod_classmethod_difference/:title]

[https://qiita.com/ysk24ok/items/848daec3886f1030f587:title]

[https://blog.pyq.jp/entry/Python_kaiketsu_190205:title]

[https://qiita.com/motoki1990/items/376fc1d1f3d59c960f5c:title]




