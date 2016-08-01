import numpy as np
from chainer import Variable, FunctionSet
import chainer.functions as F

# 元々「文字」から「文字」の出現確率を予測するRNN これをカスタマイズ
# 順伝播の書き方が古い
class CharRNN(FunctionSet):

    def __init__(self, n_vocab, n_units):
        super(CharRNN, self).__init__(
            # EmbedID = 単語のインデックスを元に埋め込みベクトルを作成するChainerの関数
            # n_vocabは単語の種類数（文章の単語をリスト化したディクショナリ？）　n_unitsは埋め込みベクトルの次元数
            # ベクトルの次元ごとにsoftmaxで確率が出る　→　各単語が出現する確率として数字を扱うことができる
            embed = F.EmbedID(n_vocab, n_units),
            # ChainerでRNNを書くときはモデルはLinear関数で記述　、　伝播の際の関数としてｌｓｔｍを指定する
            # Chainerの実装しているLSTMは通常の入力の他にinput gate, output gate, forget gateの3種類の入力があり、
            # これを1個のベクトルとしてまとめているためにこのような実装が必要となっています。
            # 最近は隠蔽されているverのlstmも使えるようになっている　
            l1_x = F.Linear(n_units, 4*n_units),
            l1_h = F.Linear(n_units, 4*n_units),
            l2_h = F.Linear(n_units, 4*n_units),
            l2_x = F.Linear(n_units, 4*n_units),
            l3   = F.Linear(n_units, n_vocab),
        )
        for param in self.parameters:
            # 初期のパラメータを-0.1〜0.1の間で与えています。
            param[:] = np.random.uniform(-0.08, 0.08, param.shape)

    # 順伝播
    # __call__と同じ
    def forward_one_step(self, x_data, y_data, state, train=True, dropout_ratio=0.5):
        x = Variable(x_data.astype(np.int32), volatile=not train)
        t = Variable(y_data.astype(np.int32), volatile=not train)

        # 特徴ベクトルはBag of wordsの形式なので潜在ベクトル空間に圧縮する
        h0      = self.embed(x)
        # 過学習をしないようにランダムに一部のデータを捨て、過去の状態のも考慮した第一の隠れ層を作成
        h1_in   = self.l1_x(F.dropout(h0, ratio=dropout_ratio, train=train)) + self.l1_h(state['h1'])
        # LSTMに現在の状態と先ほど定義した隠れ層を付与して学習し、隠れ層と状態を出力
        c1, h1  = F.lstm(state['c1'], h1_in)
        # 2層目も1層目と同様の処理を行う
        h2_in   = self.l2_x(F.dropout(h1, ratio=dropout_ratio, train=train)) + self.l2_h(state['h2'])
        c2, h2  = F.lstm(state['c2'], h2_in)
        y       = self.l3(F.dropout(h2, ratio=dropout_ratio, train=train))
        # ラベルは3層目の処理で出力された値を使用する。
        state   = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        return state, F.softmax_cross_entropy(y, t)

　　# 学習後作成されたモデルで、単語の予測をする機能
    def predict(self, x_data, state):
        x = Variable(x_data.astype(np.int32), volatile=True)

        h0      = self.embed(x)
        h1_in   = self.l1_x(h0) + self.l1_h(state['h1'])
        c1, h1  = F.lstm(state['c1'], h1_in)
        h2_in   = self.l2_x(h1) + self.l2_h(state['h2'])
        c2, h2  = F.lstm(state['c2'], h2_in)
        y       = self.l3(h2)
        state   = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
　　　　# 各単語の出力確率を予測するため、softmax関数を使用
        return state, F.softmax(y)

# 状態の初期化：確率的勾配法に必要なデータを与え、学習データと認識させる
def make_initial_state(n_units, batchsize=50, train=True):
    return {name: Variable(np.zeros((batchsize, n_units), dtype=np.float32),
            volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}


# 新しい書き方でのRNNの定義の仕方
"""
class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units, train=True):
        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        xを入力として、埋め込み層、LSTM層、LSTM層、全結合(出力)層の順に計算しています。
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y
"""