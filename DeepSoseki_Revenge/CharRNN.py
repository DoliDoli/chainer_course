import numpy as np
from chainer import Variable, FunctionSet
import chainer.functions as F

class CharRNN(FunctionSet):

    def __init__(self, n_vocab, n_units):
        super(CharRNN, self).__init__(
            # EmbedID = 単語のインデックスを元に埋め込みベクトルを作成するChainerの関数
            # n_vocabは単語の種類数（文章の単語をリスト化したディクショナリ？）　n_unitsは埋め込みベクトルの次元数
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
            param[:] = np.random.uniform(-0.08, 0.08, param.shape)

    # 順伝播
    def forward_one_step(self, x_data, y_data, state, train=True, dropout_ratio=0.5):
        x = Variable(x_data.astype(np.int32), volatile=not train)
        t = Variable(y_data.astype(np.int32), volatile=not train)

        h0      = self.embed(x)
        h1_in   = self.l1_x(F.dropout(h0, ratio=dropout_ratio, train=train)) + self.l1_h(state['h1'])
        c1, h1  = F.lstm(state['c1'], h1_in)
        h2_in   = self.l2_x(F.dropout(h1, ratio=dropout_ratio, train=train)) + self.l2_h(state['h2'])
        c2, h2  = F.lstm(state['c2'], h2_in)
        #出力
        y       = self.l3(F.dropout(h2, ratio=dropout_ratio, train=train))
        state   = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        return state, F.softmax_cross_entropy(y, t)

    def predict(self, x_data, state):
        x = Variable(x_data.astype(np.int32), volatile=True)

        h0      = self.embed(x)
        h1_in   = self.l1_x(h0) + self.l1_h(state['h1'])
        c1, h1  = F.lstm(state['c1'], h1_in)
        h2_in   = self.l2_x(h1) + self.l2_h(state['h2'])
        c2, h2  = F.lstm(state['c2'], h2_in)
        y       = self.l3(h2)
        state   = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        return state, F.softmax(y)

def make_initial_state(n_units, batchsize=50, train=True):
    return {name: Variable(np.zeros((batchsize, n_units), dtype=np.float32),
            volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}
