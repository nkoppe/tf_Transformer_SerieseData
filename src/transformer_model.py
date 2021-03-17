import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout
from layers import PositionalEncoding
from layers import LayerNormalization
from layers import MultiHeadAttention



class Transformer(Model):
    def __init__(self,
                 depth_source,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.encoder = Encoder(depth_source,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               maxlen=maxlen)
        self.decoder = Decoder(depth_target,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               maxlen=maxlen)
        
        #self.out = Dense(depth_target, activation='softmax')        #出力段(カテゴリ値)
        self.out = Dense(depth_target, activation='linear')        #出力段(線形回帰)
        
        self.maxlen = maxlen

    def call(self, source, target=None):
        """
        順伝播処理
        """
        #マスクを使用する場合(入力が可変長の場合)
        #mask_source = self.sequence_mask(source)
        #hs = self.encoder(source, mask=mask_source)     #エンコーダーに入力を渡す
        
        #マスクを使用しない場合(入力が固定長の場合)
        hs = self.encoder(source)     #エンコーダーに入力を渡す

        y = self.decoder(target, hs)    #デコーダー入力とエンコーダー出力を渡す
        output = self.out(y)            #全結合層に渡す
        return output                   #出力を返す

    def sequence_mask(self, x):
        len_sequences = \
            tf.reduce_sum(tf.cast(tf.not_equal(x, 0),
                                  tf.int32), axis=1)
        mask = \
            tf.cast(tf.sequence_mask(len_sequences,
                                     tf.shape(x)[-1]), tf.float32)
        return mask

    def subsequence_mask(self, x):
        """
        Attntionに未来の情報が入力しないためのマスク作成処理
        引数に指定したした下三角行列を生成する
        """
        shape = (x.shape[1], x.shape[1])
        mask = np.tril(np.ones(shape, dtype=np.int32), k=0)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        return tf.tile(mask[tf.newaxis, :, :], [x.shape[0], 1, 1])


class Encoder(Model):
    def __init__(self,
                 depth_source,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        
        #入力する特徴量はターケンスの定理に従った遅延埋め込みを行っている
        #そのデータに大してPositionalEncodingを適用する
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)

        #N層のエンコーダーレイヤーを生成する
        self.encoder_layers = [
            EncoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         maxlen=maxlen) for _ in range(N)
        ]

    def call(self, x, mask=None):
        """
        順伝播処理
        """
        y = self.pe(x)      #PositionalEncodingを適用
        
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y, mask=mask)

        return y


class EncoderLayer(Model):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(p_dropout)
        self.norm1 = LayerNormalization()
        self.ff = FFN(d_model, d_ff)
        self.dropout2 = Dropout(p_dropout)
        self.norm2 = LayerNormalization()

    def call(self, x, mask=None):
        """
        順伝播処理
        """

        #MultiHeadなSelfAttention
        #クエリ・キー値すべてにエンコーダー入力を設定する
        #AttentionMaskはbooleanマスクを入力。形状は[B,T,S]
        h = self.attn(x, x, x, mask=mask)

        #Attentionの出力にドロップアウトとレイヤー正規化を実施＆スキップコネクション
        h = self.dropout1(h)
        h = self.norm1(x + h)

        #全結合＆スキップコネクション
        y = self.ff(h)
        y = self.dropout2(y)
        y = self.norm2(h + y)

        return y


class Decoder(Model):
    def __init__(self,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        
        #入力データは遅延埋め込みされたデータである前提
        #PositionalEncodingを実施
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.decoder_layers = [
            DecoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         maxlen=maxlen) for _ in range(N)
        ]

    def call(self, x, hs,
             mask=None,
             mask_source=None):
        """
        順伝播処理
        """

        #PositionalEncodingを実施
        y = self.pe(x)

        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, hs,            #前段出力、エンコーダー出力
                              mask=mask,        #デコーダー入力マスク
                              mask_source=mask_source)      #エンコーダー入力マスク

        return y


class DecoderLayer(Model):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128):
        super().__init__()
        #[1]デコーダー入力にAttentionによる注目情報を追加する
        self.self_attn = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(p_dropout)
        self.norm1 = LayerNormalization()

        #[2]MultiHeadAttention(Self-Attention)にデコーダー入力とマスク情報を渡す
        self.src_tgt_attn = MultiHeadAttention(h, d_model)
        self.dropout2 = Dropout(p_dropout)
        self.norm2 = LayerNormalization()

        #[3]全結合
        self.ff = FFN(d_model, d_ff)
        self.dropout3 = Dropout(p_dropout)
        self.norm3 = LayerNormalization()

    def call(self, x, hs,
             mask=None,
             mask_source=None):
        h = self.self_attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)

        z = self.src_tgt_attn(h, hs, hs,
                              mask=mask_source)
        z = self.dropout2(z)
        z = self.norm2(h + z)

        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)

        return y


class FFN(Model):
    """
    全結合ユニット
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.l1 = Dense(d_ff, activation='relu')
        self.l2 = Dense(d_model, activation='linear')

    def call(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y

