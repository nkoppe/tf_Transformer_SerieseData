import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import metrics

from .utils import LoadCsv
from .utils import ConvertWindowDataset
from .transformer_model import Transformer


def compute_loss(t, y):
    """
    損失を計算
    """
    return tf.losses.mean_squared_error(t, y)       #回帰モデルなのでMSEにする

def train_step(train_loss, x, t, depth_t):
    """
    学習処理
    """
    with tf.GradientTape() as tape:     #勾配計算の明示的指示
        preds = model(x, t)
        # t = t[:, 1:]
        # mask_t = tf.cast(tf.not_equal(t, 0), tf.float32)
        # t = tf.one_hot(t, depth=depth_t, dtype=tf.float32)
        # t = t * mask_t[:, :, tf.newaxis]
        loss = compute_loss(t, preds)
    #勾配計算ここまで

    #損失からパラメータ更新量を算出(更新が許可されているもののみ)
    grads = tape.gradient(loss, model.trainable_variables)

    #可訓練なパラメータについて最適化アルゴリズムに従って更新
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #学習データに対するロスを算出する
    train_loss(loss)

    return preds        #推論結果を返す

def val_step(val_loss, x, t):
    """
    検証処理
    """
    preds = model(x, t)     #推論を実施
    # t = t[:, 1:]
    # mask_t = tf.cast(tf.not_equal(t, 0), tf.float32)
    # t = tf.one_hot(t, depth=depth_t, dtype=tf.float32)
    # t = t * mask_t[:, :, tf.newaxis]
    loss = compute_loss(t, preds)       #損失を計算する
    val_loss(loss)
    return preds

def test_step(x):
    preds = model(x)
    return preds




if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    """
    データセット準備
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    #カレントディレクトリをファイルの場所に設定
    data_dir = os.path.join(os.path.dirname(__file__), 'data')  #データディレクトリを取得

    split_pos = 100                     #教師データと検証データの切り分け位置
    rawData = LoadCsv("data.csv")
    trainData = rawData[:split_pos, :]
    valData = rawData[split_pos:, :]
    testData = LoadCsv("test.csv")

    #データセットへ変換する
    trainSet = ConvertWindowDataset(trainData, 10, 100)
    valSet = ConvertWindowDataset(valData, 10, 100)
    testSet = ConvertWindowDataset(testData, 10, 1)

    """
    モデル構築
    """
    depth_x = 1     #入力する特徴量の数
    depth_t = 1     #出力する特徴量の数

    model = Transformer(depth_x,
                        depth_t,
                        N=3,                #エンコーダー/デコーダーブロックの内部レイヤー数
                        h=1,                #MultiHeadAttention入力段数(入力特徴量数の公約数に設定)
                        d_model=depth_x,    #入力特徴量数を設定しておく
                        d_ff=256,
                        maxlen=20)

    """
    学習処理
    """
    criterion = tf.losses.CategoricalCrossentropy()
    optimizer = optimizers.Adam(learning_rate=0.001,
                                beta_1=0.9, beta_2=0.999, amsgrad=True)
    train_loss = metrics.Mean()
    val_loss = metrics.Mean()

    epochs = 30

    for epoch in range(epochs):
        print('-' * 20)
        print('epoch: {}'.format(epoch+1))

        for (x, t) in trainSet:
            train_step(train_loss, x, t, depth_t)

        for (x, t) in valSet:
            val_step(val_loss, x, t, depth_t)
        
        #TODO: 途中経過のモデルの保存処理を追加

        print('loss: {:.3f}, val_loss: {:.3}'.format(
            train_loss.result(),
            val_loss.result()
        ))

        for idx, (x, t) in enumerate(testSet):
            preds = test_step(x)

            #TODO: グラフ出力等の処理を追加

