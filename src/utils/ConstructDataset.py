import tensorflow as tf

def LoadCsv(path):
	"""
	CSVファイルのロードを行う
	戻り値：値配列
	"""
	import pandas as pd
	csv_frame = pd.read_csv(path)
	data = csv_frame.values
	return data

def ConvertWindowDataset(series, window_size, batch_size, shuffle_buffer=1000):
	"""
	WindowDatasetを作成する
	"""
	dataset = tf.data.Dataset.from_tensor_slices(series)		#系列をデータセット化する
	dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)		#WindowDatasetに変換する
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))	#配列の配列状態なので、2次元配列へ圧縮する
	dataset = dataset.shuffle(1000)		#シャッフルする
	dataset = dataset.map(lambda window: (window[:-3], window[-3:-1], window[-1]))		#データセットをエンコーダー入力、デコーダー入力、デコーダー出力(ラベル)に分ける)
	#dataset = dataset.map(lambda window: (window[:-1], window[-1]))		#[参考]2つに分ける場合
	dataset = dataset.batch(batch_size)		#バッチサイズを設定する
	dataset = dataset.prefetch(1)			#前のバッチが処理されているときに次のバッチを読み込むようにする
	return dataset

#データセットを結合する時はappendメソッドを使用する
#labeled_data_sets.append(labeled_dataset)


