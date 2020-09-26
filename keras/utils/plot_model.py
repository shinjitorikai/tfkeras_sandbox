import tensorflow as tf
from keras.utils import plot_model # モデル可視化用

# 実行にはpydotパッケージが必要

# model定義
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# modelの可視化
#  - Jupyterの場合はto_fileがなくてもそのままインライン出力できる(はず)
plot_model(model, # 構築したモデルを指定
    show_shapes=True, # グラフ中に出力のshapeを表示するかどうか
    show_layer_names=True, # グラフ中にレイヤー名を表示するかどうか
    expand_nested=False, # グラフ中にネストしたモデルをクラスタに展開するかどうか
    dpi=96, # 画像のdpi
    to_file='./keras/utils/plot_model.png')
