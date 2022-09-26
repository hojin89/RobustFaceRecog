'''
ConvLSTM
'''

import tensorflow as tf

class ConvLSTM3(tf.keras.Model):
    def __init__(self, input_size, num_classes, num_timesteps=10):
        super(ConvLSTM3, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        self.convlstm1 = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu', data_format='channels_last')
        self.maxpool1 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.convlstm2 = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu', data_format='channels_last')
        self.maxpool2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.convlstm3 = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu', data_format='channels_last')
        self.maxpool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.avgpool = tf.keras.layers.GlobalAvgPool3D()
        self.linear = tf.keras.layers.Dense(self.num_classes)

    def call(self, input_tensor, training=False):
        x = tf.repeat(tf.expand_dims(input_tensor, axis=1), repeats=self.num_timesteps, axis=1)
        x = self.convlstm1(x)
        x = self.maxpool1(x)
        x = self.ln1(x, training=training)

        x = self.convlstm2(x)
        x = self.maxpool2(x)
        x = self.ln2(x, training=training)

        x = self.convlstm3(x)
        x = self.maxpool3(x)
        x = self.ln3(x, training=training)

        x = self.avgpool(x)
        x = self.linear(x)

        return x
