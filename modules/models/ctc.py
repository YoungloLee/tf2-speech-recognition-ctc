from tensorflow.keras.layers import Dense, Input, Bidirectional, Masking, GRU, LayerNormalization
from tensorflow.keras.models import Model

from glob import glob
import os


def find_files(directory, pattern='**/*.h5'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


class SpeechNetwork:
    def __init__(self, save_dir, args):
        self.args = args
        self.save_dir = save_dir
        self.model = None
        self.generate_model()

    def generate_model(self):
        # Input
        input = Input(shape=[None, self.args.num_mels], name='input_speech')
        x = Masking(mask_value=0, input_shape=(None, self.args.num_mels))(input)

        # Bidirectional LSTM
        for b in range(self.args.num_lstm_layers):
            x = Bidirectional(GRU(units=self.args.num_units_per_lstm//2,
                                  return_sequences=True,
                                  recurrent_initializer='glorot_uniform'))(x)
            x = LayerNormalization(epsilon=1e-6)(x)

        # output layer with softmax
        logit = Dense(self.args.num_classes, name='speech_encoder', activation='softmax')(x)

        # Generate model
        self.model = Model(inputs=input, outputs=logit)
