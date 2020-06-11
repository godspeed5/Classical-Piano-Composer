""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import converter, instrument, note, stream, chord
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
def generate():

    network_input = prepare_sequences()
    model = create_network(network_input)
    prediction = model.predict(network_input, verbose=0)
    print(prediction.shape)
    prediction1 = numpy.sum(prediction, axis =0)
    print(prediction1/numpy.sum(prediction))

def prepare_sequences():
    """ Prepare the sequences used by the Neural Network """
    notes = []
    network_output = []
    sequence_length = 100
    file = 'data1/parker_Celerity.mid'
    notes1 = []
    midi = converter.parse(file)

    print("Parsing %s" % file)

    notes_to_parse = None

    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse() 
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
            notes1.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
            notes1.append('.'.join(str(n) for n in element.normalOrder))

        
     # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    # n_vocab = len(set(notes))


    # create input sequences and the corresponding outputs
    for i in range(0, len(notes1) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    return network_input

def create_network(network_input):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Load the weights to each node
    model.load_weights('weights-withval-disc-58-0.0902.hdf5')

    return model
if __name__ == '__main__':
    generate()