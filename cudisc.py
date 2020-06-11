""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
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
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
# tf.config.gpu_options.allow_growth = True

def train_network():
    """ Train a Neural Network to generate music """
    notes, network_output, network_input = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():

    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    network_output = []
    sequence_length = 100

    for file in glob.glob("data1/*.mid"):
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
            if 'kris' in file:
                network_output.append([1, 0, 0])
            elif 'signe' in file:
                network_output.append([0, 1, 0])
            elif 'parker' in file:
                network_output.append([0, 0, 1])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # print(n_vocab)
    # print(network_output)

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    # network_output=network_output[0:37290]
    network_output = numpy.array(network_output)
    print(network_output.shape)
    print(network_output)


    return notes, network_output, network_input



def create_network(network_input, n_vocab):
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
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model = Sequential()
    # model.add(LSTM(
    #     512,
    #     input_shape=(network_input.shape[1], network_input.shape[2]),
    #     return_sequences=True
    # ))
    # model.add(BatchNorm())
    # model.add(Dropout(0.3))
    # model.add(Dense(3))
    # model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-withval-disc-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, validation_split=0.33, epochs=200, batch_size=10, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
