#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import mido
import matplotlib.pyplot as plt
from torch import LongTensor
import torch.nn.functional as F
from random import randint, sample
from sys import exit
import os
import glob


# In[2]:


#creating the vocabulary

# vocabulary as described in Oore et al, 2018

'''
short summary of the vocabulary
vocabulary = [0<pad>,note_on, note_off, time_shift, velocity,414<start>,415<end>]

note_on = 1 - 128 (128 events) - when a note starts
note_off = 129-256 (128 events) - when a note stops
time_shifts = 257-381 (125 events) - units to represent time btw events. one unit = 8ms,can go upto 1s(thus 125 time shift events)
velocity = 382 - 413 (32 events) - how hard key is pressed

1 padding at zeroth index 
<start><end> at 414th and 414th  
'''

note_on_events = 128
note_off_events = 128
note_events = note_on_events + note_off_events
time_shift_events = 125
velocity_events = 32

max_time_btw_events = 1000 # ms (1s)
one_unit_time = max_time_btw_events//time_shift_events
vel_level = 128//velocity_events # midi has 0-127 velcoity range. we decided with 32 events. So a velocity range of approx 4 original midi values

total_midi_events = note_events + time_shift_events + velocity_events #leaving padding


# creating the vocabulary
note_on_vocab = [f"note_on_{i}" for i in range(note_on_events)]
note_off_vocab = [f"note_off_{i}" for i in range(note_off_events)]
time_shift_vocab = [f"time_shift_{i}" for i in range(time_shift_events)]
velocity_vocab = [f"set_velocity_{i}" for i in range(velocity_events)]

vocab = ['<pad>'] + note_on_vocab + note_off_vocab + time_shift_vocab + velocity_vocab + ['<start>', '<end>']

vocab_size = len(vocab)

pad_token = vocab.index("<pad>") #0
start_token = vocab.index("<start>") #414
end_token = vocab.index("<end>") #415


# In[3]:


#helper functions

def events_to_indices(event_list, _vocab=None):
    """
    converts event_list to list of indices in vocab
    """
    if _vocab is None:
        _vocab = vocab
    index_list = []
    for event in event_list:
        index_list.append(_vocab.index(event))
    return index_list


def indices_to_events(index_list, _vocab=None):
    """
    converts index_list to list of events in vocab
    """
    if _vocab is None:
        _vocab = vocab
    event_list = []
    for idx in index_list:
        event_list.append(_vocab[idx])
    return event_list

def velocity_to_bin(velocity, step=vel_level):
    """
    Velocity in midi is an int between 0 and 127 inclusive, which is unnecessarily high resolution
    To reduce number of events in vocab, velocity is resolved into (128 / step) bins

    Returns:
        _bin (int): bin into which velocity is placed
    """
    if 128 % step != 0:
        raise ValueError("128 must be divisible by bins")
    if not (0 <= velocity <= 127):
        raise ValueError(f"velocity must be between 0 and 127, not {velocity}")

    # return bin into which velocity is placed
    _bin = velocity // step
    return int(_bin)

def bin_to_velocity(_bin,step=vel_level):
    """
    finds the equivalent velocity from the corresponding bin number
    """
    if not(0 <= _bin*step <= 127):
        raise ValueError(f"bin size must be between 0 and 127, not {_bin}")

    return int(_bin*step)
                         


# In[13]:


def round_(a):
    """
    Custom rounding function for consistent rounding of 0.5 to greater integer
    """
    b = a // 1
    decimal_digits = a % 1
    adder = 1 if decimal_digits >= 0.5 else 0
    return int(b + adder)

def time_cutter(time, lth=max_time_btw_events, div=one_unit_time):
    """ 
    The time between events can be expressed as k instances of a maximum time shift followed by a leftover time shift
    time = k * max_time_shift + leftover_time_shift
    where k = time // max_time_shift; leftover_time_shift = time % max_time_shift
    """
    
    if lth % div != 0:
        raise ValueError("Max time must be divisible by the unit of time you consider")
    
    time_shifts = []
    k = time // lth
    time_shifts = [round_(lth/div) for _ in range(k)]
    
    leftover_time_shift = round_((time % lth) / div)
    if leftover_time_shift > 0:
        time_shifts.append(leftover_time_shift)
    
    return time_shifts

def time_to_events(time_btw_events, event_list=None, index_list=None, _vocab=None):
    """
    time between events are converted into time_shifts into _vocab using time_cutter
    event_list, index_list are passed by reference.
    """

    if _vocab is None:
        _vocab = vocab

    time = time_cutter(time_btw_events)

    for i in time:
        idx = note_events + i
        if event_list is not None:
            event_list.append(_vocab[idx])
        if index_list is not None:
            index_list.append(idx)
    return


# In[5]:



# In[6]:


#tokenizer from now on
def midi_parser(path_to_midi=None, mid=None):
    """
    Translates a single-track midi file into representation given by Oore et al
    args: path_to_midi (str) - path to midi file to load
          mid (mido.MidiFile) - loaded midi file
    returns: index_list (torch.Tensor): list of indices in vocab
             event_list (list) : list of events in vocab
             tempo (int) : tempo of the midi file
    """
    if not ((path_to_midi is None) ^ (mid is None)):
        raise ValueError("Input one of path_to_midi or mid, not both or neither")
    if path_to_midi is not None:
        try:
            mid = mido.MidiFile(path_to_midi)  # load the midi file
        except mido.midifiles.meta.KeySignatureError as e:
            raise ValueError(e)

    time_btw_msgs = 0  # time between midi messages
    event_list = []  # list of events
    index_list = []  # list of indices
    pedal_events = {}  # dict to handle pedal events
    pedal_flag = False  # flag to handle pedal events
    tempo = 0

    # converting to event list
    for track in mid.tracks:
        for msg in track:
            time_btw_msgs += msg.time  # how long the message lasts?

            if msg.is_meta:
                if (msg.type == "set_tempo") and (tempo == 0):
                    tempo = msg.tempo
                continue

            t = msg.type
            if t == "note_on" and msg.velocity > 0:  # key pressed
                idx = msg.note + 1
                vel = velocity_to_bin(msg.velocity)
            elif t == "note_off" or (t == "note_on" and msg.velocity == 0):  # key released
                note = msg.note
                if pedal_flag:
                    if note not in pedal_events:
                        pedal_events[note] = 0
                    pedal_events[note] += 1
                    continue
                else:  # else get idx to append to output lists
                    idx = note_on_events + note + 1
            elif t == "control_change":  # if pedal on or off and pedal_events is not empty
                if msg.control == 64:  # damper pedal
                    if msg.value >= 64:
                        # pedal down
                        pedal_flag = True
                    elif pedal_events:
                        # lift pedal
                        pedal_flag = False
                        # add the time events (0 is not a time shift, so all notes lifted at once is ok)
                        time_to_events(time_btw_msgs, event_list=event_list, index_list=index_list)
                        time_btw_msgs = 0
                        # perform note_offs that occurred when pedal was down now that pedal is up
                        for note in pedal_events:
                            idx = note_on_events + note + 1
                            # append a note_off event for all times note was released
                            for i in range(pedal_events[note]):
                                event_list.append(vocab[idx])
                                index_list.append(idx)
                        # restart pedal events dict
                        pedal_events = {}
                # to prevent adding more events to output lists, continue
                continue
            else:
                # if it's not a type of msg we care about, continue to avoid adding to output lists
                continue

            time_to_events(time_btw_msgs, event_list=event_list, index_list=index_list)
            time_btw_msgs = 0  # reset time_btw_msgs to process subsequent messages

            # append velocity if note_on
            if t == "note_on" and msg.velocity > 0:
                event_list.append(vocab[note_on_events + note_off_events + time_shift_events + vel + 1])
                index_list.append(note_on_events + note_off_events + time_shift_events + vel + 1)

            # append event and idx note events
            event_list.append(vocab[idx])
            index_list.append(idx)

    # return the lists of events
    return LongTensor(index_list), event_list, tempo
    


# In[7]:


def list_parser(index_list=None, event_list=None, fname="bloop", tempo=512820):
    """
    Translates a set of events or indices in the vocabulary into a midi file

    Args:
        index_list (list or torch.Tensor): list of indices in vocab OR
        event_list (list): list of events in vocab
        fname (str, optional): name for single track of midi file returned
        tempo (int, optional): tempo of midi file returned in µs / beat,
                               tempo_in_µs_per_beat = 60 * 10e6 / tempo_in_bpm

    Returns:
        mid (mido.MidiFile): single-track piano midi file translated from vocab
                             NOTE: mid IS NOT SAVED BY THIS FUNCTION, IT IS ONLY RETURNED
    """
    # take only one of event_list or index_list to translate
    if not ((index_list is None) ^ (event_list is None)):
        raise ValueError("Input one of index_list or event_list, not both or neither")

    # check index_list is ints, assuming 1d list
    if index_list is not None:
        try:
            # assume torch tensor
            if not all([isinstance(i.item(), int) for i in index_list]):
                raise ValueError("All indices in index_list must be int type")
        except AttributeError:
            # otherwise assume normal ,jst
            if not all([isinstance(i, int) for i in index_list]):
                raise ValueError("All indices in index_list must be int type")

    # check event_list is str, assuming 1d list and convert to index_list
    if event_list is not None:
        if not all(isinstance(i, str) for i in event_list):
            raise ValueError("All events in event_list must be str type")
        index_list = events_to_indices(event_list)

    # set up midi file
    mid = mido.MidiFile()
    meta_track = mido.MidiTrack()
    track = mido.MidiTrack()

    # meta messages; meta time is 0 everywhere to prevent delay in playing notes
    meta_track.append(mido.MetaMessage("track_name").copy(name=fname, time=0))
    meta_track.append(mido.MetaMessage("smpte_offset"))
    # assume time_signature is 4/4
    time_sig = mido.MetaMessage("time_signature")
    time_sig = time_sig.copy(numerator=4, denominator=4, time=0)
    meta_track.append(time_sig)
    # assume key_signature is C
    key_sig = mido.MetaMessage("key_signature", time=0)
    meta_track.append(key_sig)
    # assume tempo is constant at input tempo
    set_tempo = mido.MetaMessage("set_tempo")
    set_tempo = set_tempo.copy(tempo=tempo, time=0)
    meta_track.append(set_tempo)
    # end of meta track
    end = mido.MetaMessage("end_of_track").copy(time=0)
    meta_track.append(end)

    # set up the piano; default channel is 0 everywhere; program=0 -> piano
    program = mido.Message("program_change", channel=0, program=0, time=0)
    track.append(program)
    # dummy pedal off message; control should be < 64
    cc = mido.Message("control_change", time=0)
    track.append(cc)

    # things needed for conversion
    delta_time = 0
    vel = 0

    # reconstruct the performance
    for idx in index_list:
        # if torch tensor, get item
        try:
            idx = idx.item()
        except AttributeError:
            pass
        # if pad token, continue
        if idx <= 0:
            continue
        # adjust idx to ignore pad token
        idx = idx - 1

        # note messages
        if 0 <= idx < note_on_events + note_off_events:
            # note on event
            if 0 <= idx < note_on_events:
                note = idx
                t = "note_on"
                v = vel  # get velocity from previous event
            # note off event
            else:
                note = idx - note_on_events
                t = "note_off"
                v = 127

            # create note message and append to track
            msg = mido.Message(t)
            msg = msg.copy(note=note, velocity=v, time=delta_time)
            track.append(msg)

            # reinitialize delta_time and velocity to handle subsequent notes
            delta_time = 0
            vel = 0

        # time shift event
        elif note_on_events + note_off_events <= idx < note_on_events + note_off_events + time_shift_events:
            # find cut time in range (1, time_shift_events)
            cut_time = idx - (note_on_events + note_off_events - 1)
            # scale cut_time by one_unit_time (from vocabulary) to find time in ms; add to delta_time
            delta_time += cut_time * one_unit_time

        # velocity event
        elif note_on_events + note_off_events + time_shift_events <= idx < total_midi_events:
            # get velocity for next note_on in range (0, 127)
            vel = bin_to_velocity(idx - (note_on_events + note_off_events + time_shift_events))

    # end the track
    end = mido.MetaMessage("end_of_track").copy(time=0)
    track.append(end)

    # append finalized track and return midi file
    mid.tracks.append(meta_track)
    mid.tracks.append(track)
    return mid


# In[8]:


def sample_end_data(seqs, length_per_seq, factor=6):
    """
    This function is designed to sample sequences from the end of each input sequence, with some randomness in the starting point.
    This is done to ensure that the model learns how to end sequences properly. 
    args: seqs - list of sequences in the event vocabulary
          length_per_seq - an integer representing the approximate length to cut sequences into
          factor - an optional parameter with a default value of 6, used to vary the range of output lengths
    """
    data = []
    for seq in seqs:
        lower_bound = max(len(seq) - length_per_seq, 0)
        idx = randint(lower_bound, lower_bound + length_per_seq // factor)
        data.append(seq[idx:])
    return data


# In[9]:


def sample_data(seqs, lth, factor=6):
    """
    creates a dataset of shorter sequences from the full MIDI sequences
    Returns a list.

    Args:
        seqs (list): list of sequences in the event vocabulary
        lth (int): approximate length to cut sequences into
        factor (int): factor to vary range of output lengths; Default: 6. Higher factor will narrow the output range

    Returns:
        input sequs cut to length ~lth
    """
    data = []
    for seq in seqs:
        length = randint(lth - lth // factor, lth + lth // factor)
        idx = randint(0, max(0, len(seq) - length))
        data.append(seq[idx:idx+length])
        
    return data


# In[10]:


def aug(data, note_shifts=None, time_stretches=None, verbose=False):
    """
    Augments data up and down in pitch by note_shifts and faster and slower in time by time_stretches. Adds start
    and end tokens and pads to max sequence length in data

    Args:
        data (list of lists of ints): sequences to augment
        note_shifts (list): pitch transpositions to be made
        time_stretches (list): stretches in time to be made
        verbose (bool): set to True to periodically print augmentation progress

    Returns:
        input data with pitch transpositions and time stretches, concatendated to one tensor
    """
    if note_shifts is None:
        note_shifts = torch.arange(-2, 3)
    if time_stretches is None:
        time_stretches = [1, 1.05, 1.1]
    if any([i <= 0 for i in time_stretches]):
        raise ValueError("time_stretches must all be positive")

    # preprocess the time stretches
    if 1 not in time_stretches:
        time_stretches.append(1)
    ts = []
    for t in time_stretches:
        ts.append(t) if t not in ts else None
        ts.append(1 / t) if (t != 1 and 1 / t not in ts) else None
    ts.sort()
    time_stretches = ts

    # iteratively transpose and append the sequences
    note_shifted_data = []
    count = 0  # to print if verbose
    for seq in data:
        # data will be transposed by each shift in note_shifts
        for shift in note_shifts:
            # check torch tensor
            try:
                _shift = shift.item()
            except AttributeError:
                _shift = shift

            # iterate over and shift seq
            note_shifted_seq = []
            for idx in seq:
                _idx = idx + _shift  # shift the index

                # append only note values if changed, and don't go out of bounds of note events
                if (0 < idx <= note_on_events and 0 < _idx <= note_on_events) or \
                        (note_on_events < idx <= note_events and note_on_events < _idx <= note_events):
                    note_shifted_seq.append(_idx)
                else:
                    note_shifted_seq.append(idx)
            # verbose statement
            count += 1
            print(f"Transposed {count} sequences") if verbose else None
            # convert to tensor and append to data
            note_shifted_seq = torch.LongTensor(note_shifted_seq)
            note_shifted_data.append(note_shifted_seq)

    # now iterate over the note shifted data to stretch it in time
    time_stretched_data = []
    delta_time = 0  # helper
    count = 0  # to print if verbose
    for seq in note_shifted_data:
        # data will be stretched in time by each time_stretch
        for time_stretch in time_stretches:
            # iterate over and stretch time shift events in seq
            time_stretched_seq = []
            for idx in seq:
                if note_events < idx <= note_events + time_shift_events:
                    # accumulate stretched times
                    time = idx - (note_events - 1)
                    delta_time += round_(time * one_unit_time * time_stretch)
                else:
                    time_to_events(delta_time, index_list=time_stretched_seq)
                    delta_time = 0
                    time_stretched_seq.append(idx)
            # verbose statement
            count += 1
            print(f"Stretched {count} sequences") if verbose else None
            # convert to tensor and append to data
            time_stretched_seq = torch.LongTensor(time_stretched_seq)
            time_stretched_data.append(time_stretched_seq)

    # preface and suffix with start and end tokens
    aug_data = []
    for seq in time_stretched_data:
        aug_data.append(F.pad(F.pad(seq, (1, 0), value=start_token), (0, 1), value=end_token))

    # pad all sequences to max length
    aug_data = torch.nn.utils.rnn.pad_sequence(aug_data, padding_value=pad_token).transpose(-1, -2)
    return aug_data


def randomly_sample_aug_data(aug_data, k, augs=25):
    """
    Randomly samples k sets of augmented data to cut down dataset

    Args:
        aug_data (torch.Tensor): augmented dataset
        k (int): coefficient such that k * augs samples are returned
        augs (int): total number of augmentations per sequence performed on original dataset
    """
    random_indices = sample(range(len(aug_data) // augs), k=k)
    out = torch.cat(
        [t[i * augs:i * augs + augs] for i in random_indices],
        dim=0
    )
    return out




# In[1]:


# uncomment and run below code if you want to do the conversion of midi files to input sequential data


# In[ ]:


# import os
# import torch
# import glob

# SOURCE = "maestro-v3.0.0-midi/maestro-v3.0.0/"
# DESTINATION_DIR = "Sequential-input/"
# FILENAME = "sqip.pt"
# LENGTH = 512
# FROM_AUGMENTED_DATA = False
# TRANSPOSITIONS = [-5, -1, 0, 3, 6]
# TIME_STRETCHES = [0.95, 1, 1.05]
# VERBOSE = True

# os.makedirs(DESTINATION_DIR, exist_ok=True)
# DESTINATION = os.path.join(DESTINATION_DIR, FILENAME)

# DATA = []

# if not FROM_AUGMENTED_DATA:
#     print("Translating MIDI files to event vocabulary (NOTE: may take a while)...")
#     for file in glob.iglob(os.path.join(SOURCE, '**', '*.mid*'), recursive=True):
#         if VERBOSE:
#             print(f"Processing: {file}")
#         try:
#             idx_list = midi_parser(path_to_midi=file)[0]
#             DATA.append(idx_list)
#         except (OSError, ValueError, EOFError) as ex:
#             print(f"{type(ex).__name__} was raised: {ex}")
#     print(f"Number of MIDI files successfully processed: {len(DATA)}")
#     print("Done!")

# if len(DATA) == 0:
#     raise ValueError("No MIDI files were successfully processed. Check your SOURCE path and MIDI files.")

# print("Randomly sampling and cutting data to length...")
# sampled_data = sample_data(DATA, lth=LENGTH)
# end_data = sample_end_data(DATA, length_per_seq=LENGTH)
# print(f"Number of sequences after sampling: {len(sampled_data) + len(end_data)}")
# DATA = sampled_data + end_data
# print("Done!")

# if len(DATA) == 0:
#     raise ValueError("No sequences remained after sampling. Check your LENGTH parameter and sampling functions.")

# if not FROM_AUGMENTED_DATA:
#     print("Augmenting data (NOTE: may take even longer)...")
#     try:
#         DATA = aug(DATA, note_shifts=TRANSPOSITIONS, time_stretches=TIME_STRETCHES,
#                    verbose=(VERBOSE >= 2))
#         print(f"Shape of augmented data: {DATA.shape}")
#     except Exception as e:
#         print(f"Error during augmentation: {str(e)}")
#         print(f"Type of DATA: {type(DATA)}")
#         print(f"Length of DATA: {len(DATA)}")
#         if len(DATA) > 0:
#             print(f"Type of first element in DATA: {type(DATA[0])}")
#             print(f"Shape of first element in DATA: {DATA[0].shape}")
#         raise
#     print("Done!")

# DATA = DATA[torch.randperm(DATA.shape[0])]  # shuffle

# print("Saving...")
# torch.save(DATA, DESTINATION)
# print("Done!")

# print(f"Preprocessed data saved to: {DESTINATION}")

