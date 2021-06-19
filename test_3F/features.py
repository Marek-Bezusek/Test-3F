"""
This script uses the Parselmouth library. 
Common function arguments:
    sound: Sound object of parselmouth module
    vowel_a, vowel_i, vowel_u: Sound objects of parselmouth module
    F1a, F2a, F1i, F2i, F1u, F2u: Formants (F1, F2) of vowel [a], [i], [u] 
    pauses: numpy array of pause durations and puase start/end time (pauses in columns, rows: start, end, duration)
    word_time_stamps: numpy array: start/end time for each word (words in columns, rows: [1]start time, [2]end time)
"""
import math
import parselmouth
from parselmouth.praat import call
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from test_3F.audio_preprocessing import segmentation
import re
import os
from test_3F.utils import THIS_DIR
"""
EXPIRATION
Maximum Expiration Time (MET)
Standard deviation of expiration intervals (EISD) (Test 3F 4.4: ss-ss-ss)
SEO skewness
SEO kurtosis
"""


def maximum_expiration_time(sound):
    return sound.duration


def expiration_interval_SD(sound,
                           silence_treshold=-25.0,
                           min_silent_interval_duration=0.1,
                           min_sounding_interval_duration=0.05):
    intensity = sound.to_intensity()
    textgrid = call(intensity, "To TextGrid (silences)", silence_treshold, min_silent_interval_duration,
                    min_sounding_interval_duration, "silent", "sounding")
    intervals = call([sound, textgrid], "Extract intervals where...", 1, False, "is equal to", "sounding")

    durations = np.zeros(len(intervals))
    for i, interval in enumerate(intervals):
        durations[i] = interval.duration

    duration_sd = np.std(durations)
    return duration_sd


def SEO_kurtosis(sound):
    intensity = sound.to_intensity().values.T
    return kurtosis(intensity)


def SEO_skewness(sound):
    intensity = sound.to_intensity().values.T
    return skew(intensity)


"""
PHONATION /////////////////////////////////////
F2i/F2u
relF0SD
Jitter (local, absolute,RAP, PPQ5 DDP)
Shimmer (local, local dB, APQ3, APQ5, APQ11, DDA)
HNR (harmonic to noise ratio)
DUV (degree of unvoiced segments)
SSD (segmental signal to dysperiodicity ratio)
F0 range in semitones
"""


def F2i_F2u(vowel_i, vowel_u):
    formants_i = vowel_i.to_formant_burg(max_number_of_formants=2)
    formants_u = vowel_u.to_formant_burg(max_number_of_formants=2)
    F2i = call(formants_i, "Get mean", 2, 0, 0, "Hertz")
    F2u = call(formants_u, "Get mean", 2, 0, 0, "Hertz")
    return F2i / F2u


def relF0SD(sound):
    pitch = sound.to_pitch()
    F0SD = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
    return F0SD / meanF0


def jitter(sound,
           # Jitter optional arguments
           time_start=0,  # (seconds)
           time_end=0,  # (seconds) 0 = all
           shortest_period=0.0001,  # (seconds)
           longest_period=0.02,  # (seconds)
           maximum_period_factor=1.3):
    point_process = call(sound, "To PointProcess (periodic, cc)...", 75, 600)

    jitter_local = call(point_process, "Get jitter (local)...", time_start, time_end, shortest_period, longest_period,
                        maximum_period_factor)
    jitter_absolute = call(point_process, "Get jitter (local, absolute)...", time_start, time_end, shortest_period,
                           longest_period, maximum_period_factor)
    jitter_rap = call(point_process, "Get jitter (rap)...", time_start, time_end, shortest_period, longest_period,
                      maximum_period_factor)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)...", time_start, time_end, shortest_period, longest_period,
                       maximum_period_factor)
    jitter_ddp = call(point_process, "Get jitter (ddp)...", time_start, time_end, shortest_period, longest_period,
                      maximum_period_factor)

    return jitter_local, jitter_absolute, jitter_rap, jitter_ppq5, jitter_ddp


def shimmer(sound,
            # Shimmer optional arguments
            time_start=0,  # (seconds)
            time_end=0,  # (seconds) 0 = all
            shortest_period=0.0001,  # (seconds)
            longest_period=0.02,  # (seconds)
            maximum_period_factor=1.3,
            maximum_amplitude_factor=1.6):
    point_process = call(sound, "To PointProcess (periodic, cc)...", 75, 600)

    shimmer_local = call([sound, point_process], "Get shimmer (local)", time_start, time_end, shortest_period,
                         longest_period, maximum_period_factor, maximum_amplitude_factor)
    shimmer_local_db = call([sound, point_process], "Get shimmer (local_dB)", time_start, time_end, shortest_period,
                            longest_period, maximum_period_factor, maximum_amplitude_factor)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", time_start, time_end, shortest_period,
                        longest_period, maximum_period_factor, maximum_amplitude_factor)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", time_start, time_end, shortest_period,
                        longest_period, maximum_period_factor, maximum_amplitude_factor)
    shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", time_start, time_end, shortest_period,
                         longest_period, maximum_period_factor, maximum_amplitude_factor)
    shimmer_dda = call([sound, point_process], "Get shimmer (dda)", time_start, time_end, shortest_period,
                       longest_period, maximum_period_factor, maximum_amplitude_factor)

    return shimmer_local, shimmer_local_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda


def harmonic_to_noise_ratio(sound):
    HNR = sound.to_harmonicity()
    meanHNR = call(HNR, "Get mean", 0, 0)
    return meanHNR


def degree_of_unvoiced_segments(sound):
    """
    DUV from parselmouth
    """
    pitch = sound.to_pitch()
    num_frames = call(pitch, "Get number of frames")
    num_voiced_frames = pitch.count_voiced_frames()
    DUV = (num_frames - num_voiced_frames) / num_frames
    return DUV


def degree_of_unvoiced_segments_report(sound):
    """
    DUV from praat voice report 
    """
    # Get voice report
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    voice_report = parselmouth.praat.call([sound, pitch, pulses], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03,
                                          0.45)
    # Get fraction of unvoiced frames from report
    ind_start = voice_report.index("Fraction of locally unvoiced frames: ") + len(
        "Fraction of locally unvoiced frames: ")
    ind_end = voice_report.index("%", ind_start)
    DUV = float(voice_report[ind_start:ind_end]) / 100
    return DUV


def segmental_signal_to_dysperiodicity_ratio(sound, pitch_lo=75, pitch_up=350):
    # Segment input
    fs = sound.get_sampling_frequency()
    window_length = int(fs * 0.025)
    y, y_num_segments = segmentation(sound.values, math.ceil(2 * (fs / pitch_lo) + window_length),
                                     math.ceil(fs / pitch_lo + window_length))

    # Calculate segmental signal-to-dysperiodicity ratio
    sdr = np.zeros((y_num_segments, 1))
    for i in range(y_num_segments):
        # Get the center frame
        x = y[math.floor((fs / pitch_lo)):math.floor((fs / pitch_lo) + window_length), i]

        # Get the optimal lag
        yy = segmentation(y[:, i], len(x), len(x) - 1)[0]

        ax = np.arange(0, yy.shape[1]) - math.floor((1 / pitch_lo) * fs)
        brd_left = np.where(ax == math.floor(-(1 / pitch_up) * fs))[0]
        brd_right = np.where(ax == math.ceil((1 / pitch_up) * fs))[0]

        yy = np.delete(yy, np.arange(brd_left, brd_right), axis=1)
        xx = np.tile(x, (yy.shape[1], 1)).T

        alpha = np.sqrt(np.sum(xx ** 2, axis=0) / np.sum(yy ** 2, axis=0))
        alpha = np.tile(alpha, (yy.shape[0], 1))

        tmp = np.sum((xx - alpha * yy) ** 2, axis=0)
        pos = np.where(tmp == np.min(tmp))[0]  # get index of minimum val
        # t_opt = ax[pos]

        # Get the dysperiodicity trace
        x = x.reshape(len(x), 1)
        e = x - alpha[:, pos] * yy[:, pos]

        # Segment the center frame and the dysperiodicity trace to 5 ms
        xxx = segmentation(x, 0.005 * fs)[0]
        eee = segmentation(e, 0.005 * fs)[0]

        # Signal-to-dysperiodicity ratio
        sdr[i] = np.mean(10 * np.log10(sum(xxx ** 2) / sum(eee ** 2)))

    return np.mean(sdr)


def F0_range_semitones(sound):
    pitch = sound.to_pitch()
    F0_min = call(pitch, "Get minimum", 0, 0, "semitones re 100 Hz", "Parabolic")
    F0_max = call(pitch, "Get maximum", 0, 0, "semitones re 100 Hz", "Parabolic")
    return F0_max - F0_min


"""
PROSODY /////////////////////////////////////
relF0VR
relSEOVR
relSEOSD

(Get word timestamps)
(Get pauses)
Percent Pause Time (PPT)
Number of pauses (NoP)
Speech index of rhythmicity (SPIR)
Ratio of Intra-Word Pauses (RIWP)

Rhythm similarity (RS) (Comparison of the spectrogram of a speech recording with a reference using the DTW algorithm)
Articulation rate (AR)
"""


def relF0VR(sound, time_step=None, pitch_floor=75.0):
    pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor)

    # Get voiced segments
    F0_values = pitch.selected_array['frequency']
    mask = ~np.asarray(F0_values, bool)
    F0_array = pitch.to_array()['frequency']
    F0_voiced = np.delete(F0_array, mask, 1)

    # Compute range of values in segments
    F0_voiced[F0_voiced == 0] = np.nan
    F0_max = np.nanmax(F0_voiced, axis=0)
    F0_min = np.nanmin(F0_voiced, axis=0)
    F0_range = F0_max - F0_min

    F0_range_mean = np.nanmean(F0_range)
    F0_range_sd = np.nanstd(F0_range)
    return F0_range_sd / F0_range_mean


def relSEOVR(sound):
    """
    returns: relative variation of intensity [dB(SPL)] 
    """
    # Segment intensity
    sampling_period = sound.get_sampling_period()
    sampling_rate = sound.get_sampling_frequency()
    intensity = sound.to_intensity(time_step=sampling_period).values
    segmented_intensity, num_frames = segmentation(intensity, window_length=int(sampling_rate * 0.03))

    # Compute range of values in segments
    segmented_intensity[segmented_intensity == 0] = np.nan
    intensity_max = np.nanmax(segmented_intensity, axis=0)
    intensity_min = np.nanmin(segmented_intensity, axis=0)
    intensity_range = intensity_max - intensity_min

    intensity_range_mean = np.mean(intensity_range)
    intensity_range_sd = np.std(intensity_range)
    return intensity_range_sd / intensity_range_mean


def relSEOSD(sound):
    """
    returns: relative standard deviation of intensity [dB(SPL)] 
    """
    intensity = sound.to_intensity()
    SEOSD = call(intensity, "Get standard deviation", 0, 0)
    meanSEO = call(intensity, "Get mean", 0, 0, 'energy')
    return SEOSD / meanSEO


"""
Pauses //////////////
"""


def word_timestamps(sound, threshold_rel_median="auto", min_silence_dur=0.06, min_sounding_dur=0.05, window_hz=100):
    """
    Return:  numpy array. Start time and end time for each word. Words in columns, start/end time in rows.
    """
    intensity = sound.to_intensity(window_hz)
    intensity_max = parselmouth.praat.call(intensity, "Get maximum...", 0, 0, "none")
    intensity_median = parselmouth.praat.call(intensity, "Get quantile...", 0, 0, 0.5)
    if threshold_rel_median == "auto":
        threshold_rel_median = -int(call(intensity, "Get standard deviation", 0, 0)/2)
    threshold_rel_max = int(-intensity_max + intensity_median + threshold_rel_median)

    # Get words
    textgrid = parselmouth.praat.call(intensity, "To TextGrid (silences)", threshold_rel_max, min_silence_dur, min_sounding_dur, "silent", "sounding")
    silencetier = parselmouth.praat.call(textgrid, "Extract tier", 1)
    word_table = parselmouth.praat.call(silencetier, "Down to TableOfReal", "sounding")
    num_words = parselmouth.praat.call(word_table, "Get number of rows")
    word_timestamps = np.empty([2, num_words], dtype=float)
    for iword in range(num_words):
        word = iword + 1
        word_timestamps[0, iword] = parselmouth.praat.call(word_table, "Get value", word, 1)
        word_timestamps[1, iword] = parselmouth.praat.call(word_table, "Get value", word, 2)

    return word_timestamps

#%%
def get_pauses(sound, min_pause_duration=0.06):
    """
    Returns: numpy array: pauses in columns:
                        1st row: pause start time, 2nd row: pause end time, 3rd row: pause duration
    """
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

    # Get number of glottal pulses
    n_pulses = parselmouth.praat.call(pulses, "Get number of points")

    # Get pauses in between pulses
    pauses = np.empty((3, 1), float)
    for i in range(1, n_pulses):
        pause_start = parselmouth.praat.call(pulses, "Get time from index", i)
        pause_end = parselmouth.praat.call(pulses, "Get time from index", i + 1)
        pause_duration = pause_end - pause_start

        if pause_duration > min_pause_duration:
            pause = np.stack((pause_start, pause_end, pause_duration), axis=0).reshape(3, 1)
            pauses = np.append(pauses, pause, axis=1)

    return pauses


def percent_pause_time(sound, pauses):
    # Sum pause durations
    total_pause_time = np.sum(pauses, axis=1)[2]
    return total_pause_time / sound.duration * 100


def number_of_pauses(pauses):
    return pauses.shape[1]


def speech_index_of_rhythmicity(sound, pauses, word_time_stamps):
    """
    returns: number of Intra-Word Pauses per minute
    """
    if word_time_stamps.ndim == 1: return np.nan
    # Count intra-word pauses
    num_intra_word_pauses = 0
    for pause in range(pauses.shape[1]):
        for word in range(word_time_stamps.shape[1]):
            if pauses[0, pause] > word_time_stamps[0, word] and pauses[1, pause] < word_time_stamps[1, word]:
                num_intra_word_pauses += 1

    return num_intra_word_pauses / sound.duration * 60


def ratio_of_intra_word_pauses(pauses, word_time_stamps):
    """
    returns: percentage of Intra-Word Pauses to Total Pause Time
    """
    if word_time_stamps.ndim == 1: return np.nan
    # Get duration of intra-word pauses
    duration_intra_word_pauses = 0.0
    for pause in range(pauses.shape[1]):
        for word in range(word_time_stamps.shape[1]):
            if pauses[0, pause] > word_time_stamps[0, word] and pauses[1, pause] < word_time_stamps[1, word]:
                duration_intra_word_pauses += pauses[2, pause]

    # Get duration of all pauses  
    total_pause_time = np.sum(pauses, axis=1)[2]
    return duration_intra_word_pauses / total_pause_time * 100


def rhythm_similarity(sound, reference_sound):
    sound_spec = sound.to_spectrogram()
    reference_sound_spec = reference_sound.to_spectrogram()
    dtw = call([sound_spec, reference_sound_spec], "To DTW...", False, False, "no restriction")
    dist = call(dtw, "Get distance (weighted)")
    return dist


def articulation_rate(sound, transcript):
    """
    Count syllables defined in file "5.4_syllable_list" from the speech transcript. Calculate articulation rate.
    transcript: string
    returns: (float) articulation rate, (int) number of syllables
    """
    transcript = transcript.lower()

    # Load list of syllables from text file
    syllables_filename = "5.4_syllable_list"
    syllables_filepath = os.path.join(THIS_DIR, os.pardir,"data", "test3f_tasks", syllables_filename + ".txt")
    with open(syllables_filepath, 'r', encoding='utf-8') as file:
        syllables = file.read().lower()
    syllables = re.split(r'-|\s', syllables)  # syllable delimiter
    syllables = [x for x in syllables if x]  # remove empty cells
    syllables = list(dict.fromkeys(syllables))  # remove duplicates

    # Count syllables
    num_syllables = 0
    for syllable in syllables:
        if syllable == "asi" and syllable in transcript:
            num_syllables += 2
        elif syllable == "te" and syllable in transcript:
            num_syllables += len(re.findall(syllable, transcript)) - len(re.findall("teď", transcript))
        else:
            num_syllables += transcript.count(syllable)

    AR = num_syllables / sound.duration
    return AR, num_syllables


"""
ARTICULATION /////////////////////////////////////
(Vowel Formants)
Vowel Space Area (VSA)
Vowel Articulation Index (VAI)
Formant Centralization Ratio (FCR)
Standard deviation of F1 relative to mean (relF1SD) 
Standard deviation of F2 relative to mean (relF2SD)
"""


def vowel_formants(vowel_a, vowel_i, vowel_u):
    """
    returns F1a, F2a, F1i, F2i, F1u, F2u
    """
    # Function arguments to dictionary
    args = locals()

    # Get first formant (F1) and second formant (F2) of vowel [a], [i], [u]
    formants = np.empty([2, 3])  # Vowels in columns, formants(F1 and F2) in rows
    for i, arg in enumerate(args):
        formant_object = args[arg].to_formant_burg(max_number_of_formants=2)
        formants[0, i] = call(formant_object, "Get mean", 1, 0, 0, "Hertz")  # F1
        formants[1, i] = call(formant_object, "Get mean", 2, 0, 0, "Hertz")  # F2

        #  F1a             F2a             F1i             F2i             F1u             F2u
    return formants[0, 0], formants[1, 0], formants[0, 1], formants[1, 1], formants[0, 2], formants[1, 2]


def vowel_space_area(F1a, F2a, F1i, F2i, F1u, F2u):
    """
    Compute area of the vowel triangle ([a], [i], [u]) in formant space (F1, F2)
    """
    # Compute triangle sides
    dist_iu = np.sqrt((F1i - F1u) ** 2 + (F2i - F2u) ** 2)
    dist_ia = np.sqrt((F1i - F1a) ** 2 + (F2i - F2a) ** 2)
    dist_au = np.sqrt((F1a - F1u) ** 2 + (F2a - F2u) ** 2)
    S = (dist_iu + dist_ia + dist_au) / 2  # semi-perimeter
    VSA = np.sqrt(S * (S - dist_iu) * (S - dist_ia) * (S - dist_au))  # triangle area
    return VSA


def vowel_articulation_index(F1a, F2a, F1i, F2i, F1u, F2u):
    return (F2i + F1a) / (F1i + F1u + F2u + F2a)


def formant_centralization_ratio(F1a, F2a, F1i, F2i, F1u, F2u):
    return (F1i + F1u + F2u + F2a) / (F2i + F1a)


def relF1SD(sound):
    F1 = sound.to_formant_burg(max_number_of_formants=1)
    F1SD = call(F1, "Get standard deviation", 1, 0, 0, "Hertz")
    meanF1 = call(F1, "Get mean", 1, 0, 0, "Hertz")
    return F1SD / meanF1


def relF2SD(sound):
    formants = sound.to_formant_burg(max_number_of_formants=2)
    F2SD = call(formants, "Get standard deviation", 2, 0, 0, "Hertz")
    meanF2 = call(formants, "Get mean", 2, 0, 0, "Hertz")
    return F2SD / meanF2
