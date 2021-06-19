import os
import pandas as pd
import numpy as np
import parselmouth
from test_3F import features
from test_3F.utils import THIS_DIR, DATA_DIR, DATASET_DIR
from test_3F.utils import get_task_path, tasks_dictionary, read_all_tasks
#%% Tasks that wont be included in feature extraction
TASKS_TO_REMOVE = ["_0.", "_5.3", "_7.4", "_10."]

#%% Task 5.1
def task_5_1(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()
    # Get features
    subject_features["MET_5.1"] = (features.maximum_expiration_time(sound))
    return subject_features

#%% Task 5.2
def task_5_2(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()
    # Get features
    subject_features["MET_5.2"] = features.maximum_expiration_time(sound)
    return subject_features

#%% Task 7.1 (subtasks 1 to 4 (vowels: [a], [e], [i], [o], [u]))
def task_7_1(task_path_vowels, subject_features):
    # Load speech file
    vowel_a = parselmouth.Sound(task_path_vowels[0])
    vowel_i = parselmouth.Sound(task_path_vowels[2])
    vowel_u = parselmouth.Sound(task_path_vowels[4])
    vowel_a.pre_emphasize()
    vowel_i.pre_emphasize()
    vowel_u.pre_emphasize()

    # Get features
    formants = features.vowel_formants(vowel_a, vowel_i, vowel_u)
    subject_features["F2i/F2u_7.1"].append(features.F2i_F2u(vowel_i, vowel_u))
    subject_features["VSA_7.1"].append(features.vowel_space_area(*formants))
    subject_features["VAI_7.1"].append(features.vowel_articulation_index(*formants))
    subject_features["FCR_7.1"].append(features.formant_centralization_ratio(*formants))
    return subject_features

#%% Task 8.1
def task_8_1(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()
    # Get word timestamps
    word_timestamps = features.word_timestamps(sound)

    # Get features
    pauses = features.get_pauses(sound)
    subject_features["NoP_8.1"] = features.number_of_pauses(pauses)
    subject_features["PPT_8.1"] = features.percent_pause_time(sound, pauses)
    subject_features["SPIR_8.1"] = features.speech_index_of_rhythmicity(sound, pauses, word_timestamps)
    subject_features["RIWP_8.1"] = features.ratio_of_intra_word_pauses(pauses, word_timestamps)
    return subject_features

#%% Task 8.2  (subtasks 1 to 3)
def task_8_2(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()

    # Get features
    subject_features["relF0VR_8.2"].append(features.relF0VR(sound))
    subject_features["relF0SD_8.2"].append(features.relF0SD(sound))
    return subject_features

#%% Task 8.3 (subtasks 1 to 3)
def task_8_3(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()

    # Load reference speech file for rhythm comparison
    reference_task_path = os.path.join(THIS_DIR, os.pardir, "data", "reference", "reference_8.3-" + task_path.split("_8.3-")[1])
    reference_sound = parselmouth.Sound(reference_task_path)
    reference_sound.pre_emphasize()
    # Get features
    subject_features["RS_8.3"].append(features.rhythm_similarity(sound, reference_sound))
    return subject_features

#%% Task 8.4 (subtasks 1 to 4)
def task_8_4(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()
    # Get features
    subject_features["relF0VR_8.4"].append(features.relF0VR(sound))
    subject_features["relF0SD_8.4"].append(features.relF0VR(sound))
    return subject_features

#%% Task 9.1 (subtasks 1 to 16)
def task_9_1(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()
    # Get features
    pauses = features.get_pauses(sound)
    subject_features["relF1SD_9.1"].append(features.relF1SD(sound))
    subject_features["relF2SD_9.1"].append(features.relF2SD(sound))
    subject_features["relF0SD_9.1"].append(features.relF0SD(sound))
    subject_features["relF0VR_9.1"].append(features.relF0VR(sound))
    subject_features["relSEOSD_9.1"].append(features.relSEOSD(sound))
    subject_features["relSEOVR_9.1"].append(features.relSEOVR(sound))
    subject_features["RIWP_9.1"].append(features.percent_pause_time(sound, pauses))  # Ratio of Intra-Word Pauses
    subject_features["SPIR_9.1"].append(features.number_of_pauses(pauses)/sound.duration * 60)  # SPIR
    return subject_features

#%% Task 9.2 (subtasks 1 to 5)
def task_9_2(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()
    # Get word timestamps
    word_timestamps = features.word_timestamps(sound)
    # Get features
    pauses = features.get_pauses(sound)
    subject_features["relF1SD_9.2"].append(features.relF1SD(sound))
    subject_features["relF2SD_9.2"].append(features.relF2SD(sound))
    subject_features["relF0SD_9.2"].append(features.relF0SD(sound))
    subject_features["relF0VR_9.2"].append(features.relF0VR(sound))
    subject_features["relSEOSD_9.2"].append(features.relSEOSD(sound))
    subject_features["relSEOVR_9.2"].append(features.relSEOVR(sound))
    subject_features["RIWP_9.2"].append(features.ratio_of_intra_word_pauses(pauses, word_timestamps))
    subject_features["SPIR_9.2"].append(features.speech_index_of_rhythmicity(sound, pauses, word_timestamps))
    subject_features["PPT_9.2"].append(features.percent_pause_time(sound, pauses))
    subject_features["NoP_9.2"].append(features.number_of_pauses(pauses))
    return subject_features

#%% Task 9.4
def task_9_4(task_path, subject_features):
    # Load speech file
    sound = parselmouth.Sound(task_path)
    sound.pre_emphasize()
    # Get word timestamps
    word_timestamps = features.word_timestamps(sound)
    # Get features
    pauses = features.get_pauses(sound)
    subject_features["relF1SD_9.4"] = features.relF1SD(sound)
    subject_features["relF2SD_9.4"] = features.relF2SD(sound)
    subject_features["relF0SD_9.4"] = features.relF0SD(sound)
    subject_features["relF0VR_9.4"] = features.relF0VR(sound)
    subject_features["relSEOSD_9.4"] = features.relSEOSD(sound)
    subject_features["relSEOVR_9.4"] = features.relSEOVR(sound)
    subject_features["RIWP_9.4"] = features.ratio_of_intra_word_pauses(pauses, word_timestamps)
    subject_features["SPIR_9.4"] = features.speech_index_of_rhythmicity(sound, pauses, word_timestamps)
    subject_features["PPT_9.4"] = features.percent_pause_time(sound, pauses)
    subject_features["NoP_9.4"] = features.number_of_pauses(pauses)
    return subject_features
#%%


def extract_all_features(subject_id):

    # Initialize dictionary of all features
    list_of_features = ["MET_5.1", "MET_5.2",
                        "F2i/F2u_7.1", "VSA_7.1", "VAI_7.1", "FCR_7.1",
                        "NoP_8.1", "PPT_8.1", "SPIR_8.1", "RIWP_8.1",
                        "relF0VR_8.2", "relF0SD_8.2",
                        "RS_8.3",  # Rhythm similarity
                        "relF0VR_8.4", "relF0SD_8.4",
                        "relF1SD_9.1", "relF2SD_9.1", "relF0SD_9.1", "relF0VR_9.1", "relSEOSD_9.1", "relSEOVR_9.1",
                        "RIWP_9.1", "SPIR_9.1",
                        "relF1SD_9.2", "relF2SD_9.2", "relF0SD_9.2", "relF0VR_9.2", "relSEOSD_9.2", "relSEOVR_9.2",
                        "RIWP_9.2", "SPIR_9.2", "PPT_9.2", "NoP_9.2",
                        "relF1SD_9.4", "relF2SD_9.4", "relF0SD_9.4", "relF0VR_9.4", "relSEOSD_9.4", "relSEOVR_9.4",
                        "RIWP_9.4", "SPIR_9.4", "PPT_9.4", "NoP_9.4"
                        ]
    subject_features = {key: [] for key in list_of_features}

    # Map task_id to function name (Format: _0.0 to task_0_0)
    func_dict = {k: eval("task" + k.replace(".", "_")) for k in read_all_tasks(remove_subtasks=True, tasks_to_remove=TASKS_TO_REMOVE)}

    # Get features for all tasks
    task_dict = tasks_dictionary(TASKS_TO_REMOVE)
    for base_task, subtasks in task_dict.items():
        for subtask in subtasks:
            if isinstance(subtask, str):
                task_path = get_task_path(subject_id, subtask)

            # Special task formats, e.g. task 7.1
            elif isinstance(subtask, list):
                task_path = []
                for subsubtask in subtask:
                    task_path.append(get_task_path(subject_id, subsubtask))
                if any([True for subsubtask in task_path if subsubtask is None]):
                    task_path = None

            # Calculate features
            if task_path is not None:
                subject_features = func_dict[base_task](task_path, subject_features)

    # Get mean of subtasks
    for key, val in subject_features.items():
        if isinstance(val, list) and len(val) == 0:
            subject_features[key] = np.nan
        elif isinstance(val, list) and len(val) > 0:
            subject_features[key] = np.mean(val)

    return subject_features

#%% Get features for all subjects
subject_ids = os.listdir(DATA_DIR)
features_all = []
for i, subject_id in enumerate(subject_ids):
    print(f"Feature extraction progress: Subject: {subject_id} [{i+1}/{len(subject_ids)}]")
    features_all.append(extract_all_features(subject_id))

# Create dataframe from features_all and save to csv
dataset = pd.DataFrame(features_all, index=subject_ids)
dataset_file_path = os.path.join(DATASET_DIR, "features_dataset.csv")
os.makedirs(os.path.dirname(dataset_file_path), exist_ok=True)
dataset.to_csv(dataset_file_path, sep=";")
