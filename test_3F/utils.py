import os
import numpy as np
from itertools import groupby
import logging
#%% Project paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, os.pardir, "data", "labeled")
SPEECH_TO_TEXT_DIR = os.path.join(THIS_DIR, os.pardir, "data", "speech_to_text")
DATASET_DIR = os.path.join(THIS_DIR, os.pardir, "data", "dataset")
RESULTS_DIR = os.path.join(THIS_DIR, os.pardir, "data", "results")
#%%


def get_task_path(subject_id, task):
    """
    subject_id: (string). Format example: P2049
    task: (string). Format example: _9.4_1 or _9.1-8_1 or _9.4-l_1
    returns: filepath to task if it does exist otherwise returns filepath to rerecorded task if it does exist.
             Returns None if task and rerecorded task does not exist.
    """
    speech_file_path = os.path.join(DATA_DIR, subject_id, subject_id + task + ".wav")
    # Is task a subtask?
    if "-" in task:
        is_subtask = True
    else:
        is_subtask = False

    if os.path.isfile(speech_file_path):
        logging.info(f"Task exists: {subject_id + task}. Path to task returned")
        return speech_file_path
    # If task does not exist use rerecorded task instead
    elif is_subtask:
        try:
            tmp = task.split("-")
            rerecorded_task = tmp[0] + "-l-" + tmp[1]
            speech_file_path = os.path.join(DATA_DIR, subject_id, subject_id + rerecorded_task + ".wav")
            if os.path.isfile(speech_file_path):
                logging.info(f"Task does not exist: {subject_id + task}. Path to rerecorded task returned: {subject_id + rerecorded_task}")
                return speech_file_path
            else:
                logging.warning(f"Task does not exist: {subject_id + task}. Rerecorded task does not exist.")
                return None
        except IndexError:
            raise ValueError(f"Wrong task format: {subject_id + task}")
    elif not is_subtask:
        try:
            tmp = task.split("_", 2)
            rerecorded_task = tmp[0] + "_" + tmp[1] + "-l_" + tmp[2]
            speech_file_path = os.path.join(DATA_DIR, subject_id, subject_id + rerecorded_task + ".wav")
            if os.path.isfile(speech_file_path):
                logging.info(f"Task does not exist: {subject_id + task}. Path to rerecorded task returned: {subject_id + rerecorded_task}")
                return speech_file_path
            else:
                logging.warning(f"Task does not exist: {subject_id + task}. Rerecorded task does not exist.")
                return None
        except IndexError:
            raise ValueError(f"Wrong task format: {subject_id + task}")
    else:
        logging.warning(f"Task does not exist: {subject_id + task}")
        return None


def get_all_tasks(remove_rerecorded_tasks=False, save_to_csv=False):
    """
    Scan dataset directory (DATA_DIR) and return: all tasks found.
    options: remove rerecorded tasks, default: False. Save to csv file, default: False.
    """
    # Get all subtasks (and tasks) to transcribe
    subject_ids = os.listdir(os.path.join(DATA_DIR))
    all_tasks = []
    for subject in subject_ids:
        subject_tasks = os.listdir(os.path.join(DATA_DIR, subject))
        subject_tasks = list(map(lambda x: x.replace(subject, "").replace(".wav", ""), subject_tasks))
        all_tasks.extend(subject_tasks)
    all_tasks = list(dict.fromkeys(all_tasks))  # remove duplicates

    if remove_rerecorded_tasks:
        all_tasks = [task for task in all_tasks if "l" not in task]

    if save_to_csv:
        path = os.path.join(THIS_DIR, os.pardir, "data", "test3f_tasks", "list_of_tasks.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, all_tasks, delimiter=";", fmt='% s')
    return all_tasks.sort()


def read_all_tasks(remove_rerecorded_tasks=False, remove_subtasks=False, tasks_to_remove=[]):
    """
    Read tasks from file. If file does not exist scan dataset directory (DATA_DIR) and return: all tasks found.
    options: remove rerecorded tasks, default: False. Remove subtasks, default: False. tasks_to_remove:(list of strings)
    """
    try:
        path = os.path.join(THIS_DIR, os.pardir, "data", "test3f_tasks", "list_of_tasks.csv")
        all_tasks = np.genfromtxt(path, delimiter=';', dtype=None, encoding="utf8").tolist()
    except FileNotFoundError:
        all_tasks = get_all_tasks(remove_rerecorded_tasks=False, save_to_csv=False)

    if remove_rerecorded_tasks:
        all_tasks = [task for task in all_tasks if "l" not in task]

    if len(tasks_to_remove) != 0:
        all_tasks = [task for task in all_tasks if not task.startswith(tuple(tasks_to_remove))]

    if remove_subtasks:
        tasks = []
        for task in all_tasks:
            tasks.append(task.split(".")[0]+"."+task.split(".")[1][0])
        all_tasks = list(dict.fromkeys(tasks))  # remove duplicates

    return all_tasks


def tasks_dictionary(TASKS_TO_REMOVE=[]):
    """
    Group subtasks by base-task to dictionary.
    Returns: dictionary: key=base-task, value=list-of-subtasks. Example: base_tasks["_5.2"] = ["5.2-1_1", "5.2-2_1"]
    """
    base_tasks = {k:[] for k in read_all_tasks(remove_subtasks=True, tasks_to_remove=TASKS_TO_REMOVE)}
    all_tasks = read_all_tasks(remove_rerecorded_tasks=True, tasks_to_remove=TASKS_TO_REMOVE)
    for key, val in base_tasks.items():
        for subtask in all_tasks:
            if key in subtask:
                base_tasks[key].append(subtask)

    # Special task formats
    tasks_7_1 = base_tasks["_7.1"]
    base_tasks["_7.1"] = []
    for key, group in groupby(tasks_7_1, lambda x: x.split("-")[1]):
        base_tasks["_7.1"].append(list(group))

    return base_tasks
