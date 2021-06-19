from google.cloud import speech
import io
import os
import numpy as np
from test_3F.utils import THIS_DIR, SPEECH_TO_TEXT_DIR, DATA_DIR
from test_3F.utils import get_task_path, get_all_tasks, read_all_tasks
#%%
def transcribe_speech(speech_file_path):
    """
    Parameters
    ----------
    speech_file_path : string. Supported file format: WAV or FLAC
    
    Returns
    -------
    transcript : string
    time_stamps : numpy array. Start time and end time for each word. Words in columns, start/end time in rows.
    words : list of strings
    num_words : int
    """
    # Google Cloud credentials from private key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(SPEECH_TO_TEXT_DIR, "key.json")
    
    #%% Load speech file, set transcription parameters
    speech_file_path = os.path.abspath(speech_file_path)
    
    with io.open(speech_file_path, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="cs-CZ",
        enable_word_time_offsets=True,
    )
    
    #%% Instantiates a client
    client = speech.SpeechClient()
    # Detects speech in the audio file
    response = client.recognize(config=config, audio=audio)
    
    transcript = ""
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        transcript += result.alternatives[0].transcript
        
    # Get time stamps 
    num_words = len(result.alternatives[0].words)
    time_stamps = np.zeros([2, num_words])    
    words = []    
    for i, word_info in enumerate(result.alternatives[0].words):
        time_stamps[0, i] = word_info.start_time.total_seconds()
        time_stamps[1, i] = word_info.end_time.total_seconds()
        words.append(word_info.word)
       
    return transcript, time_stamps, words, num_words

#%%
def write_transcript(transcript_filename, transcript):
    file_path = os.path.join(SPEECH_TO_TEXT_DIR, "transcripts", transcript_filename + ".txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file = open(file_path, "w+")
    file.write(transcript)
    file.close()


def write_time_stamps(time_stamps_filename, time_stamps):
    time_stamps_path = os.path.join(SPEECH_TO_TEXT_DIR, "time_stamps", time_stamps_filename + ".csv")
    os.makedirs(os.path.dirname(time_stamps_path), exist_ok=True)
    np.savetxt(time_stamps_path, time_stamps, delimiter=";")


def read_transcript(transcript_filename):
    path = os.path.join(SPEECH_TO_TEXT_DIR, "transcripts", transcript_filename + ".txt")
    with open(path, 'r', encoding='utf-8') as file:
        transcript = file.read()
        return transcript


def read_time_stamps(time_stamps_filename):
    path = os.path.join(SPEECH_TO_TEXT_DIR, "time_stamps", time_stamps_filename + ".csv")
    time_stamps = np.genfromtxt(path, delimiter=';').tolist()
    return np.array(time_stamps)


#%% Transcribe speech for all required tasks in Test 3F

def transcribe_tasks(tasks_to_transcribe):
    """
    Get transcription and word time stamps of tasks and save them to files. Finds all necessary subtasks.
    tasks_to_transcribe: list of strings. Format example: ["8.1", "9.2", "9.4"]
    Returns: missing_tasks: list of strings.
    """
    # Get all tasks
    subject_ids = os.listdir(os.path.join(DATA_DIR))
    all_tasks = read_all_tasks(remove_rerecorded_tasks=True)

    # Find subtasks for tasks listed in tasks_to_transcribe
    tasks = [task for task in all_tasks if any(substring in task for substring in tasks_to_transcribe)]

    # Transcribe
    missing_tasks = []
    for i, subject_id in enumerate(subject_ids):
        for task in tasks:
            speech_file_path = get_task_path(subject_id, task)
            if speech_file_path is not None:
                transcript, word_time_stamps = transcribe_speech(speech_file_path)[0:2]
                write_time_stamps(os.path.join(subject_id, subject_id + task), word_time_stamps)
                write_transcript(os.path.join(subject_id, subject_id + task), transcript)
            else:
                missing_tasks.append(subject_id + task)
        print(f"Speech-to-text progress: subject: {i + 1} / {len(subject_ids)}")
    print("Speech-to-text progress: Finished")

    return missing_tasks

