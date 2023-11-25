import evadb
import os
import malaya_speech
from malaya_speech import Pipeline
import pandas as pd
import subprocess
from pydub import AudioSegment
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
from moviepy.editor import VideoFileClip, AudioFileClip

# OPEN AI used for Text to Speech ($0.015 / 1K characters)
os.environ["OPENAI_API_KEY"] = "<set OPENAI API KEY here>"

# Initialize base variables
video_url = "video.mp4" # Local Video URL
source_language = "en" # Original video language in ISO 639-1
target_language = "hi" # Output video language in ISO 639-1

# Function to batch audio files into segment_lengths with overlap
def load_audio(file_path, segment_length, overlap):
    audio = AudioSegment.from_wav(file_path)
    segment_length_ms = segment_length * 60 * 1000
    overlap_ms = overlap * 60 * 1000

    for start_time in range(0, len(audio), segment_length_ms - overlap_ms):
        end_time = start_time + segment_length_ms
        segment = str(audio[start_time:end_time].get_array_of_samples())
        inner_list_str = segment[segment.index('[') + 1:-2]
        inner_list = [int(num) for num in inner_list_str.split(', ')]
        cursor.query(f"""
            INSERT INTO VideoToAudio (audio) VALUES ('{str(inner_list)}')
        """).df()

# Function to diarize speech based on gender
def speech_diarization(): 
    y, sr = malaya_speech.load('audio.wav')
    vggvox_v2 = malaya_speech.gender.deep_model(model = 'vggvox-v2')
    vad = malaya_speech.vad.deep_model(model = 'vggvox-v2')
    frames = list(malaya_speech.utils.generator.frames(y, 30, sr))
    p = Pipeline()
    pipeline = (
        p.batching(5)
        .foreach_map(vad.predict)
        .flatten()
    )
    p.visualize()
    result = p.emit(frames)
    result.keys()
    frames_vad = [(frame, result['flatten'][no]) for no, frame in enumerate(frames)]
    grouped_vad = malaya_speech.utils.group.group_frames(frames_vad)
    grouped_vad = malaya_speech.utils.group.group_frames_threshold(grouped_vad, threshold_to_stop = 0.3)
    p_vggvox_v2 = Pipeline()
    pipeline = (
        p_vggvox_v2.foreach_map(vggvox_v2)
        .flatten()
    )
    p_vggvox_v2.visualize()
    samples_vad = [g[0] for g in grouped_vad]
    result_vggvox_v2 = p_vggvox_v2.emit(samples_vad)
    samples_vad = [g[0] for g in grouped_vad]
    samples_vad_vggvox_v2 = [(frame, result_vggvox_v2['flatten'][no]) for no, frame in enumerate(samples_vad)]

    m = malaya_speech.group.group_frames(samples_vad_vggvox_v2)
    grouped = pd.DataFrame.from_dict(m)
    return grouped

cursor = evadb.connect().cursor()

# Convert Video File to Audio for processing
command = f"ffmpeg -i {video_url} -ab 160k -ac 2 -ar 44100 -vn audio.wav"
subprocess.call(command, shell=True)

cursor.query("DROP TABLE IF EXISTS VideoToAudio;").df()
cursor.query("""
    CREATE TABLE IF NOT EXISTS VideoToAudio (
        audio TEXT
    )
""").df()

load_audio("audio.wav", 15, 1)
loaded_data = speech_diarization()

cursor.query("DROP TABLE IF EXISTS TranslationTable;").df()
cursor.query("""
    CREATE TABLE IF NOT EXISTS TranslationTable (
        original TEXT,
        translated TEXT,
        voice TEXT,
        start FLOAT,
        end FLOAT
    )
""").df()

# Load Translation Model
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
tokenizer.src_lang = source_language

# Load Whispher Speech-to-Text Model
speech_recognition_model = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base",
    chunk_length_s=30, 
    generate_kwargs={"num_beams": 10}
)

# Result of Whispher Speech-to-Text Model
result = speech_recognition_model("audio.wav", return_timestamps=True, generate_kwargs={"task": "translate"})

# Algorithm to integrate gender diarization segments with above result -> load into table
for results in result['chunks']:
    dist = 0
    gender = "male"
    for i in range(len(loaded_data)):
        start = loaded_data[0][i].timestamp
        end = loaded_data[0][i].timestamp + loaded_data[0][i].duration
        if results['timestamp'][0] <= end and results['timestamp'][1] >= end:
            cur = abs(end - results['timestamp'][0])
            if cur >= dist and loaded_data[1][i] != "not a gender":
                dist = cur
                gender = loaded_data[1][i]
        elif results['timestamp'][0] <= start and results['timestamp'][1] >= start:
            cur = abs(results['timestamp'][1] - start)
            if cur >= dist and loaded_data[1][i] != "not a gender":
                dist = cur
                gender = loaded_data[1][i]
        elif (results['timestamp'][0] >= start and results['timestamp'][1] <= end) or (start >= results['timestamp'][0] and end <= results['timestamp'][1]):
            cur = abs((results['timestamp'][1] - results['timestamp'][0]) - (end-start))
            if cur >= dist and loaded_data[1][i] != "not a gender":
                dist = cur
                gender = loaded_data[1][i]
    translated_text = tokenizer.batch_decode(model.generate(**(tokenizer(results['text'], return_tensors='pt')), forced_bos_token_id=tokenizer.get_lang_id(target_language)), skip_special_tokens=True)[0]
    original = results['text'].replace("'", "")
    translated  = translated_text.replace("'", "")
    query = f"INSERT INTO TranslationTable (original, translated, voice, start, end) VALUES ('{original}', '{translated}', '{'onyx' if gender == 'male' else 'nova'}', {results['timestamp'][0]}, {results['timestamp'][1]})"
    cursor.query(query).df()

# Convert Text to Speech
cursor.query("DROP FUNCTION IF EXISTS TextToSpeech;").df()
cursor.query("CREATE FUNCTION TextToSpeech IMPL 'evadb/functions/tts.py';").df()
audio = cursor.query("SELECT TextToSpeech(translated) FROM TranslationTable;").df()

# Replace audio segments from above model into original audio
m = cursor.query("SELECT start,end FROM TranslationTable;").df()
final_audio = AudioSegment.from_wav("audio.wav")

for i in range(0, len(m['start'].values)):
    duration= m['end'].values[i] - m['start'].values[i]
    audio_data = AudioSegment(audio['response'].values[i].tobytes(), frame_rate=22050, sample_width=2, channels=1)
    audio_data = AudioSegment(audio['response'].values[i].tobytes(), frame_rate=int(((len(audio_data) / 1000) / duration) * audio_data.frame_rate), sample_width=2, channels=1)
    final_audio = final_audio[:m['start'].values[i]*1000] + audio_data + final_audio[m['start'].values[i]*1000 + len(audio_data):]

# Save output audio
final_audio.export("output_audio.wav", format="wav")

# Save Final Video (original video + output audio)
video_clip = VideoFileClip(video_url)
video_clip = video_clip.set_audio(AudioFileClip("output_audio.wav"))
video_clip.write_videofile("output_video.mp4", codec="libx264", audio_codec="aac")
video_clip.close()