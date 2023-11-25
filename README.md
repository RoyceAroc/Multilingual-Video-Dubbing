# Multilingual Video Dubbing
This project converts a video in one language to another language.
<details>
<summary><span style="font-weight: bold;">Supported target languages</span></summary>
Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.
</details>

## Installation Guidelines
Clone the repository
```shell
git clone https://github.com/RoyceAroc/Multilingual-Video-Dubbing.git && cd Multilingual-Video-Dubbing
```
Setup the environment and install dependencies
```shell
python -m venv video-dubbing
source video-dubbing/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
Adjust the base variables (video url, source, and target language) in ```main.py``` and run the file
```shell
python main.py
```
## Examples
### Original Video (in English)


https://github.com/RoyceAroc/Multilingual-Video-Dubbing/assets/47615786/70e6f50c-3264-4c3e-ad52-ea5833239fb5


### Final Video (in Hindi)


https://github.com/RoyceAroc/Multilingual-Video-Dubbing/assets/47615786/1f04dafa-773e-4457-befa-240026ee0550

## Limitations and Future Directions
- Lack of Lip Synchronization. AI pose models can be used to improve results.
- Lack of AI Voice Cloning. Using speaker embeddings on voices from the original video to generate voice cloning in different languages will improve results.
- Lack of Speaker Diarization (identifying speaker 1, speaker 2, etc). Speaker Diarization models (spectral clustering, affinization, etc) can be used to improve results by using the other voice models that Whisper offers.
- Weak Gender Diarization model. Current model was trained only on Malaya speech and may have biases on identifying between male/female voices.
