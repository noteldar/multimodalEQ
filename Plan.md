Plan for today - full scope emotion detection from video
1) text is using inflection LLM
2) audio - Hume emotion recognition
3) video - most important, using 3 verticals
A) multimodal Gemini with a prompt
B) facial action coding: 
https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/
https://github.com/tensorsense/faceflow?tab=readme-ov-file
https://github.com/tomas-gajarsky/facetorch
C) remote photoplethysmogtaphy
https://github.com/ubicomplab/rPPG-Toolbox
https://github.com/SamProell/yarppg
https://github.com/Rouast-Labs/vitallens-python


Output must be a json file that has several fields that describe emotion from different sources


Use whisper and inflection AI to get top 3 emotions with the most intencity scores from the video


Emotion #1 
Score

Emotion #2
Score

Emotion #3
Score