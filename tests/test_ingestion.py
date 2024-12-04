import pytest
from  ingestion.text_ingestion import ingest_text
from  ingestion.image_ingestion import ingest_image

def test_text_ingestion():
    text = ingest_text("data/text/sample.txt")
    assert len(text) > 0

def test_image_ingestion():
    text_from_image = ingest_image("data/images/sample.png")
    assert len(text_from_image) > 0  # Ensure OCR extracts text

# # Test Audio Ingestion
# def test_audio_ingestion():
#     audio_text = ingest_audio("data/audio/sample_audio.mp3")
#     assert len(audio_text) > 0  # Ensure audio is transcribed

# # Test Video Ingestion
# def test_video_ingestion():
#     video_text = ingest_video("data/video/sample_video.mp4")
#     assert len(video_text) > 0  # Ensure video is transcribed
