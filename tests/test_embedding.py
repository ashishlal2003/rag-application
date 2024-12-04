import pytest
from  embeddings.text_embeddings import text_embeddings
from  embeddings.image_embeddings import image_embeddings

def test_text_embedding():
    text = "This is a sample text for testing."
    embedding = text_embeddings(text)
    assert len(embedding) == 384

def test_image_embedding():
    image_path = "data/images/sample.png"
    embedding = image_embeddings(image_path)
    assert len(embedding[0]) == 512

# Test Audio Embedding (assuming text is transcribed)
# def test_audio_embedding():
#     audio_path = "data/audio/sample_audio.mp3"
#     # In real case, the text is transcribed first; here we simulate it
#     transcribed_text = "This is a sample audio transcription."
#     embedding = generate_audio_embeddings(transcribed_text)
#     assert len(embedding) == 768  # Check for expected dimensionality
