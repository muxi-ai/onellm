#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Tests for OneLLM client interface (OpenAI compatibility)
#
# Copyright (C) 2025 Ran Aroussi
#

import pytest
from unittest.mock import patch, MagicMock

from onellm import Client, OpenAI


def test_client_initialization():
    """Test that client initializes with correct attributes."""
    client = Client(api_key="test-key", organization="test-org")

    # Verify client attributes
    assert client.api_key == "test-key"
    assert client.config.get("organization") == "test-org"

    # Verify client has all required resources
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client, "completions")
    assert hasattr(client, "embeddings")
    assert hasattr(client, "images")
    assert hasattr(client, "audio")
    assert hasattr(client.audio, "transcriptions")
    assert hasattr(client.audio, "translations")
    assert hasattr(client, "speech")
    assert hasattr(client, "files")


def test_openai_alias():
    """Test that OpenAI is an alias for Client."""
    assert OpenAI == Client

    client = OpenAI(api_key="test-key")
    assert isinstance(client, Client)


@patch("onellm.ChatCompletion.create")
def test_chat_completions_create(mock_create):
    """Test that client.chat.completions.create delegates to ChatCompletion.create."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]
    response = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.7)

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(
        model="openai/gpt-4",  # Note the provider prefix added
        messages=messages,
        stream=False,
        fallback_models=None,  # The client always passes this
        temperature=0.7,
    )


@patch("onellm.Completion.create")
def test_completions_create(mock_create):
    """Test that client.completions.create delegates to Completion.create."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct", prompt="Hello world", max_tokens=100
    )

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(
        model="openai/gpt-3.5-turbo-instruct",  # Note the provider prefix added
        prompt="Hello world",
        stream=False,
        fallback_models=None,  # The client always passes this
        max_tokens=100,
    )


@patch("onellm.Embedding.create")
def test_embeddings_create(mock_create):
    """Test that client.embeddings.create delegates to Embedding.create."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.embeddings.create(model="text-embedding-3-small", input="Hello world")

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(
        model="openai/text-embedding-3-small",  # Note the provider prefix added
        input="Hello world",
        fallback_models=None,  # The client always passes this
    )


@patch("onellm.ChatCompletion.create")
def test_explicit_provider_prefix_maintained(mock_create):
    """Test that explicitly provided provider prefixes are maintained."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test with explicit provider prefix
    client = Client(api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]
    response = client.chat.completions.create(model="anthropic/claude-3-opus", messages=messages)

    # Verify provider prefix is maintained and response is correct
    assert response == mock_response
    mock_create.assert_called_once_with(
        model="anthropic/claude-3-opus",  # Should keep the existing prefix
        messages=messages,
        stream=False,
        fallback_models=None,  # The client always passes this
    )


@pytest.mark.asyncio
@patch("onellm.ChatCompletion.acreate")
async def test_async_chat_completions(mock_acreate):
    """Test that client.chat.completions.acreate delegates to ChatCompletion.acreate."""
    # Setup mock
    mock_response = MagicMock()
    mock_acreate.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]
    response = await client.chat.completions.acreate(model="gpt-4", messages=messages, stream=True)

    # Verify
    assert response == mock_response
    mock_acreate.assert_called_once_with(
        model="openai/gpt-4", messages=messages, stream=True, fallback_models=None
    )


@patch("onellm.ChatCompletion.create")
def test_chat_completions_with_fallbacks(mock_create):
    """Test that client.chat.completions.create handles fallback models correctly."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]
    response = client.chat.completions.create(
        model="gpt-4", fallback_models=["claude-3-opus", "llama3-70b"], messages=messages
    )

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(
        model="openai/gpt-4",
        messages=messages,
        stream=False,
        fallback_models=["openai/claude-3-opus", "openai/llama3-70b"],  # Prefixes added
    )


@patch("onellm.ChatCompletion.create")
def test_chat_completions_with_mixed_fallbacks(mock_create):
    """Test that client.chat.completions.create handles mixed provider prefixes correctly."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test with mix of prefixed and non-prefixed models
    client = Client(api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]
    response = client.chat.completions.create(
        model="gpt-4", fallback_models=["anthropic/claude-3-opus", "llama3-70b"], messages=messages
    )

    # Verify provider prefixes are handled correctly and response is correct
    assert response == mock_response
    mock_create.assert_called_once_with(
        model="openai/gpt-4",
        messages=messages,
        stream=False,
        fallback_models=["anthropic/claude-3-opus", "openai/llama3-70b"],  # Existing prefix kept
    )


@patch("onellm.Completion.create")
def test_completions_with_fallbacks(mock_create):
    """Test that client.completions.create handles fallback models correctly."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Hello world",
        fallback_models=["claude-instant", "llama3-8b"],
    )

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(
        model="openai/gpt-3.5-turbo-instruct",
        prompt="Hello world",
        stream=False,
        fallback_models=["openai/claude-instant", "openai/llama3-8b"],
    )


@patch("onellm.Embedding.create")
def test_embeddings_with_fallbacks(mock_create):
    """Test that client.embeddings.create handles fallback models correctly."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello world",
        fallback_models=["text-embedding-3-large", "ada"],
    )

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(
        model="openai/text-embedding-3-small",
        input="Hello world",
        fallback_models=["openai/text-embedding-3-large", "openai/ada"],
    )


@patch("onellm.Image.create")
def test_images_create(mock_create):
    """Test that client.images.create delegates to Image.create."""
    # Set up mock
    mock_create.return_value = MagicMock()

    # Test
    client = Client(api_key="test-key")
    client.images.create(model="dall-e-3", prompt="A cute baby sea otter")

    # We only verify the function was called with right parameters
    mock_create.assert_called_once_with(model="openai/dall-e-3", prompt="A cute baby sea otter")


@patch("onellm.AudioTranscription.create")
def test_audio_transcriptions_create(mock_create):
    """Test that client.audio.transcriptions.create delegates to AudioTranscription.create."""
    # Set up mock
    mock_create.return_value = MagicMock()

    # Test
    client = Client(api_key="test-key")
    client.audio.transcriptions.create(model="whisper-1", file="audio.mp3")

    # We only verify the function was called with right parameters
    mock_create.assert_called_once_with(model="openai/whisper-1", file="audio.mp3")


@patch("onellm.AudioTranslation.create")
def test_audio_translations_create(mock_create):
    """Test that client.audio.translations.create delegates to AudioTranslation.create."""
    # Set up mock
    mock_create.return_value = MagicMock()

    # Test
    client = Client(api_key="test-key")
    client.audio.translations.create(model="whisper-1", file="audio.mp3")

    # We only verify the function was called with right parameters
    mock_create.assert_called_once_with(model="openai/whisper-1", file="audio.mp3")


@patch("onellm.Speech.create")
def test_speech_create(mock_create):
    """Test that client.speech.create delegates to Speech.create."""
    # Set up mock
    mock_create.return_value = MagicMock()

    # Test
    client = Client(api_key="test-key")
    client.speech.create(model="tts-1", input="Hello world", voice="alloy")

    # We only verify the function was called with right parameters
    mock_create.assert_called_once_with(model="openai/tts-1", input="Hello world", voice="alloy")


@pytest.mark.asyncio
@patch("onellm.Completion.acreate")
async def test_async_completions(mock_acreate):
    """Test that client.completions.acreate delegates to Completion.acreate."""
    # Setup mock
    mock_response = MagicMock()
    mock_acreate.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.completions.acreate(
        model="gpt-3.5-turbo-instruct", prompt="Hello world", stream=True
    )

    # Verify
    assert response == mock_response
    mock_acreate.assert_called_once_with(
        model="openai/gpt-3.5-turbo-instruct",
        prompt="Hello world",
        stream=True,
        fallback_models=None,
    )


@pytest.mark.asyncio
@patch("onellm.Embedding.acreate")
async def test_async_embeddings(mock_acreate):
    """Test that client.embeddings.acreate delegates to Embedding.acreate."""
    # Setup mock
    mock_response = MagicMock()
    mock_acreate.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.embeddings.acreate(model="text-embedding-3-small", input="Hello world")

    # Verify
    assert response == mock_response
    mock_acreate.assert_called_once_with(
        model="openai/text-embedding-3-small", input="Hello world", fallback_models=None
    )


@pytest.mark.asyncio
@patch("onellm.Image.create")
async def test_images_acreate(mock_create):
    """Test that client.images.acreate delegates to Image.create properly.
    Note: Image class doesn't have acreate, but returns a coroutine from create
    when called from client.images.acreate."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.images.acreate(model="dall-e-3", prompt="A cute baby sea otter")

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(model="openai/dall-e-3", prompt="A cute baby sea otter")


@pytest.mark.asyncio
@patch("onellm.AudioTranscription.create")
async def test_audio_transcriptions_acreate(mock_create):
    """Test that client.audio.transcriptions.acreate delegates correctly."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.audio.transcriptions.acreate(model="whisper-1", file="audio.mp3")

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(model="openai/whisper-1", file="audio.mp3")


@pytest.mark.asyncio
@patch("onellm.AudioTranslation.create")
async def test_audio_translations_acreate(mock_create):
    """Test that client.audio.translations.acreate delegates to AudioTranslation.create properly.
    Note: AudioTranslation class doesn't have acreate, but returns a coroutine from create
    when called from client.audio.translations.acreate."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.audio.translations.acreate(model="whisper-1", file="audio.mp3")

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(model="openai/whisper-1", file="audio.mp3")


@pytest.mark.asyncio
@patch("onellm.Speech.create")
async def test_speech_acreate(mock_create):
    """Test that client.speech.acreate delegates to Speech.create properly.
    Note: Speech class doesn't have acreate, but returns a coroutine from create
    when called from client.speech.acreate."""
    # Setup mock
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.speech.acreate(model="tts-1", input="Hello world", voice="alloy")

    # Verify
    assert response == mock_response
    mock_create.assert_called_once_with(model="openai/tts-1", input="Hello world", voice="alloy")


# File operations tests - update to use upload/download instead of create/retrieve
@patch("onellm.File.upload")
def test_files_create(mock_upload):
    """Test that client.files.create delegates to File.upload."""
    # Setup mock
    mock_response = MagicMock()
    mock_upload.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.files.create(file="test.txt", purpose="assistants")

    # Verify
    assert response == mock_response
    mock_upload.assert_called_once_with(file="test.txt", purpose="assistants", provider="openai")


@pytest.mark.asyncio
@patch("onellm.File.aupload")
async def test_files_acreate(mock_aupload):
    """Test that client.files.acreate delegates to File.aupload."""
    # Setup mock
    mock_response = MagicMock()
    mock_aupload.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.files.acreate(file="test.txt", purpose="fine-tune")

    # Verify
    assert response == mock_response
    mock_aupload.assert_called_once_with(file="test.txt", purpose="fine-tune", provider="openai")


@patch("onellm.File.download")
def test_files_retrieve(mock_download):
    """Test that client.files.retrieve delegates to File.download."""
    # Setup mock
    mock_response = MagicMock()
    mock_download.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.files.retrieve(file_id="file-123")

    # Verify
    assert response == mock_response
    mock_download.assert_called_once_with(file_id="file-123", provider="openai")


@pytest.mark.asyncio
@patch("onellm.File.adownload")
async def test_files_aretrieve(mock_adownload):
    """Test that client.files.aretrieve delegates to File.adownload."""
    # Setup mock
    mock_response = MagicMock()
    mock_adownload.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.files.aretrieve(file_id="file-123")

    # Verify
    assert response == mock_response
    mock_adownload.assert_called_once_with(file_id="file-123", provider="openai")


@patch("onellm.File.download")
def test_files_content(mock_download):
    """Test that client.files.content delegates to File.download."""
    # Setup mock
    mock_response = MagicMock()
    mock_download.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.files.content(file_id="file-123")

    # Verify
    assert response == mock_response
    mock_download.assert_called_once_with(file_id="file-123", provider="openai")


@pytest.mark.asyncio
@patch("onellm.File.adownload")
async def test_files_acontent(mock_adownload):
    """Test that client.files.acontent delegates to File.adownload."""
    # Setup mock
    mock_response = MagicMock()
    mock_adownload.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.files.acontent(file_id="file-123")

    # Verify
    assert response == mock_response
    mock_adownload.assert_called_once_with(file_id="file-123", provider="openai")


@patch("onellm.File.list")
def test_files_list(mock_list):
    """Test that client.files.list delegates to File.list."""
    # Setup mock
    mock_response = MagicMock()
    mock_list.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.files.list(purpose="fine-tune")

    # Verify
    assert response == mock_response
    mock_list.assert_called_once_with(purpose="fine-tune", provider="openai")


@pytest.mark.asyncio
@patch("onellm.File.alist")
async def test_files_alist(mock_alist):
    """Test that client.files.alist delegates to File.alist."""
    # Setup mock
    mock_response = MagicMock()
    mock_alist.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.files.alist()

    # Verify
    assert response == mock_response
    mock_alist.assert_called_once_with(provider="openai")


@patch("onellm.File.delete")
def test_files_delete(mock_delete):
    """Test that client.files.delete delegates to File.delete."""
    # Setup mock
    mock_response = MagicMock()
    mock_delete.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = client.files.delete(file_id="file-123")

    # Verify
    assert response == mock_response
    mock_delete.assert_called_once_with(file_id="file-123", provider="openai")


@pytest.mark.asyncio
@patch("onellm.File.adelete")
async def test_files_adelete(mock_adelete):
    """Test that client.files.adelete delegates to File.adelete."""
    # Setup mock
    mock_response = MagicMock()
    mock_adelete.return_value = mock_response

    # Test
    client = Client(api_key="test-key")
    response = await client.files.adelete(file_id="file-123")

    # Verify
    assert response == mock_response
    mock_adelete.assert_called_once_with(file_id="file-123", provider="openai")
