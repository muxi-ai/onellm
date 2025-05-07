import asyncio
"""
Tests for improved coverage of image.py in muxi-llm.

This file aims to achieve 90%+ coverage of image.py by testing:
- Image.create method
- Image.create_sync method
- Image._download_image method
"""

import pytest
from unittest import mock

from muxi_llm.image import Image


# --- Module-level async context manager mocks ---
class MockAsyncContextManager:
    def __init__(self, response):
        self._response = response
    async def __aenter__(self):
        return self._response
    async def __aexit__(self, exc_type, exc, tb):
        return None

class MockSessionContextManager:
    def __init__(self, session):
        self._session = session
    async def __aenter__(self):
        return self._session
    async def __aexit__(self, exc_type, exc, tb):
        return None


class TestImageCreation:
    """Tests for Image.create method."""

    @pytest.mark.asyncio
    async def test_create_basic(self):
        """
        Test basic image creation with the default model.
        """
        prompt = "A beautiful sunset over a mountain"
        model = "openai/dall-e-3"
        expected_result = {
            "created": 1617043084,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": (
                        "A beautiful sunset casting golden light over a mountain range"
                    )
                }
            ]
        }
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider:
            mock_provider = mock.MagicMock()
            mock_provider.create_image = mock.AsyncMock(return_value=expected_result)
            mock_get_provider.return_value = (mock_provider, "dall-e-3")
            result = await Image.create(prompt=prompt, model=model)
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_create_with_parameters(self):
        """
        Test image creation with custom parameters, including all supported kwargs
        and multiple images.
        """
        prompt = "A futuristic cityscape"
        model = "openai/dall-e-3"
        expected_result = {
            "created": 1617043084,
            "data": [
                {
                    "b64_json": "base64encodedimage1",
                    "revised_prompt": (
                        "A vibrant futuristic cityscape with flying cars"
                    )
                },
                {
                    "b64_json": "base64encodedimage2",
                    "revised_prompt": (
                        "A detailed futuristic cityscape with "
                        "tall skyscrapers"
                    )
                }
            ]
        }
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider:
            mock_provider = mock.MagicMock()
            mock_provider.create_image = mock.AsyncMock(return_value=expected_result)
            mock_get_provider.return_value = (mock_provider, "dall-e-3")
            result = await Image.create(
                prompt=prompt,
                model=model,
                n=2,
                size="512x512",
                quality="hd",
                style="vivid",
                response_format="b64_json",
                user="test-user"
            )
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_create_with_fallbacks(self):
        """Test image creation with fallback configuration."""
        prompt = "A beautiful sunset over a mountain"
        model = "openai/dall-e-3"
        fallback_models = ["openai/dall-e-2", "stability/sdxl"]
        fallback_config = {"max_attempts": 3, "retry_delay": 1}
        expected_result = {
            "created": 1617043084,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": "A beautiful sunset casting golden light over a mountain"
                }
            ]
        }
        class DummyFallbackConfig:
            def __init__(self, **kwargs):
                pass
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider, \
             mock.patch("muxi_llm.image.FallbackConfig", DummyFallbackConfig):
            mock_provider = mock.MagicMock()
            mock_provider.create_image = mock.AsyncMock(return_value=expected_result)
            mock_get_provider.return_value = (mock_provider, "dall-e-3")
            result = await Image.create(
                prompt=prompt,
                model=model,
                fallback_models=fallback_models,
                fallback_config=fallback_config
            )
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_create_with_output_dir_url(self):
        """Test image creation with output directory for URL-based responses."""
        prompt = "A beautiful sunset over a mountain"
        model = "openai/dall-e-3"
        output_dir = "/tmp/images"
        test_image_bytes = b"fake image data"
        expected_result = {
            "created": 1617043084,
            "data": [
                {
                    "url": "https://example.com/image.png"
                }
            ]
        }
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider, \
             mock.patch("muxi_llm.image.os.makedirs"), \
             mock.patch(
                 "muxi_llm.image.Image._download_image",
                 new=mock.AsyncMock(return_value=test_image_bytes),
             ), \
             mock.patch("builtins.open", mock.mock_open()):
            mock_provider = mock.MagicMock()
            mock_provider.create_image = mock.AsyncMock(return_value=expected_result)
            mock_get_provider.return_value = (mock_provider, "dall-e-3")
            result = await Image.create(
                prompt=prompt,
                model=model,
                output_dir=output_dir
            )
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_create_with_output_dir_b64json(self):
        """Test image creation with output directory for base64-based responses."""
        prompt = "A beautiful sunset over a mountain"
        model = "openai/dall-e-3"
        output_dir = "/tmp/images"
        test_base64 = "aW1hZ2VkYXRh"  # "imagedata" in base64
        test_image_bytes = b"imagedata"
        expected_result = {
            "created": 1617043084,
            "data": [
                {
                    "b64_json": test_base64
                }
            ]
        }
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider, \
             mock.patch("muxi_llm.image.os.makedirs") as mock_makedirs, \
             mock.patch("base64.b64decode", return_value=test_image_bytes) as mock_b64decode, \
             mock.patch("builtins.open", mock.mock_open()) as mock_open:
            mock_provider = mock.MagicMock()
            mock_provider.create_image = mock.AsyncMock(return_value=expected_result)
            mock_get_provider.return_value = (mock_provider, "dall-e-3")
            result = await Image.create(
                prompt=prompt,
                model=model,
                output_dir=output_dir
            )
            assert result == expected_result

            # Verify directory creation
            mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)

            # Verify base64 decoding
            mock_b64decode.assert_called_once_with(test_base64)

            # Verify file writing
            mock_open.assert_called()
            mock_open().write.assert_called_once_with(test_image_bytes)

            # Verify filepath was added to the result
            assert "filepath" in result["data"][0]

    @pytest.mark.asyncio
    async def test_create_with_output_dir_custom_format(self):
        """Test image creation with output directory and custom output format."""
        prompt = "A beautiful sunset over a mountain"
        model = "openai/dall-e-3"
        output_dir = "/tmp/images"
        output_format = "jpg"
        test_image_bytes = b"fake image data"

        expected_result = {
            "created": 1617043084,
            "data": [
                {
                    "url": "https://example.com/image.png"
                }
            ]
        }

        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider, \
             mock.patch("muxi_llm.image.os.makedirs") as mock_makedirs, \
             mock.patch("muxi_llm.image.Image._download_image", new=mock.AsyncMock(return_value=test_image_bytes)) as mock_download, \
             mock.patch("builtins.open", mock.mock_open()) as mock_open:

            mock_provider = mock.MagicMock()
            mock_provider.create_image = mock.AsyncMock(return_value=expected_result)
            mock_get_provider.return_value = (mock_provider, "dall-e-3")

            result = await Image.create(
                prompt=prompt,
                model=model,
                output_dir=output_dir,
                output_format=output_format
            )

            mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
            mock_download.assert_called_once_with("https://example.com/image.png")
            filepath = mock_open.call_args[0][0]
            assert filepath.endswith(f".{output_format}")
            assert result["data"][0]["filepath"].endswith(
                f".{output_format}"
            )

    @pytest.mark.asyncio
    async def test_create_with_output_dir_continue_branch(self):
        """Test Image.create with output_dir and a data entry missing both 'url' and 'b64_json' (covers continue branch)."""
        prompt = "A prompt"
        model = "openai/dall-e-3"
        output_dir = "/tmp/images"
        expected_result = {
            "created": 1617043084,
            "data": [
                {"irrelevant_key": "no image data"}
            ]
        }
        with mock.patch("muxi_llm.image.get_provider_with_fallbacks") as mock_get_provider, \
             mock.patch("muxi_llm.image.os.makedirs"), \
             mock.patch("builtins.open", mock.mock_open()):
            mock_provider = mock.MagicMock()
            mock_provider.create_image = mock.AsyncMock(return_value=expected_result)
            mock_get_provider.return_value = (mock_provider, "dall-e-3")
            result = await Image.create(
                prompt=prompt,
                model=model,
                output_dir=output_dir
            )
            assert result == expected_result


class TestImageCreateSync:
    """Tests for Image.create_sync method."""

    def test_create_sync_basic(self):
        """Test basic synchronous image creation."""
        prompt = "A beautiful sunset over a mountain"
        model = "openai/dall-e-3"

        expected_result = {
            "created": 1617043084,
            "data": [
                {
                    "url": "https://example.com/image.png"
                }
            ]
        }

        # Mock the asyncio.run and create method
        with mock.patch("muxi_llm.image.asyncio.run") as mock_run:
            mock_run.return_value = expected_result

            # Call the synchronous method
            result = Image.create_sync(
                prompt=prompt,
                model=model
            )

            # Verify the result
            assert result == expected_result

            # Verify the async method was properly called
            mock_run.assert_called_once()
            # Extract the coroutine passed to asyncio.run
            coroutine = mock_run.call_args[0][0]
            assert coroutine.__qualname__ == "Image.create"


class TestImageDownload:
    """Tests for Image._download_image method."""

    @pytest.mark.asyncio
    async def test_download_image_success(self):
        """Test successful image download."""
        url = "https://example.com/image.png"
        image_content = b"test image data"

        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.read = mock.AsyncMock(return_value=image_content)

        mock_session = mock.MagicMock()
        mock_session.get.side_effect = lambda *a, **kw: MockAsyncContextManager(mock_response)

        with mock.patch("aiohttp.ClientSession", return_value=MockSessionContextManager(mock_session)):
            result = await Image._download_image(url)
            assert result == image_content
            mock_session.get.assert_called_once_with(url)
            mock_response.read.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_image_error_status(self):
        """Test image download with error status code."""
        url = "https://example.com/image.png"

        mock_response = mock.AsyncMock()
        mock_response.status = 404
        mock_response.read = mock.AsyncMock()

        mock_session = mock.MagicMock()
        mock_session.get.side_effect = lambda *a, **kw: MockAsyncContextManager(mock_response)

        with mock.patch("aiohttp.ClientSession", return_value=MockSessionContextManager(mock_session)):
            with pytest.raises(ValueError) as excinfo:
                await Image._download_image(url)
            assert "Failed to download image: 404" in str(excinfo.value)
            mock_session.get.assert_called_once_with(url)
            mock_response.read.assert_not_called()

    @pytest.mark.asyncio
    async def test_download_image_error_status_500(self):
        """Test image download with 500 error status code (covers error branch)."""
        url = "https://example.com/image.png"

        mock_response = mock.AsyncMock()
        mock_response.status = 500
        mock_response.read = mock.AsyncMock()

        mock_session = mock.MagicMock()
        mock_session.get.side_effect = lambda *a, **kw: MockAsyncContextManager(mock_response)

        with mock.patch("aiohttp.ClientSession", return_value=MockSessionContextManager(mock_session)):
            with pytest.raises(ValueError) as excinfo:
                await Image._download_image(url)
            assert "Failed to download image: 500" in str(excinfo.value)
            mock_session.get.assert_called_once_with(url)
            mock_response.read.assert_not_called()
