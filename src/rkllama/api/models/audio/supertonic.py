import io
import json
import logging
import os
import re
import time
import uuid
from typing import Optional
from unicodedata import normalize

import numpy as np
import soundfile as sf
from pydub import AudioSegment

logger = logging.getLogger("rkllama.audio.supertonic")

# RKNN component names expected inside the Supertonic model directory.
DURATION_PREDICTOR = "duration_predictor"
TEXT_ENCODER = "text_encoder"
VECTOR_ESTIMATOR = "vector_estimator"
VOCODER = "vocoder"
COMPONENTS = (DURATION_PREDICTOR, TEXT_ENCODER, VECTOR_ESTIMATOR, VOCODER)

# Marker / configuration file that identifies a Supertonic model directory.
SUPERTONIC_CONFIG = "supertonic.json"

# Fallbacks matching assets/onnx/tts.json when not present in supertonic.json.
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BASE_CHUNK_SIZE = 512
DEFAULT_CHUNK_COMPRESS_FACTOR = 6
DEFAULT_TOTAL_STEP = 4
DEFAULT_SPEED = 1.05
DEFAULT_LANG = "en"
DEFAULT_SILENCE_DURATION = 0.3

AVAILABLE_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
    "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
]


def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    """Convert lengths (B,) to a binary mask (B, 1, max_len)."""
    max_len = max_len or int(lengths.max())
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    return length_to_mask(latent_lengths)


class UnicodeProcessor:
    """Port of the Supertonic text processor (scripts/helper.py)."""

    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str, lang: str) -> str:
        text = normalize("NFKD", text)

        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        replacements = {
            "\u2013": "-",
            "\u2011": "-",
            "\u2014": "-",
            "_": " ",
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u00b4": "'",
            "`": "'",
            "[": " ",
            "]": " ",
            "|": " ",
            "/": " ",
            "#": " ",
            "\u2192": " ",
            "\u2190": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        text = re.sub(r"[\u2665\u2606\u2661\u00a9\\]", "", text)

        expr_replacements = {
            "@": " at ",
            "e.g.,": "for example, ",
            "i.e.,": "that is, ",
        }
        for k, v in expr_replacements.items():
            text = text.replace(k, v)

        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)

        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        while "``" in text:
            text = text.replace("``", "`")

        text = re.sub(r"\s+", " ", text).strip()

        if not re.search(r"[.!?;:,'\"')\]}\u2026\u3002\u300d\u300f\u3011\u3009\u300b\u203a\u00bb]$", text):
            text += "."

        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")
        text = f"<{lang}>" + text + f"</{lang}>"
        return text

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        return np.array([ord(char) for char in text], dtype=np.uint16)

    def __call__(
        self, text_list: list[str], lang_list: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        text_list = [
            self._preprocess_text(t, lang) for t, lang in zip(text_list, lang_list)
        ]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), int(text_ids_lengths.max())), dtype=np.int64)
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            text_ids[i, : len(unicode_vals)] = np.array(
                [self.indexer[val] for val in unicode_vals], dtype=np.int64
            )
        text_mask = length_to_mask(text_ids_lengths)
        return text_ids, text_mask


class SupertonicTTSModelRKNN:
    """Supertonic multilingual TTS running on RKNN (RK3588 NPU).

    The pipeline chains four RKNN models: duration_predictor, text_encoder,
    vector_estimator (flow-matching, run ``total_step`` times) and vocoder.
    """

    def __init__(self, model_runtime: dict, model_path: str):
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"Not a model directory: {model_path}")

        self.model_path = model_path
        self.config = self._load_config(model_path)

        # Resolve the four RKNN runtimes by matching component name in the path.
        self.dp_rknn = self._find_runtime(model_runtime, DURATION_PREDICTOR)
        self.text_encoder_rknn = self._find_runtime(model_runtime, TEXT_ENCODER)
        self.vector_estimator_rknn = self._find_runtime(model_runtime, VECTOR_ESTIMATOR)
        self.vocoder_rknn = self._find_runtime(model_runtime, VOCODER)

        # Text processor (unicode codepoint indexer).
        indexer_path = os.path.join(model_path, "unicode_indexer.json")
        if not os.path.isfile(indexer_path):
            raise FileNotFoundError(
                f"Missing unicode_indexer.json in Supertonic model dir: {model_path}"
            )
        self.processor = UnicodeProcessor(indexer_path)

        # Fixed shapes baked into the exported RKNN models.
        self.text_length = int(
            self.dp_rknn.rknn_runtime.get_tensor_attr(0, is_output=False).dims[1]
        )
        vocoder_in_dims = self.vocoder_rknn.rknn_runtime.get_tensor_attr(
            0, is_output=False
        ).dims
        self.latent_dim = int(vocoder_in_dims[1])
        self.latent_length = int(vocoder_in_dims[2])

        # Pipeline configuration (with sensible fallbacks).
        self.sample_rate = int(self.config.get("sample_rate", DEFAULT_SAMPLE_RATE))
        self.base_chunk_size = int(
            self.config.get("base_chunk_size", DEFAULT_BASE_CHUNK_SIZE)
        )
        self.chunk_compress_factor = int(
            self.config.get("chunk_compress_factor", DEFAULT_CHUNK_COMPRESS_FACTOR)
        )
        self.total_step = int(self.config.get("total_step", DEFAULT_TOTAL_STEP))
        self.default_speed = float(self.config.get("speed", DEFAULT_SPEED))
        self.default_lang = str(self.config.get("lang", DEFAULT_LANG))
        self.default_voice = self.config.get("voice", None)
        self.silence_duration = float(
            self.config.get("silence_duration", DEFAULT_SILENCE_DURATION)
        )

        logger.debug(
            "Loaded Supertonic model: text_length=%d, latent_length=%d, latent_dim=%d, "
            "sample_rate=%d, total_step=%d",
            self.text_length,
            self.latent_length,
            self.latent_dim,
            self.sample_rate,
            self.total_step,
        )

    def _load_config(self, model_path: str) -> dict:
        config_path = os.path.join(model_path, SUPERTONIC_CONFIG)
        if not os.path.isfile(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _find_runtime(self, model_runtime: dict, component: str):
        """Return the RKNN runtime whose file path contains ``component``."""
        for path, runtime in model_runtime.items():
            name = os.path.basename(path).lower()
            if name.endswith(".rknn") and component in name:
                return runtime
        raise FileNotFoundError(
            f"Missing RKNN model for '{component}' in Supertonic model dir: "
            f"{self.model_path}"
        )

    def _load_voice_style(self, voice: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
        """Load a voice style (style_ttl, style_dp) from the model directory.

        Falls back to the configured default voice or the first available style.
        """
        styles_dir = os.path.join(self.model_path, "voice_styles")
        if not os.path.isdir(styles_dir):
            raise FileNotFoundError(
                f"Missing voice_styles directory in Supertonic model dir: "
                f"{self.model_path}"
            )

        candidate = None
        for name in (voice, self.default_voice):
            if not name:
                continue
            for filename in (f"{name}.json", name):
                path = os.path.join(styles_dir, filename)
                if os.path.isfile(path):
                    candidate = path
                    break
            if candidate:
                break

        if candidate is None:
            available = sorted(
                f for f in os.listdir(styles_dir) if f.endswith(".json")
            )
            if not available:
                raise FileNotFoundError(f"No voice styles found in {styles_dir}")
            candidate = os.path.join(styles_dir, available[0])
            logger.debug("Voice '%s' not found, using default style %s", voice, candidate)

        with open(candidate, "r") as f:
            style = json.load(f)

        ttl_dims = style["style_ttl"]["dims"]
        dp_dims = style["style_dp"]["dims"]
        style_ttl = (
            np.array(style["style_ttl"]["data"], dtype=np.float32)
            .flatten()
            .reshape(1, ttl_dims[1], ttl_dims[2])
        )
        style_dp = (
            np.array(style["style_dp"]["data"], dtype=np.float32)
            .flatten()
            .reshape(1, dp_dims[1], dp_dims[2])
        )
        return style_ttl, style_dp

    def _pad_text(
        self, text_ids: np.ndarray, text_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if text_ids.shape[1] > self.text_length:
            raise ValueError(
                f"Input text needs {text_ids.shape[1]} tokens, but the RKNN model "
                f"text_length is {self.text_length}"
            )
        ids = np.zeros((1, self.text_length), dtype=text_ids.dtype)
        mask = np.zeros((1, 1, self.text_length), dtype=np.float32)
        ids[:, : text_ids.shape[1]] = text_ids
        mask[:, :, : text_mask.shape[2]] = text_mask
        return ids, mask

    def _infer_chunk(
        self,
        text: str,
        lang: str,
        style_ttl: np.ndarray,
        style_dp: np.ndarray,
        speed: float,
        force: bool = False,
    ) -> Optional[np.ndarray]:
        """Synthesize a single text chunk.

        Returns ``None`` (unless ``force``) when the predicted audio exceeds the
        model's fixed ``latent_length``, signalling the caller to split further.
        """
        text_ids_raw, text_mask_raw = self.processor([text], [lang])
        text_ids, text_mask = self._pad_text(text_ids_raw, text_mask_raw)

        # Duration predictor.
        duration = self.dp_rknn.inference(
            inputs=[text_ids, style_dp, text_mask]
        )[0].astype(np.float32)
        duration = np.maximum(duration / speed, 0.05)

        # Text encoder.
        text_emb = self.text_encoder_rknn.inference(
            inputs=[text_ids, style_ttl, text_mask]
        )[0].astype(np.float32)

        # Build padded latent mask and noisy latent.
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        required_latent_len = int((wav_lengths.max() + chunk_size - 1) // chunk_size)
        if required_latent_len > self.latent_length:
            if not force:
                # Signal the caller to split this chunk into smaller pieces so
                # the predicted audio fits the fixed latent length.
                return None
            # Unsplittable (single long token): cap the duration to what the
            # fixed latent can hold. Trims a small tail rather than crashing.
            max_wav = self.latent_length * chunk_size
            wav_lengths = np.minimum(wav_lengths, max_wav)
            duration = np.minimum(duration, max_wav / self.sample_rate)

        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor
        )
        mask_padded = np.zeros((1, 1, self.latent_length), dtype=np.float32)
        mask_padded[:, :, : latent_mask.shape[2]] = latent_mask
        xt = (
            np.random.randn(1, self.latent_dim, self.latent_length).astype(np.float32)
            * mask_padded
        )

        # Flow-matching denoising loop.
        total_step = np.array([self.total_step], dtype=np.float32)
        for step in range(self.total_step):
            current_step = np.array([step], dtype=np.float32)
            xt = self.vector_estimator_rknn.inference(
                inputs=[
                    xt,
                    text_emb,
                    style_ttl,
                    mask_padded,
                    text_mask,
                    current_step,
                    total_step,
                ]
            )[0].astype(np.float32)

        # Vocoder.
        wav = self.vocoder_rknn.inference(inputs=[xt])[0].astype(np.float32)

        # Trim to the predicted duration.
        output_samples = min(int(float(duration[0]) * self.sample_rate), wav.shape[1])
        return wav[0, :output_samples]

    @staticmethod
    def _split_text(text: str) -> list[str]:
        """Split a chunk into two roughly equal halves (by words, else chars)."""
        words = text.split()
        if len(words) > 1:
            mid = len(words) // 2
            return [" ".join(words[:mid]), " ".join(words[mid:])]
        # Single long token: fall back to a character split.
        if len(text) > 1:
            mid = len(text) // 2
            return [text[:mid], text[mid:]]
        return [text]

    def _synthesize(
        self, text: str, lang: str, voice: Optional[str], speed: float
    ) -> np.ndarray:
        style_ttl, style_dp = self._load_voice_style(voice)

        # Char budget per chunk: text_length minus room for the <lang></lang> tags.
        tag_len = len(f"<{lang}></{lang}>") + 4
        max_len = max(16, self.text_length - tag_len)
        chunks = chunk_text(text, max_len=max_len)
        if not chunks:
            chunks = [text]

        wav_cat = None
        silence = np.zeros(
            int(self.silence_duration * self.sample_rate), dtype=np.float32
        )
        pending = list(chunks)
        while pending:
            chunk = pending.pop(0)
            wav = self._infer_chunk(chunk, lang, style_ttl, style_dp, speed)
            if wav is None:
                # Predicted audio exceeds latent_length: split and retry.
                parts = self._split_text(chunk)
                if len(parts) > 1:
                    pending[:0] = parts
                    continue
                # Cannot split further: force synthesis (trims a small tail).
                wav = self._infer_chunk(
                    chunk, lang, style_ttl, style_dp, speed, force=True
                )
            if wav_cat is None:
                wav_cat = wav
            else:
                wav_cat = np.concatenate([wav_cat, silence, wav])
        return wav_cat

    def generate_speech(
        self,
        input,
        voice=None,
        response_format=None,
        stream_format=None,
        speed=None,
    ) -> tuple[bytes, str]:
        """Generate speech audio bytes for the given text input."""
        lang = self.default_lang
        speed = float(speed) if speed else self.default_speed
        voice = voice or self.default_voice
        response_format = response_format or "wav"

        start = time.time()
        wav = self._synthesize(input, lang, voice, speed)
        logger.debug("Supertonic synthesis time: %.2fs", time.time() - start)

        # Write to a temp WAV file, then convert to the requested format.
        temp_output_path = os.path.join(self.model_path, f"{uuid.uuid4()}.wav")
        sf.write(file=temp_output_path, data=wav, samplerate=self.sample_rate)
        return convert_wav_to_bytes(temp_output_path, response_format)

    def release_rknn_models(self):
        for runtime in (
            self.dp_rknn,
            self.text_encoder_rknn,
            self.vector_estimator_rknn,
            self.vocoder_rknn,
        ):
            try:
                runtime.release()
            except Exception:
                pass


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    """Split text into chunks by paragraphs and sentences (from helper.py)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]

    chunks: list[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        pattern = (
            r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
            r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
            r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
            r"(?<!\b[A-Z]\.)(?<=[.!?])\s+"
        )
        sentences = re.split(pattern, paragraph)

        current_chunk = ""
        for sentence in sentences:
            # A single sentence longer than max_len is split on whitespace.
            while len(sentence) > max_len:
                head = sentence[:max_len]
                cut = head.rfind(" ")
                if cut <= 0:
                    cut = max_len
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()

            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


def convert_wav_to_bytes(wav_path: str, output_format: str) -> tuple[bytes, str]:
    """Convert a WAV file on disk to the requested audio format in memory."""
    output_format = output_format.lower()
    supported = {"mp3", "opus", "aac", "flac", "pcm", "wav"}
    if output_format not in supported:
        raise ValueError(f"Unsupported format: {output_format}")

    audio = AudioSegment.from_wav(wav_path)
    os.remove(wav_path)

    out_buffer = io.BytesIO()
    if output_format == "wav":
        audio.export(out_buffer, format="wav")
        media_type = "audio/wav"
    elif output_format == "mp3":
        audio.export(out_buffer, format="mp3")
        media_type = "audio/mpeg"
    elif output_format == "opus":
        audio.export(out_buffer, format="opus")
        media_type = "audio/opus"
    elif output_format == "aac":
        audio.export(out_buffer, format="adts")
        media_type = "audio/aac"
    elif output_format == "flac":
        audio.export(out_buffer, format="flac")
        media_type = "audio/flac"
    elif output_format == "pcm":
        out_buffer.write(audio.raw_data)
        media_type = "audio/pcm"

    return out_buffer.getvalue(), media_type
