from collections.abc import Generator
from re import compile as rc
from struct import pack

_fa_search = rc('[\u0600-\u06ff]').search


def detect_lang(text: str) -> str:
    return 'fa' if _fa_search(text) else 'en'


CHANNELS = 1
BITS_PER_SAMPLE = 16
BLOCK_ALIGN = CHANNELS * (BITS_PER_SAMPLE // 8)


def create_wav_header(sample_rate: int) -> bytes:
    """
    Creates a streaming WAV header.

    The sizes are set to 0xFFFFFFFF because the final audio length
    is not known when streaming begins.
    """

    unknown_size = 0xFFFFFFFF

    header = pack(
        '<4sI4s',
        b'RIFF',
        unknown_size,
        b'WAVE',
    )
    byte_rate = sample_rate * BLOCK_ALIGN

    header += pack(
        '<4sIHHIIHH',
        b'fmt ',
        16,  # PCM fmt chunk size
        1,  # PCM format
        CHANNELS,
        sample_rate,
        byte_rate,
        BLOCK_ALIGN,
        BITS_PER_SAMPLE,
    )

    header += pack(
        '<4sI',
        b'data',
        unknown_size,
    )

    return header


_splitter = rc(r'(?<=[.!?])\s+|\n\s*').split


def split_text(text: str, target_chars: int = 250) -> Generator[str]:
    """Split text into reasonably sized chunks."""
    sentences = _splitter(text)

    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if current and current_len + len(sentence) > target_chars:
            yield ' '.join(current)
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)

    if current:
        yield ' '.join(current)
