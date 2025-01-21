import asyncio
import struct
import pygame
import sounddevice as sd
from opuslib import Decoder
from constants import (
    SERVER_IP, AUDIO_PORT, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_FRAME_SIZE
)


def create_audio_buffer():
    """Create an asyncio queue for buffering audio data."""
    return asyncio.Queue(maxsize=2)

async def receive_audio(host, port):
    audio_buffer = create_audio_buffer()  # Create a buffer
    writer = None  # Ensure writer is defined in the outer scope

    async def audio_consumer():
        """Consume audio data from the buffer and play it."""
        with sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype='int16') as stream:
            while True:
                audio_samples = await audio_buffer.get()  # Get audio data from buffer
                stream.write(audio_samples)  # Write to output stream
                audio_buffer.task_done()

    async def audio_producer():
        """Produce audio data by receiving it from the server."""
        nonlocal writer
        try:
            # Attempt to open a connection
            reader, writer = await asyncio.open_connection(host, port)
            logging.info("AUDIO server Connected.")

            # Initialize Opus decoder
            decoder = Decoder(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)

            while True:
                # Read the frame size (4 bytes)
                frame_size_data = await reader.readexactly(4)
                frame_size = struct.unpack(">I", frame_size_data)[0]

                # Read the encoded audio frame
                encoded_audio = await reader.readexactly(frame_size)

                # Remove padding if present
                if len(encoded_audio) > 0 and encoded_audio[-1] == 0:
                    encoded_audio = encoded_audio[:-1]

                # Decode Opus audio to PCM int16 samples
                audio_samples = decoder.decode(encoded_audio, FRAME_SIZE)

                # Add decoded samples to the buffer
                await audio_buffer.put(np.frombuffer(audio_samples, dtype=np.int16))
                logging.debug(f"Buffered audio frame of size: {len(audio_samples)} samples")
        except asyncio.IncompleteReadError:
            logging.error("Audio server disconnected unexpectedly.")
        except Exception as e:
            logging.error(f"Error in audio_producer: {e}")
        finally:
            # Safely close the writer if it was initialized
            if writer:
                writer.close()
                await writer.wait_closed()
            logging.info("AUDIO server Disconnected.")

    # Run producer and consumer concurrently
    await asyncio.gather(audio_producer(), audio_consumer())
    
    

