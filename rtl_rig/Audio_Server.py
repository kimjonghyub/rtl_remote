#from audio_handler import AudioHandler
import asyncio
import logging
import struct
import numpy as np
import sounddevice as sd
from opuslib import Encoder
import multiprocessing

# Logging setup

logging.basicConfig(level=logging.INFO, format="%(message)s")

SERVER_IP = "192.168.10.216"
AUDIO_PORT = 5633



AUDIO_SAMPLE_RATE = 48000  
AUDIO_CHANNELS = 1
FRAME_DURATION = 10 # ms
FRAME_SIZE = int(AUDIO_SAMPLE_RATE * FRAME_DURATION / 1000)

class AudioHandler:
    def __init__(self,queue_maxsize=100):
        self.audio_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.encoder = Encoder(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, application='voip')
        self.clients = {}
        self.audio_task = None

    async def start_audio_capture(self):
        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio stream status: {status}")
            mono_audio = indata[:, 0]
            if self.audio_queue.full():
                self.audio_queue.get_nowait() 
            self.audio_queue.put_nowait(mono_audio.copy())

        try:
            with sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype='int16',
                callback=callback,
                blocksize=FRAME_SIZE 
            ):
                logging.info("Audio capturing started...")
                await asyncio.Event().wait()  # Keep running
        except asyncio.CancelledError:
            logging.info("Audio capturing task cancelled.")
        except Exception as e:
            logging.error(f"Audio capture error: {e}")

    async def add_client(self, transport, peername):
        """Add a new audio client."""
        logging.info(f"Adding audio client: {peername}")
        send_task = asyncio.create_task(self.send_audio_to_client(transport, peername))
        self.clients[transport] = send_task
        logging.info(f"Client added: {peername}")

    async def send_audio_to_client(self, transport, peername):
        """Continuously send audio data to a client."""
        try:
            while not transport.is_closing():
                raw_audio = await self.audio_queue.get()
                logging.debug(f"Captured raw audio frame of size: {len(raw_audio)}")
                try:
                    encoded_audio = self.encoder.encode(raw_audio.tobytes(), FRAME_SIZE)
                    #if len(encoded_audio) % 2 != 0:
                        #encoded_audio += b'\x00'
                    #if len(encoded_audio) % 2 != 0:
                        #encoded_audio = np.append(encoded_audio, [0], axis=0)
                    transport.write(struct.pack(">I", len(encoded_audio)) + encoded_audio)
                    logging.debug(f"Sent encoded audio frame of size: {len(encoded_audio)} to {peername}")
                    await asyncio.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError) as e:
                    logging.warning(f"Connection lost with {peername}: {e}")
                    break
        except asyncio.CancelledError:
            logging.info(f"Audio transmission to {peername} cancelled.")
        except Exception as e:
            logging.error(f"Error sending audio to {peername}: {e}")
        finally:
            self.remove_client(transport, peername)

    def remove_client(self, transport, peername):
        """Remove a client from the list."""
        if transport in self.clients:
            task = self.clients.pop(transport)
            task.cancel()  # Cancel the audio sending task
            logging.info(f"Client {peername} removed.")
        if not self.clients:
            self.clear_queue()

    def clear_queue(self):
        """Clear the audio queue."""
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
        logging.info("Audio queue cleared as no clients are connected.")

class AudioProtocol(asyncio.Protocol):
    def __init__(self, audio_handler):
        self.audio_handler = audio_handler
        self.transport = None

    def connection_made(self, transport):
        peername = transport.get_extra_info("peername")
        logging.info(f"Audio Client connected: {peername}")
        asyncio.create_task(self.audio_handler.add_client(transport, peername))

    def connection_lost(self, exc):
        peername = self.transport.get_extra_info("peername") if self.transport else "Unknown"
        if exc:
            logging.error(f"Audio client connection lost for {peername} with error: {exc}")
        else:
            logging.info(f"Audio client connection closed cleanly for {peername}")
        self.audio_handler.remove_client(self.transport, peername)
        self.transport = None

async def audio_start_server(audio_handler, host, port):
    """Start the AUDIO server."""
    loop = asyncio.get_running_loop()
    server = await loop.create_server(lambda: AudioProtocol(audio_handler), host, port)
    logging.info(f"Audio Server started on {host}:{port}")
    async with server:
        await server.serve_forever()
