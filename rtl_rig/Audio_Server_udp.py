import asyncio
import logging
import struct
import numpy as np
import sounddevice as sd
from opuslib import Encoder

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(message)s")

SERVER_IP = "0.0.0.0"  # Listen on all interfaces
AUDIO_PORT = 5633

application = 'voip'
bitrate = 64000  # 64kbps

AUDIO_SAMPLE_RATE = 48000  # Hz
AUDIO_CHANNELS = 1
FRAME_DURATION = 100  # ms
FRAME_SIZE = int(AUDIO_SAMPLE_RATE * FRAME_DURATION / 1000)
#print("Available audio input devices:")
#print(sd.query_devices())
#device_id = 0  
#info = sd.query_devices(device_id, "input")
#print(info)
class AudioHandler:
    def __init__(self, queue_maxsize=50):
        self.audio_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.encoder = Encoder(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, application)
        self.encoder.bitrate = bitrate
        self.clients = set()

    async def start_audio_capture(self):
        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio stream status: {status}")
    
            if indata is not None:
                mono_audio = indata[:, 0] 
                #logging.info(f"Captured audio data: {mono_audio[:10]}... (size: {len(mono_audio)})")

                if self.audio_queue.full():
                    self.audio_queue.get_nowait()  
                self.audio_queue.put_nowait(mono_audio.copy())
                #logging.info("Captured audio data added to queue.")
            else:
                logging.error("No audio data captured.")

        try:
            with sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype='int16',
                callback=callback,
                blocksize=FRAME_SIZE,
                device=0,
            ):
                logging.info("Audio capturing started...")
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            logging.info("Audio capturing task cancelled.")
        except Exception as e:
            logging.error(f"Audio capture error: {e}")

    async def send_audio_to_clients(self, transport):
        try:
            while True:
               
                if not self.clients:
                    #logging.warning("No clients connected. Skipping audio frame transmission.")
                    await asyncio.sleep(FRAME_DURATION / 1000)
                    continue
                try:
                     raw_audio = await asyncio.wait_for(self.audio_queue.get(), timeout=FRAME_DURATION / 1000)
                except asyncio.TimeoutError:
                    continue  
                #logging.info("Captured audio data added to queue.")
                encoded_audio = self.encoder.encode(raw_audio.tobytes(), FRAME_SIZE)
                packet = struct.pack(">I", len(encoded_audio)) + encoded_audio

                for client in self.clients:
                    try:
                        transport.sendto(packet, client)
                        logging.debug(f"Sent audio to {client}")
                    except Exception as e:
                        logging.error(f"Error sending to {client}: {e}")
                        self.clients.remove(client)
        except asyncio.CancelledError:
            logging.info("Audio sending task cancelled.")
        except Exception as e:
            logging.error(f"Error sending audio: {e}")

    def add_client(self, addr):
        """Add a client to the list."""
        if addr not in self.clients:
            self.clients.add(addr)
            logging.info(f"Client {addr} added. Total clients: {len(self.clients)}")
        else:
            logging.info(f"Client {addr} is already connected.")
    def remove_client(self, addr):
        """Remove a client from the list."""
        if addr in self.clients:
            self.clients.remove(addr)
            logging.info(f"Client {addr} removed. Total clients: {len(self.clients)}")

    def get_clients(self):
        """Return a list of connected clients."""
        return list(self.clients)

    async def send_initial_audio_to_client(self, addr, transport):
        """Send the first audio packet to a newly added client."""
        try:
            if self.audio_queue.empty():
                logging.warning("Audio queue is empty. Unable to send initial audio packet.")
                return

            raw_audio = await self.audio_queue.get()
            encoded_audio = self.encoder.encode(raw_audio.tobytes(), FRAME_SIZE)
            packet = struct.pack(">I", len(encoded_audio)) + encoded_audio

            transport.sendto(packet, addr)
            logging.info(f"Sent initial audio frame to {addr}")
        except Exception as e:
            logging.error(f"Error sending initial audio to {addr}: {e}")


class AudioProtocol(asyncio.DatagramProtocol):
    def __init__(self, audio_handler):
        self.audio_handler = audio_handler

    def connection_made(self, transport):
        self.transport = transport
        logging.info("UDP Audio Server is ready.")

    def datagram_received(self, data, addr):
        self.audio_handler.add_client(addr)
        logging.info(f"Received packet from {addr}")
        asyncio.create_task(self.audio_handler.send_initial_audio_to_client(addr, self.transport))

    def error_received(self, exc):
        logging.error(f"Error received: {exc}")

async def audio_start_server(audio_handler, host, port):
    """Start the UDP server."""
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: AudioProtocol(audio_handler), local_addr=(host, port)
    )
    logging.info(f"Audio Server started on {host}:{port}")

    # Send audio in the background
    audio_task = asyncio.create_task(audio_handler.send_audio_to_clients(transport))

    try:
        await asyncio.Event().wait()  # Keep running
    except asyncio.CancelledError:
        logging.info("Server is shutting down...")
    finally:
        audio_task.cancel()
        await audio_task
        transport.close()

async def main():
    audio_handler = AudioHandler()
    capture_task = asyncio.create_task(audio_handler.start_audio_capture())
    server_task = asyncio.create_task(audio_start_server(audio_handler, SERVER_IP, AUDIO_PORT))
    try:
        await asyncio.gather(capture_task, server_task)
    except asyncio.CancelledError:
        logging.info("Server shutting down...")
    finally:
        capture_task.cancel()
        server_task.cancel()
        await asyncio.gather(capture_task, server_task, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
