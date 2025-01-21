
import asyncio
import logging
import struct
import numpy as np
import sounddevice as sd
from opuslib import Decoder
import signal

# Server Configuration
SERVER_IP = "222.117.38.98"
AUDIO_PORT = 5633

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000  
AUDIO_CHANNELS = 1
FRAME_DURATION = 20 
AUDIO_FRAME_SIZE = int(AUDIO_SAMPLE_RATE * FRAME_DURATION / 1000)

#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
decoder = Decoder(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)


QUEUE_MAXSIZE = 10

async def process_audio_queue(audio_queue):
   
    with sd.OutputStream(
        samplerate=AUDIO_SAMPLE_RATE, 
        channels=AUDIO_CHANNELS, 
        dtype='int16',
        latency='high',
    ) as stream:
        while True:
            audio_samples = await audio_queue.get()
            if audio_samples is not None:
                    stream.write(audio_samples)
            audio_queue.task_done()

async def connect_to_audio_server(host, port):
   
    writer = None
    audio_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    audio_task = asyncio.create_task(process_audio_queue(audio_queue))

    try:
        #logging.info(f"Attempting to connect to audio server at {host}:{port}")
        reader, writer = await asyncio.open_connection(host, port)
        #logging.info(f"Connected to audio server at {host}:{port}")

        while True:
            frame_size_data = await reader.readexactly(4)
            frame_size = struct.unpack(">I", frame_size_data)[0]
            encoded_audio = await reader.readexactly(frame_size)
            decoded_audio = decoder.decode(encoded_audio, AUDIO_FRAME_SIZE)
            audio_samples = np.frombuffer(decoded_audio, dtype=np.int16)
            if len(audio_samples) % 2 != 0:
                audio_samples = np.append(audio_samples, [0], axis=0)
            if audio_queue.full():
                audio_queue.get_nowait()
            audio_queue.put_nowait(audio_samples)
    except asyncio.CancelledError:
        logging.info("connect_to_audio_server() Cancle.")
        raise
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if writer:
            writer.close()
            await writer.wait_closed()
        audio_task.cancel()
        try:
            await audio_task
        except asyncio.CancelledError:
            logging.info("Audio processing task cancelled.")
        logging.info("Audio connection closed.")

async def main():
 
    loop = asyncio.get_running_loop()

    def handle_termination_signal(signal_num, frame):
        logging.info(f"Termination signal ({signal_num}) received.")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    
    signal.signal(signal.SIGTERM, handle_termination_signal)
    signal.signal(signal.SIGINT, handle_termination_signal)

    try:
        await connect_to_audio_server(SERVER_IP, AUDIO_PORT)
    except asyncio.CancelledError:
        logging.info("Main task cancelled.")
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
    finally:
        logging.info("Main task finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Audio client stopped manually.")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        sys.exit(1)
