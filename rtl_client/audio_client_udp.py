import socket
import struct
import numpy as np
import sounddevice as sd
from opuslib import Decoder
import logging
SERVER_IP = "222.117.38.98"
AUDIO_PORT = 5633
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 1
FRAME_DURATION = 100  # ms
FRAME_SIZE = int(AUDIO_SAMPLE_RATE * FRAME_DURATION / 1000)
DATA_BUFFER = 1024
decoder = Decoder(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5.0)
    try:
        sock.sendto(b"", (SERVER_IP, AUDIO_PORT))
        print(f"Audio server at {SERVER_IP}:{AUDIO_PORT}")
    except Exception as e:
        print(f"Failed to send initial packet: {e}")
        return
        
    #sock.bind(("", AUDIO_PORT))
    #sock.sendto(b"", (SERVER_IP, AUDIO_PORT))  0
    

    try:
        with sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype='int16') as stream:
            while True:
                try:
                   
                    data, addr = sock.recvfrom(DATA_BUFFER)
                    logging.info(f"Received data of size {len(data)} from server")
                   
                    frame_size = struct.unpack(">I", data[:4])[0]
                    encoded_audio = data[4:4 + frame_size]
                    decoded_audio = decoder.decode(encoded_audio, FRAME_SIZE)
                    
                   
                    audio_samples = np.frombuffer(decoded_audio, dtype=np.int16)
                    stream.write(audio_samples)

                except socket.timeout:
                    print("No data received from server. Retrying...")
                except Exception as e:
                    print(f"Error receiving or processing audio: {e}")
                    break

    except Exception as e:
        print(f"Error initializing audio stream: {e}")
    finally:
        sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()
