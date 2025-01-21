import asyncio
import struct
import pygame
import numpy as np
from utils import process_iq_data, compute_fft, update_waterfall, draw_waterfall, draw_fft_graph, draw_button, ctlbox
from constants import (
    SERVER_IP, SERVER_PORT, BUFFER_SIZE, WIDTH, HEIGHT, FPS,
    BLACK, WHITE, BUTTON_AREA_HEIGHT, FREQ_RANGE, SAMPLE_RATE
)

MAGIC_HEADER = b"RTL0"
CMD_SET_FREQ = 0x01
CMD_SET_SAMPLERATE = 0x02
CMD_SET_GAIN = 0x04
CMD_SET_FREQ_CORRECTION = 0x05
CMD_SET_AGC_MODE = 0x08
CMD_SET_DIRECT_SAMPLING = 0x09
CMD_SET_GAIN_MODE = 0x03
CMD_SET_OFFSET_TUNING = 0x0A


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RTL-TCP IQ Data Visualization")
clock = pygame.time.Clock()

data_queue = asyncio.Queue(maxsize=2)

async def set_freq_range(freq_range):
    global FREQ_RANGE, tasks
    FREQ_RANGE = freq_range
    if writer:
        try:
            #await send_command(writer, CMD_SET_SAMPLERATE, SAMPLE_RATE)
            #await send_command(writer, CMD_SET_FREQ_CORRECTION, FREQ_CORRECTION)
            await send_command(writer, CMD_SET_FREQ, FREQ)
            #await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
            #await send_command(writer, CMD_SET_DIRECT_SAMPLING, DIRECT_SAMPLING)
            #await send_command(writer, CMD_SET_GAIN_MODE, GAIN_MODE)
            #print(f"set {SAMPLE_RATE}")
            
        except Exception as e:
            print(f"set fail: {e}")
    print(f"set freq range : {FREQ_RANGE}kHz")

def process_signal(complex_signal, freq_range):
    global FPS, BUFFER_SIZE
    if freq_range == 256:
        complex_signal = complex_signal[::4]  
    elif freq_range == 512:
        complex_signal = complex_signal[::2]  
    elif freq_range == 1024:
        complex_signal = complex_signal[::1]  
    return complex_signal

async def send_command(writer, cmd_id, param):
    command = struct.pack(">BI", cmd_id, param)
    writer.write(command)
    await writer.drain()

async def receive_data(reader, queue):
    buffer = b""
    global is_connected
    try:
        while is_connected:
            data = await reader.read(BUFFER_SIZE)
            if not data:
                if is_connected:
                    print("No data received. Connection may be closed.")
                break
            buffer += data
            while len(buffer) >= BUFFER_SIZE:
                iq_data = buffer[:BUFFER_SIZE]
                buffer = buffer[BUFFER_SIZE:]
                if queue.full():
                    await queue.get()
                await queue.put(iq_data)
    except Exception as e:
        print(f"Error receiving data: {e}")
    finally:
         is_connected = False

async def handle_connection():
    global reader, writer, is_connected, rig_frequency
    try:
        reader, writer = await asyncio.open_connection(SERVER_IP, SERVER_PORT)
        await send_command(writer, CMD_SET_SAMPLERATE, SAMPLE_RATE)
        await send_command(writer, CMD_SET_FREQ_CORRECTION, FREQ_CORRECTION)
        await send_command(writer, CMD_SET_FREQ, FREQ)
        await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
        await send_command(writer, CMD_SET_DIRECT_SAMPLING, DIRECT_SAMPLING)
        await send_command(writer, CMD_SET_GAIN_MODE, GAIN_MODE)
        await send_command(writer, CMD_SET_GAIN, GAIN)
                
        is_connected = True
        print("IQ server Connected.")
        
        await asyncio.sleep(0.5)  
    except Exception as e:
        print(f"IQ server Connection failed: {e}")
        is_connected = False
        
async def close_connection():
    global reader, writer, is_connected, data_queue

    if writer:
        try:
            writer.write(b"END")
            await writer.drain()  
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            print(f"Error while disconnecting: {e}")
    reader, writer = None, None
    is_connected = False
   
    while not data_queue.empty():
        try:
            data_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    print("IQ server Disconnected.")

async def toggle_connection():
    global is_connected, tasks
    if is_connected:
        #print("IQ server Disconnecting...")
        is_connected = False
        
        for task in tasks:
            task.cancel()            
            try:
                await task
            except asyncio.CancelledError:
                pass
        tasks.clear()
        await close_connection()
        await close_rig_connection()  
    else:
        #print("IQ server Connecting...")
        await handle_connection()
        await handle_rig_connection()  
        if is_connected:
            tasks.append(asyncio.create_task(receive_data(reader, data_queue)))
      
