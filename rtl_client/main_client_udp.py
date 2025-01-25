import multiprocessing
import asyncio
import struct
import pygame
import numpy as np
import scipy.signal
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import pygame.gfxdraw
import logging
import subprocess
import signal
import threading
import queue
from asyncio import Lock
import time

BLACK =    (  0,   0,   0)
WHITE =    (255, 255, 255)
GREEN =    (  0, 255,   0)
BLUE =     (  0,   0, 255)
RED =      (255,   0,   0)
YELLOW =   (192, 192,   0)
DARK_RED = (128,   0,   0)
LITE_RED = (255, 100, 100)
BGCOLOR =  (255, 230, 200)
BLUE_GRAY= (100, 100, 180)
ORANGE =   (255, 150,   0)
GRAY =     ( 60,  60,  60)
SCREEN =   (254, 165,   0)

MAGIC_HEADER = b"RTL0"
CMD_SET_FREQ = 0x01
CMD_SET_SAMPLERATE = 0x02
CMD_SET_GAIN = 0x04
CMD_SET_FREQ_CORRECTION = 0x05
CMD_SET_AGC_MODE = 0x08
CMD_SET_DIRECT_SAMPLING = 0x09
CMD_SET_GAIN_MODE = 0x03
CMD_SET_OFFSET_TUNING = 0x0A
CMD_RIG_SET_FREQ = 0x20
CMD_RIG_GET_FREQ = 0x21
CMD_RIG_SET_VFO = 0x22
CMD_RIG_SET_MODE = 0x23
FRAME_SIZE = 512
BUFFER_SIZE = FRAME_SIZE * 20
FREQ_RANGE = 256

SERVER_IP = "222.117.38.98"
SERVER_PORT = 5611
RIG_PORT = 5622
AUDIO_PORT = 5633
FREQ = 69011500
SAMPLE_RATE = 1024000
GAIN = 496
FREQ_CORRECTION = 0
AGC_MODE = 1
DIRECT_SAMPLING = 0
GAIN_MODE = 0

WIDTH, HEIGHT = 400, 800
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 35
FPS = 20
BUTTON_AREA_HEIGHT = 200
MARGIN = 50
FFT_GRAPH_HEIGHT = (HEIGHT - BUTTON_AREA_HEIGHT) // 2
WATERFALL_HEIGHT = HEIGHT - BUTTON_AREA_HEIGHT - FFT_GRAPH_HEIGHT
MIN_DB = -50
MAX_DB = 120
MODE = "000"
VFO = "A"

class SmoothValue:
    def __init__(self, smoothing_factor=0.1):
        self.smoothing_factor = smoothing_factor
        self.value = None

    def update(self, new_value):
        if self.value is None:
              self.value = new_value
        else:
              self.value = (self.smoothing_factor * new_value) + ((1 - self.smoothing_factor) * self.value)
        return self.value

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RTL-TCP IQ Data Visualization")
clock = pygame.time.Clock()


data_queue = asyncio.Queue(maxsize=2)
rig_command_queue = asyncio.Queue(maxsize=10)
rig_response_queue = asyncio.Queue(maxsize=10)

rig_connected = False
is_connected = False
reader = None
writer = None
rig_reader = None
rig_writer = None
tasks = []  
audio_process = None

waterfall_buffer = np.zeros((WATERFALL_HEIGHT, WIDTH - MARGIN), dtype=np.uint8)
waterfall_surface = pygame.Surface((WIDTH - MARGIN, WATERFALL_HEIGHT))
COLORMAP = np.zeros((256, 3), dtype=np.uint8)
for i in range(256):
    if i < 128: 
        r = 0
        g = int(i * 2)       
        b = int(255 - i * 2) 
    else: 
        r = int((i - 128) * 2) 
        g = int(255 - (i - 128) * 2) 
        b = 0

    COLORMAP[i] = [r, g, b]
    
center_magnitude_smoother = SmoothValue(smoothing_factor=0.1)
center_magnitude = 0
smoothed_center_magnitude = 0

button_rect = pygame.Rect((10, 10), (BUTTON_WIDTH, BUTTON_HEIGHT))
button_color = (RED) 
button_text = "Connect"
text_color = (BLACK)
font = pygame.font.SysFont(None, 24)
font_tick = pygame.font.SysFont(None, 20)
rig_frequency = 0000000

freq_dot_font = pygame.font.SysFont(None, 70)
freq_mhz_font = pygame.font.SysFont(None, 30)
freq_font = pygame.font.Font("Digital Display.ttf", 60)


def fft_worker(input_queue, output_queue, sample_rate):
    
    while True:
        try:
            complex_signal = input_queue.get()
            if complex_signal is None: 
                break
            window = scipy.signal.get_window("hann", len(complex_signal))
            windowed_signal = complex_signal * window
            fft_data = np.fft.fftshift(np.fft.fft(windowed_signal))
            fft_magnitude = 20 * np.log10(np.abs(fft_data) + 1e-6)
            freqs = np.fft.fftshift(np.fft.fftfreq(len(complex_signal), d=1/sample_rate))
            output_queue.put((freqs, fft_magnitude))
        except Exception as e:
            print(f"Error in FFT worker: {e}")

fft_input_queue = queue.Queue()
fft_output_queue = queue.Queue()
fft_thread = threading.Thread(target=fft_worker, args=(fft_input_queue, fft_output_queue, SAMPLE_RATE))
fft_thread.start()

def run_audio_client_as_subprocess():
    global audio_process
    if audio_process is None or audio_process.poll() is not None:
        audio_process = subprocess.Popen(["python3", "audio_client_udp.py"])

def stop_audio_client_subprocess():
    global audio_process
    if audio_process and audio_process.poll() is None:
        audio_process.terminate()

def ctlbox():
    global smoothed_center_magnitude, MODE, VFO, rig_frequency
    ctl_box = pygame.Surface(((WIDTH), BUTTON_AREA_HEIGHT))
    ctl_box.fill(GRAY)
    screen.blit(ctl_box, (0, 0))

    # Display current mode
    msg = f"Mode: {MODE}"
    screen.blit(font.render(msg, True, WHITE, GRAY), (10, 130))

    # Display current VFO
    msg = f"VFO: {VFO}"
    screen.blit(font.render(msg, True, WHITE, GRAY), (130, 130))

    # Display frequency
    draw_frequency(rig_frequency, 10, 55)

    # Display signal strength meter
    meter_rect = pygame.Rect(10, 160, 290, 25)
    draw_level_meter(screen, smoothed_center_magnitude, meter_rect)
    
    

    
    
def format_frequency_parts(freq_hz):
    rigfreq1 = str(freq_hz).zfill(8)  
    integer_part = rigfreq1[0:2] + rigfreq1[2:5] + rigfreq1[5:]
    separators = [".", "."]
    unit = "MHz"
    return integer_part, separators, unit
    
def draw_frequency(freq_hz, x, y):
    
    integer_part, separators, unit = format_frequency_parts(freq_hz)
    rendered_parts = []
    for idx, char in enumerate(integer_part):
        if idx in {2, 5}: 
            rendered_parts.append(freq_dot_font.render(separators.pop(0), True, (WHITE)))
        rendered_parts.append(freq_font.render(char, True, (WHITE)))

    rendered_unit = freq_mhz_font.render(unit, True, (WHITE))
    offset_x = x
    for part in rendered_parts:
        screen.blit(part, (offset_x, y))
        offset_x += part.get_width()
    screen.blit(rendered_unit, (offset_x + 5, y))  
    
def draw_level_meter(screen, center_magnitude, meter_rect):
    min_dB = 0  
    max_dB = 150 
    normalized_value = (smoothed_center_magnitude - min_dB) / (max_dB - min_dB)
    normalized_value = np.clip(normalized_value, 0, 1)  
    pygame.draw.rect(screen, GRAY, meter_rect)
    fill_width = int(normalized_value * meter_rect.width)
    filled_rect = pygame.Rect(meter_rect.x, meter_rect.y, fill_width, meter_rect.height)
    pygame.draw.rect(screen, GREEN, filled_rect)
    pygame.draw.rect(screen, WHITE, meter_rect, 2)
    msg = "%0.2f db" % (smoothed_center_magnitude) 
    screen.blit(font.render(msg, 1, WHITE, GRAY),(meter_rect.x + meter_rect.width - 70, meter_rect.y+5))

def text_objects(text, text_font, color):
    textSurface = font.render(text, 1, color)
    return textSurface, textSurface.get_rect()

last_click_time = 0
click_cooldown = 0.2  # 200ms

def button(msg,x,y,w,h,ic,ac,tc,hc,action=None):
    global last_click_time
    
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(screen, ac,(x,y,w,h))
        text_color = hc
        if click[0] == 1 and action != None:
            current_time = time.time()
            if current_time - last_click_time > click_cooldown: 
                last_click_time = current_time
                text_color = tc
                if action is not None:
                    asyncio.create_task(action())
    else:
        pygame.draw.rect(screen, ic,(x,y,w,h))
        text_color = tc
    textSurf, textRect = text_objects(msg, font, text_color)
    textRect.center = ( (int(x)+(int(w/2))), (int(y)+(int(h/2))) )
    screen.blit(textSurf, textRect)

def upsample_signal(data, factor):
    x = np.arange(len(data))
    x_up = np.linspace(0, len(data) - 1, len(data) * factor)
    spline = make_interp_spline(x, data)
    return spline(x_up)
    
def remove_dc_component(iq_signal):
    real_part = np.real(iq_signal)
    imag_part = np.imag(iq_signal)

    real_part -= np.mean(real_part)
    imag_part -= np.mean(imag_part)

    return real_part + 1j * imag_part

def process_iq_data(iq_data):
    iq_data = np.frombuffer(iq_data, dtype=np.uint8).astype(np.float32)
    iq_data = (iq_data - 127.5) / 127.5
    real = iq_data[1::2]
    imag = iq_data[0::2]
    return real + 1j * imag

async def compute_fft(complex_signal, sample_rate):
    window = scipy.signal.get_window("hann", len(complex_signal))
    windowed_signal = complex_signal * window
    fft_data = np.fft.fftshift(np.fft.fft(windowed_signal))
    fft_magnitude = 20 * np.log10(np.abs(fft_data) + 1e-6)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(complex_signal), d=1/sample_rate))
    return freqs, fft_magnitude

def update_waterfall(buffer, fft_row):
    buffer[1:] = buffer[:-1]
    buffer[0] = np.interp(
        np.linspace(0, len(fft_row) - 1, WIDTH-MARGIN),
        np.arange(len(fft_row)),
        fft_row
    ).astype(np.uint8)

def draw_waterfall(surface, buffer):
    rgb_buffer = COLORMAP[buffer].reshape((WATERFALL_HEIGHT, WIDTH-MARGIN, 3))
    pygame.surfarray.blit_array(surface, np.transpose(rgb_buffer, (1, 0, 2)))
    screen.blit(surface, (MARGIN//2, HEIGHT - WATERFALL_HEIGHT))

def smooth_fft_graph(freqs, magnitudes, num_points=500):
    x_smooth = np.linspace(freqs.min(), freqs.max(), num_points)
    spline = make_interp_spline(freqs, magnitudes, k=3) 
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def smooth_magnitudes(magnitudes, window_length=11, polyorder=2):
    return savgol_filter(magnitudes, window_length, polyorder)
    
def draw_antialiased_line(screen, points, color):
    for i in range(len(points) - 1):
        pygame.draw.aaline(screen, color, points[i], points[i + 1])

def draw_fft_graph(screen, freqs, magnitudes):
    global center_magnitude, smoothed_center_magnitude
    x_smooth, y_smooth = smooth_fft_graph(freqs, magnitudes)
    y_smooth = smooth_magnitudes(y_smooth)
    graph_width = WIDTH - MARGIN
    graph_height = FFT_GRAPH_HEIGHT - MARGIN
    graph_x = MARGIN/2
    graph_y = BUTTON_AREA_HEIGHT + MARGIN // 2
    pygame.draw.rect(screen, GRAY, (graph_x, graph_y, graph_width, graph_height))  
    mid_x = graph_x + graph_width // 2
    x_coords = graph_x + (x_smooth - freqs.min()) / (freqs.max() - freqs.min()) * graph_width
    denominator = magnitudes.max() - magnitudes.min()
    y_coords = graph_y + graph_height - (
    np.divide(
        (y_smooth - magnitudes.min() + 5),
        denominator,
        out=np.zeros_like(y_smooth),
        where=denominator != 0
    ) * graph_height)
    y_coords = np.clip(y_coords, graph_y, graph_y + graph_height)
    points = np.column_stack((x_coords, y_coords)).astype(int)
    polygon_points = [(graph_x, graph_y + graph_height)] + points.tolist() + [
        (graph_x + graph_width, graph_y + graph_height)
    ]
    pygame.draw.polygon(screen, BLUE_GRAY, polygon_points)
    draw_antialiased_line(screen, points.tolist(), WHITE)
    
    # Draw x-axis
    pygame.draw.line(
        screen, RED, 
        (graph_x, graph_y + graph_height - 1), 
        (graph_x + graph_width, graph_y + graph_height - 1), 1)
    max_freq = SAMPLE_RATE / 1 / (1024 // FREQ_RANGE) 
    num_xticks = 4  
    tick_spacing = graph_width // num_xticks 
    
    for i in range(num_xticks + 1):
        x_pos = i * tick_spacing
        freq = int((i - num_xticks // 2) * (max_freq / num_xticks) / 1e3)  
        tick_label = f"{freq}kHz"
        pygame.draw.line(
            screen, DARK_RED, 
            (x_pos+graph_x, graph_y), #+ graph_height - 5), 
            (x_pos+graph_x, graph_y + graph_height + 5), 1)
        label = font_tick.render(tick_label, True, WHITE)
        label_rect = label.get_rect(center=(x_pos+graph_x, graph_y + graph_height + 15))
        screen.blit(label, label_rect)
        
    # Draw y-axis 
    num_yticks = 5
    tick_spacing = graph_height / (num_yticks - 1)

    for i in range(num_yticks):
        y_tick = graph_y + i * tick_spacing
        tick_value = magnitudes.min() + (magnitudes.max() - magnitudes.min()) * (num_yticks - 1 - i) / (num_yticks - 1)
        pygame.draw.line(
            screen, DARK_RED,
            (graph_x - 10, int(y_tick)),
            (graph_x+graph_width, int(y_tick)), 1)
        label = font_tick.render(f"{tick_value:.0f} dB", True, WHITE)
        screen.blit(label, (graph_x - 20, int(y_tick) - label.get_height() // 2))
    center_x = graph_x + (0 - freqs.min()) / (freqs.max() - freqs.min()) * graph_width
    center_index = np.argmin(np.abs(freqs - 0))
    raw_center_magnitude = magnitudes[center_index]
    smoothed_center_magnitude = center_magnitude_smoother.update(raw_center_magnitude)

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
        stop_audio_client_subprocess()
        
        
    else:
        #print("IQ server Connecting...")
        await handle_connection()
        await handle_rig_connection()  
        run_audio_client_as_subprocess()
        if is_connected:
            tasks.append(asyncio.create_task(receive_data(reader, data_queue)))
      
async def handle_rig_connection():
    global rig_reader, rig_writer, rig_connected
    try:
        rig_reader, rig_writer = await asyncio.open_connection(SERVER_IP, RIG_PORT)
        rig_connected = True
        print("RIG server Connected.")
        
        await asyncio.sleep(0.5) 
        asyncio.create_task(rig_reader_task())
        asyncio.create_task(rig_command_task())
        asyncio.create_task(set_rig_vfo(0))
        #mode_index = 2
        #asyncio.create_task(set_rig_mode())           
    except Exception as e:
        print(f"Failed to connect to RIG server: {e}")
        rig_connected = False
"""
async def rig_reader_task():
    
    global rig_reader, rig_connected, rig_frequency
    try:
        while rig_connected:
            try:
                # Read 4 bytes from the server
                data = await rig_reader.readexactly(4)
                #print(f"Raw data received: {data}")
                freq = struct.unpack(">I", data)[0]
                if 100000 <= freq <= 30000000:  
                    rig_frequency = freq
                    print(f"Received frequency notification: {freq} Hz")
                #else:
                    #print(f"Invalid frequency received: {freq} Hz")
 
            except asyncio.IncompleteReadError as e:
                
                rig_connected = False
                break

            except Exception as e:
                print(f"Error in rig_reader_task: {e}")

    except asyncio.CancelledError:
        print("RIG reader task canceled.")
    #finally:
        #print("RIG reader task stopped.")
"""
modes = [1, 2, 3, 4, 5]
mode_names = {
    1: "USB",
    2: "LSB",
    3: "AM",
    4: "FM",
    5: "CW"
}
mode_index = 0
mode_lock = Lock()

vfo_names = {
    11: "A",
    12: "B",
}

async def rig_reader_task():
    """Reads data from the RIG server and processes it."""
    global rig_reader, rig_connected, rig_frequency, MODE
    try:
        while rig_connected:
            try:
                # Read 4 bytes from the server
                data = await rig_reader.readexactly(4)

                # Interpret the received data
                value = struct.unpack(">I", data)[0]
                if value in mode_names:  # Check if value corresponds to a mode
                    MODE = mode_names[value]
                    print(f"Received mode notification: {MODE}")
                elif value in vfo_names:  # Check if value corresponds to a VFO
                    VFO = vfo_names[value]
                    print(f"Received VFO notification: {VFO}")
                elif 100000 <= value <= 30000000:  # Check if value is a valid frequency
                    rig_frequency = value
                    print(f"Received frequency notification: {rig_frequency} Hz")
                else:
                    print(f"Received unknown value: {value}")
            except asyncio.IncompleteReadError:
                rig_connected = False
                break
            except Exception as e:
                print(f"Error in rig_reader_task: {e}")
    except asyncio.CancelledError:
        print("RIG reader task canceled.")
        
async def rig_command_task():
    """Processes commands and waits for responses."""
    global rig_writer, rig_connected
    try:
        while rig_connected:
            response_event = None
            try:
                # Wait for a command in the queue
                if not rig_command_queue.empty():
                    command, response_event = await rig_command_queue.get()
                    # Send the command to the server
                    rig_writer.write(command)
                    await rig_writer.drain()
                    if response_event:
                        response_event.set()

            except Exception as e:
                print(f"Error in rig_command_task: {e}")
                if response_event:
                    response_event.set()

            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        print("RIG command task canceled.")
    finally:
        print("RIG command task stopped.")
    
async def close_rig_connection():
    global rig_reader, rig_writer, rig_connected, rig_command_queue
    if rig_writer:
        try:
            rig_writer.close()
            await rig_writer.wait_closed()
        except Exception as e:
            print(f"Error closing RIG server connection: {e}")
    rig_reader, rig_writer = None, None
    rig_connected = False
    while not rig_command_queue.empty():
        try:
            rig_command_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    print("Rig server Disconnected.")
    
async def toggle_rig_connection():
    global rig_connected
    if rig_connected:
        print("Disconnecting from RIG server...")
        await close_rig_connection()
    else:
        print("Connecting to RIG server...")
        await handle_rig_connection()

async def set_rig_vfo(PARAM):
    """Send CMD_RIG_GET_FREQ and wait for response."""
    global rig_command_queue, VFO

    # Prepare command and enqueue it
    cmd_id = CMD_RIG_SET_VFO
    param = PARAM
    command = struct.pack(">BI", cmd_id, param)
    response_event = asyncio.Event()

    await rig_command_queue.put((command, response_event))
    #print(f"Sending command to RIG server: cmd_id={cmd_id}, param={param}")

    # Wait for the response or timeout
    await asyncio.wait_for(response_event.wait(), timeout=1.0)
    #print(f"Frequency received: {rig_frequency} Hz")
    vfo_name = "A" if param == 0 else "B"
    VFO = vfo_name
    #logging.info(f"VFO changed to: {vfo_name}")
    


async def set_rig_mode():
    """Change the mode and send it to the server."""
    global mode_index, MODE
    async with mode_lock:
        
        mode = modes[mode_index]
        mode_name = mode_names.get(mode, "Unknown")  
        MODE = mode_name
        try:
            # Prepare command
            cmd_id = CMD_RIG_SET_MODE
            command = struct.pack(">BI", cmd_id, mode)
            response_event = asyncio.Event()

            # Enqueue the command
            await rig_command_queue.put((command, response_event))
            print(f"Sending mode change command: {mode} ({mode_name})")

            # Wait for response or timeout
            await asyncio.wait_for(response_event.wait(), timeout=1.0)
            print(f"Mode changed to {mode_name} successfully.")
        except Exception as e:
            print(f"Error in change_mode: {e}")

        # Update mode index for the next call
        mode_index = (mode_index + 1) % len(modes)
    
       
def draw_button():
    
    button("connect",10,10,100,35,RED,SCREEN,GRAY,WHITE)
    button("256Khz",130,10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(256))
    button("512Khz",220,10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(512))
    button("1024Khz",310,10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(1024))
    button("A", 310, 55, 35, 35, BLUE_GRAY, SCREEN, WHITE, RED, lambda:set_rig_vfo(0))
    button("B", 355, 55, 35, 35, BLUE_GRAY, SCREEN, WHITE, RED, lambda:set_rig_vfo(1))
    button("Mode", 310, 100, 80, 35, BLUE_GRAY, SCREEN, WHITE, RED, lambda:set_rig_mode())
  
async def shutdown():
    """Cancel all pending tasks."""
    logging.info("Shutting down...")
    pending_tasks = asyncio.all_tasks() 
    for task in pending_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logging.info(f"Task {task.get_name()} cancelled.")
      
async def main():
    global button_color, button_text, is_connected, fft_input_queue, fft_output_queue
    running = True
  
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                fft_input_queue.put(None)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    if is_connected:
                        button_color = (RED)  
                        button_text = "Connect"
                        await toggle_connection()
                        stop_audio_client_subprocess()
                        
                        for task in tasks:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        tasks.clear()
                                                
                    else:
                        button_color = (GREEN)  
                        button_text = "Disconnect"
                        await toggle_connection()
                        #if is_connected:
                            
                        
                       
        screen.fill((0, 0, 0))
        ctlbox()
        
        if is_connected:
            try:
                iq_data = await data_queue.get()
                complex_signal = process_signal(process_iq_data(iq_data), FREQ_RANGE)
                complex_signal = remove_dc_component(complex_signal)
                
                if not fft_input_queue.full():
                    fft_input_queue.put(complex_signal)
                
                if not fft_output_queue.empty():
                    freqs, magnitudes = fft_output_queue.get()
                    magnitudes = np.clip(magnitudes, -30, 100)
                    exponent = 4  
                    processed_magnitudes = (magnitudes - magnitudes.min()) ** exponent
                    scaled_magnitudes = np.interp(
                        processed_magnitudes, [processed_magnitudes.min(), processed_magnitudes.max()], [0, 255]
                    ).astype(np.uint8)
                
                    wf_magnitudes = np.interp(magnitudes, [MIN_DB, MAX_DB], [0, 255]).astype(np.uint8)
                    update_waterfall(waterfall_buffer, wf_magnitudes)
                    draw_waterfall(waterfall_surface, waterfall_buffer)
                    draw_fft_graph(screen, freqs, scaled_magnitudes)
                
            
            except asyncio.QueueEmpty:
                pass
        
        draw_button()
        pygame.display.flip()
        clock.tick(FPS)

    if is_connected:
        await toggle_connection()
        await toggle_rig_connection()
    
    fft_input_queue.put(None)
    fft_thread.join()    
    stop_audio_client_subprocess()
    pygame.quit()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stop_audio_client_subprocess()
        print("Shutting down...")
    finally:
        asyncio.run(shutdown())
