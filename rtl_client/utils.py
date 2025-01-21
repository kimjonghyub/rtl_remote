import pygame
import numpy as np
import scipy.signal
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import pygame.gfxdraw
from constants import (MIN_DB, MAX_DB, WIDTH, HEIGHT, BLACK, WHITE, BLUE_GRAY, WATERFALL_HEIGHT, MARGIN,BUTTON_WIDTH, BUTTON_HEIGHT, FPS, BLACK, WHITE, GREEN, RED, GRAY, SCREEN)

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
waterfall_buffer = np.zeros((WATERFALL_HEIGHT, WIDTH-MARGIN), dtype=np.uint8)
waterfall_surface = pygame.Surface((WIDTH-MARGIN, WATERFALL_HEIGHT))
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
rig_frequency = 00000000

freq_dot_font = pygame.font.SysFont(None, 70)
freq_mhz_font = pygame.font.SysFont(None, 30)
freq_font = pygame.font.Font("Digital Display.ttf", 60)


def ctlbox():
    global smoothed_center_magnitude
    ctl_box = pygame.Surface(((WIDTH), BUTTON_AREA_HEIGHT) )
    ctl_box.fill(GRAY)
    screen.blit(ctl_box, (0,0))
    #msg = "Server IP : %s" % (SERVER_IP) 
    #screen.blit(font.render(msg, 1, WHITE, GRAY),(10,60))
    #msg = "Port : %s" % (SERVER_PORT) 
    #screen.blit(font.render(msg, 1, WHITE, GRAY),(220,60))
    #msg = "%sMhz" % (format_frequency(rig_frequency)) 
    #screen.blit(mzfont.render(msg, 1, WHITE, GRAY),(10,60))
    draw_frequency(rig_frequency, 10, 60)
    meter_rect = pygame.Rect(10, 120, 300, 20)  
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
    screen.blit(font.render(msg, 1, WHITE, GRAY),(meter_rect.x + meter_rect.width + 10, meter_rect.y))

def text_objects(text, text_font, color):
    textSurface = font.render(text, 1, color)
    return textSurface, textSurface.get_rect()

def button(msg,x,y,w,h,ic,ac,tc,hc,action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(screen, ac,(x,y,w,h))
        text_color = hc
        if click[0] == 1 and action != None:
            text_color = tc
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
    screen.blit(surface, (MARGIN/2, HEIGHT - WATERFALL_HEIGHT))

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
    graph_y = BUTTON_AREA_HEIGHT + MARGIN/2
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
       
def draw_button():
    """
    pygame.draw.rect(screen, button_color, button_rect)
    text = font.render(button_text, True, (text_color))
    screen.blit(text, (button_rect.x + (BUTTON_WIDTH - text.get_width()) // 2,
                       button_rect.y + (BUTTON_HEIGHT - text.get_height()) // 2))
    """
    button("connect",(WIDTH-390),10,100,35,RED,SCREEN,GRAY,WHITE)
    button("256Khz",(WIDTH-280),10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(256))
    button("512Khz",(WIDTH-190),10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(512))
    button("1024Khz",(WIDTH-100),10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(1024))
    button("Rig Freq", (WIDTH - 100), 55, 80, 35, BLUE_GRAY, SCREEN, WHITE, RED, get_rig_frequency)
    
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
      
