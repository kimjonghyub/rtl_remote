#from rtl_tcp_handler import RtlTcpHandler
import asyncio
import logging
import struct
import numpy as np
from rtlsdr import RtlSdr
import multiprocessing

# Logging setup

logging.basicConfig(level=logging.INFO, format="%(message)s")
# RTL-TCP Command IDs
MAGIC_HEADER = b"RTL0"
CMD_SET_FREQ = 0x01
CMD_SET_SAMPLERATE = 0x02
CMD_SET_GAIN_MODE = 0x03
CMD_SET_GAIN = 0x04
CMD_SET_FREQ_CORRECTION = 0x05
CMD_SET_TUNER_IF_GAIN = 0x06
CMD_SET_TESTMODE = 0x07
CMD_SET_AGC_MODE = 0x08
CMD_SET_DIRECT_SAMPLING = 0x09
CMD_SET_OFFEST_TURNING =  0x0a
CMD_SET_XTAL_FREQ = 0x0b
CMD_SET_TUNER_FREQ = 0x0c
CMD_SET_GAIN_BY_INDEX = 0x0d
CMD_SET_BIAS_TEE = 0x0e
CMD_RIG_SET_FREQ = 0x20
CMD_RIG_GET_FREQ = 0x21
SERVER_IP = "192.168.10.216"
IQ_PORT = 5611



# Default Configuration
DEFAULT_FREQ = 69011500  # 100 MHz
DEFAULT_SAMPLERATE = 1024000  # 2.048 MSPS
DEFAULT_GAIN = 'auto'  # Auto gain



class RtlTcpHandler:
    """Manages the RTL-SDR device operations."""
    def __init__(self):
        self.sdr = None
        self.iq_buffer = asyncio.Queue(maxsize=5)
                
    def initialize_device(self):
        """Initialize the RTL-SDR device."""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = DEFAULT_SAMPLERATE
            self.sdr.center_freq = DEFAULT_FREQ
            self.sdr.gain = DEFAULT_GAIN
            self.sdr.set_bias_tee(False)
            self.sdr.RTLSDR_BUFFER_LENGTH = 512 * 12
            logging.info("RTL-SDR initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize RTL-SDR: {e}")
            raise

    def set_frequency(self, frequency_hz):
        """Set the center frequency."""
        self.sdr.center_freq = frequency_hz

    def set_sample_rate(self, sample_rate_hz):
        """Set the sample rate."""
        self.sdr.sample_rate = sample_rate_hz

    def set_gain_mode(self, auto_gain):
        """Set gain mode."""
        self.sdr.gain = 'auto' if auto_gain else 49.6

    def set_gain(self, gain_db):
        """Set gain."""
        self.sdr.gain = gain_db

    def get_dongle_info(self):
        """Return the dongle information header."""
        tuner_type = self.sdr.get_tuner_type()
        gain_count = len(self.sdr.valid_gains_db)
        return MAGIC_HEADER + struct.pack(">II", tuner_type, gain_count)

    
    async def stream_iq_data(self, transport):
        """Stream raw I/Q data to the client."""
        async for samples in self.sdr.stream():
            iq_data = np.empty(samples.size * 2, dtype=np.uint8)
            real = np.clip((samples.real + 1) * 127.5, 0, 255).astype(np.uint8)
            imag = np.clip((samples.imag + 1) * 127.5, 0, 255).astype(np.uint8)
            iq_data[0::2], iq_data[1::2] = real, imag

            if not self.iq_buffer.full():
                await self.iq_buffer.put(iq_data.tobytes())

    async def send_iq_data(self, transport):
        """Send I/Q data to the client."""
        while True:
            iq_data = await self.iq_buffer.get()
            if transport.is_closing():
                break
            transport.write(iq_data)
            self.iq_buffer.task_done()
            await asyncio.sleep(0.01)

    def stop_streaming(self):
        """Stop the RTL-SDR streaming."""
        if self.sdr:
            self.sdr.cancel_read_async()

    def close(self):
        """Close the RTL-SDR device."""
        if self.sdr:
            self.sdr.close()
            logging.info("RTL-SDR device closed")


class ClientHandler(asyncio.Protocol):
    """Handles client connections and forwards commands to the RTL-SDR device."""
    def __init__(self, rtl_handler):
        self.rtl_handler = rtl_handler
        self.transport = None
        self.streaming_task = None
        self.sending_task = None
        self.buffer = b""

    def connection_made(self, transport):
        self.transport = transport
        peername = transport.get_extra_info("peername")
        logging.info(f"IQ Client connected: {peername}")
        #transport.set_write_buffer_limits(high=512 * 1024)
        # Send dongle information
        dongle_info = self.rtl_handler.get_dongle_info()
        if dongle_info:
            self.transport.write(dongle_info)
            logging.info("Sent dongle information to client")
        else:
            logging.error("Failed to send dongle information")
            self.transport.close()
        self.streaming_task = asyncio.create_task(self.rtl_handler.stream_iq_data(self.transport))
        self.sending_task = asyncio.create_task(self.rtl_handler.send_iq_data(self.transport))

    def data_received(self, data):
        self.buffer += data
        #logging.info(f"Received raw command data: {self.buffer.hex()}")
        asyncio.create_task(self.process_buffer())

    async def process_buffer(self):
        """
         while len(self.buffer) >= 5:
            
            cmd_id = self.buffer[0]
            param = struct.unpack(">I", self.buffer[1:5])[0]
            await self.handle_command(cmd_id, param)
            #logging.info(f"Processed 5-byte command: cmd_id={cmd_id}, param={param}")
            self.buffer = self.buffer[5:]
        """
        while len(self.buffer) >= 3:  
            if self.buffer[:3] == b"END":
                logging.info("IQ Client END Received")
                self.transport.close()
                return  
            if len(self.buffer) >= 5:
                cmd_id = self.buffer[0]
                param = struct.unpack(">I", self.buffer[1:5])[0]
                await self.handle_command(cmd_id, param)
                self.buffer = self.buffer[5:]  
            else:
                break 
    def connection_lost(self, exc):
        logging.info("IQ Client connection closed")
        if self.streaming_task:
            self.streaming_task.cancel()
        if self.sending_task:
            self.sending_task.cancel()
        self.rtl_handler.stop_streaming()
        if self.transport:
            self.transport.close()
        if exc:
            logging.warning(f"IQ Connection lost with error: {exc}")
        else:
            logging.info("IQ Connection closed cleanly")

    async def handle_command(self, cmd_id, param):
        if cmd_id == CMD_SET_FREQ:
            self.rtl_handler.set_frequency(param)
            logging.info(f"set freq {param} Hz")
         
        elif cmd_id == CMD_SET_SAMPLERATE:
            logging.info(f"set sample rate {param}")
            self.rtl_handler.set_sample_rate(param)
        elif cmd_id == CMD_SET_GAIN_MODE:
            logging.info(f"set gain mode {param}")
            if param == 1:  
                self.rtl_handler.set_gain_mode(auto_gain=True)
            elif param == 0:  
                 default_gain = 49.6  
                 self.rtl_handler.set_gain_mode(auto_gain=False)
                 self.rtl_handler.set_gain(default_gain)
            else:
                raise ValueError(f"Invalid gain mode value: {param}")
        elif cmd_id == CMD_SET_GAIN:
            logging.info(f"set gain {param/10.0} db")
            self.rtl_handler.set_gain(param / 10.0)
        elif cmd_id == CMD_SET_FREQ_CORRECTION:
            logging.info(f"set freq correction {param}")
        elif cmd_id == CMD_SET_AGC_MODE:
            logging.info(f"set agc mode {param}")
            self.rtl_handler.sdr.agc_mode = bool(param)  
        elif cmd_id == CMD_SET_DIRECT_SAMPLING:
            logging.info(f"set direct sampling {param}")
            if param in (0, 1, 2):
                self.rtl_handler.sdr.set_direct_sampling(param)
                mode = ["Off", "I-ADC", "Q-ADC"][param]
                logging.info(f"set direct sampling {mode}")
            else:
                raise ValueError(f"Invalid direct sampling mode: {param}")
        elif cmd_id == CMD_SET_OFFEST_TURNING:  
            logging.info(f"set offset tuning: {param}")
            if param in (0, 1):
                #self.rtl_handler.sdr.set_offset_tuning(bool(param))
                mode = "On" if param else "Off"
                logging.info(f"set offset tuning {mode}")
            else:
                raise ValueError(f"Invalid Offset Tuning mode: {param}")
        elif cmd_id == CMD_SET_XTAL_FREQ:  
            logging.info(f"set rtl xtal: {param} Hz")
            if param > 0:
                self.rtl_handler.sdr.set_xtal_freq(rtl_freq=param)
                logging.info(f"set rtl xtal {param} Hz")
            else:
                raise ValueError(f"Invalid RTL oscillator frequency: {param}")
        elif cmd_id == CMD_SET_TUNER_FREQ: 
            logging.info(f"set tuner xtal: {param} Hz")
            if param > 0:
                self.rtl_handler.sdr.set_xtal_freq(tuner_freq=param)
                logging.info(f"set tuner xtal {param} Hz")
            else:
                raise ValueError(f"Invalid Tuner oscillator frequency: {param}")
        elif cmd_id == CMD_SET_GAIN_BY_INDEX: 
            logging.info(f"Client requested tuner gain set by index: {param}")
            valid_gains = self.rtl_handler.sdr.valid_gains_db
            if not valid_gains:
                raise ValueError("No valid gains available for this tuner")
            if 0 <= param < len(valid_gains):
                gain_to_set = valid_gains[param]
                self.rtl_handler.set_gain(gain_to_set)
                logging.info(f"set tuner gain by index {gain_to_set} dB (Index: {param})")
            else:
                raise ValueError(f"Invalid gain index: {param}")
        elif cmd_id == CMD_SET_BIAS_TEE:  
            logging.info(f"set bias tee: {param}")
            if param in (0, 1):
                self.rtl_handler.sdr.set_bias_tee(bool(param))
                mode = "On" if param else "Off"
                logging.info(f"Bias-T mode set to {mode}")
            else:
                raise ValueError(f"Invalid Bias-T mode: {param}")
  
        else:
            logging.warning(f"Unknown command ID: {cmd_id}")


async def iq_start_server(rtl_handler, host, port):
    """Start the IQ server."""
    loop = asyncio.get_running_loop()
    server = await loop.create_server(lambda: ClientHandler(rtl_handler), host, port)
    logging.info(f"IQ Server listening on {host}:{port}")
    async with server:
        await server.serve_forever()
