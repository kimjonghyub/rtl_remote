#from rig_control_handler import RigControlHandler
import asyncio
import logging
import struct
import numpy as np
import Hamlib
import multiprocessing
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")

CMD_RIG_SET_FREQ = 0x20
CMD_RIG_GET_FREQ = 0x21
CMD_RIG_SET_VFO = 0x22
CMD_RIG_SET_MODE = 0x23
#SERVER_IP = "192.168.10.216"
RIG_PORT = 5622

class RigControlHandler(asyncio.Protocol):
    def __init__(self):
        self.rig_buffer = asyncio.Queue(maxsize=5)
        self.rig = None
        self.current_freq = None  
        self.clients = []  
        self.frequency_task = None
        self.current_vfo = None
        self.current_mode = None
        
    def initialize_device(self):
        """Initialize the RTL-SDR device."""
        try:
            Hamlib.rig_set_debug(Hamlib.RIG_DEBUG_NONE)
            self.rig = Hamlib.Rig(3021)
            self.rig.set_conf("rig_pathname", "/dev/ttyUSB0") 
            self.rig.open() 
            logging.info("Hamlib rig control initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize RTL-SDR: {e}")
            raise
    async def monitor_frequency(self):
        """Monitor frequency and notify clients."""
        last_freq = None
        last_mode = None
        last_vfo = None
        while True:
            try:
                # Get the current frequency from the rig
                current_freq = self.get_freq(Hamlib.RIG_VFO_CURR)
                current_mode = self.get_mode()
                current_vfo = self.get_vfo()
                if current_freq and 100000 <= current_freq <= 30000000:
                    if current_freq != last_freq:
                        logging.info(f"Frequency changed to {current_freq} Hz")
                        last_freq = current_freq
                        await self.notify_clients(data=current_freq, is_mode=False, is_vfo=False)
                    
                    if current_mode != last_mode:
                        logging.info(f"Mode changed to {current_mode}")
                        last_mode = current_mode
                        await self.notify_clients(data=current_mode, is_mode=True, is_vfo=False)
                    
                    if current_vfo != last_vfo:
                        logging.info(f"VFO changed to {current_vfo}")
                        last_vfo = current_vfo
                        await self.notify_clients(data=current_vfo, is_mode=False, is_vfo=True)
                    
                else:
                    logging.warning(f"Invalid frequency detected: {current_freq}")
            except Exception as e:
                logging.error(f"Error while monitoring frequency: {e}")
            await asyncio.sleep(0.1)

    async def notify_clients(self, data, is_mode=False, is_vfo=False):
        """Send the current frequency to all connected clients."""
        disconnected_clients = []
        for client in self.clients:
            try:
                if is_mode:
                    client.transport.write(struct.pack(">I", data))
                    logging.info(f"Notified client with mode: {data}")
                elif is_vfo:
                    client.transport.write(struct.pack(">I", data))
                    logging.info(f"Notified client with vfo: {data}")
                else:
                    client.transport.write(struct.pack(">I", data))
                    logging.info(f"Notified client with frequency: {data} Hz")
            except Exception as e:
                logging.error(f"Failed to notify client: {e}")
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.remove(client)
            logging.info("Removed disconnected client.")

    def start_monitoring(self):
        """Start frequency monitoring task."""
        if self.frequency_task is None or self.frequency_task.done():
            self.frequency_task = asyncio.create_task(self.monitor_frequency())
            logging.info("Started monitoring frequency.")

    async def stop_monitoring(self):
        """Stop frequency monitoring task."""
        if self.frequency_task is not None and not self.frequency_task.done():
            self.frequency_task.cancel()
            try:
                await self.frequency_task
            except asyncio.CancelledError:
                logging.info("Frequency monitoring task cancelled.")
            finally:
                self.frequency_task = None

    def close(self):
        """Close the RIG device."""
        if self.rig:
            self.rig.close()
            logging.info("RIG device closed")

    def set_freq(self, vfo, freq):
        """Set the frequency of the rig."""
        try:
            vfo = int(vfo)  # Ensure VFO is an integer
            freq = int(freq)  # Ensure frequency is an integer
            self.rig.set_freq(vfo, freq)
            logging.info(f"Set rig frequency to {freq} Hz")
        except Exception as e:
            logging.error(f"Failed to set rig frequency: {e}")
            raise
    
    def get_freq(self, vfo):
       
        try:
            vfo = int(vfo)  # Ensure VFO is an integer
            freq = self.rig.get_freq(vfo)
            freq = int(freq)  # Convert frequency to integer
            #logging.info(f"Got rig frequency: {freq} Hz")
            return freq
        except Exception as e:
            logging.error(f"Failed to get rig frequency: {e}")
            raise

    def get_vfo(self, vfo_name=None):
        """Set or infer the current VFO and return its state."""
        try:
            # If a specific VFO name is provided, set it explicitly
            if vfo_name is not None:
                if vfo_name == 0 and self.current_vfo != 11:
                    self.rig.set_vfo(Hamlib.RIG_VFO_A)
                    self.current_vfo = 11  # VFO A
                    logging.info("VFO explicitly set to A")
                elif vfo_name == 1 and self.current_vfo != 12:
                    self.rig.set_vfo(Hamlib.RIG_VFO_B)
                    self.current_vfo = 12  # VFO B
                    logging.info("VFO explicitly set to B")
                return self.current_vfo
            """
            # If no VFO name is provided, infer based on the current frequency
            freq_a = self.rig.get_freq(Hamlib.RIG_VFO_A)
            freq_b = self.rig.get_freq(Hamlib.RIG_VFO_B)
            current_freq = self.rig.get_freq(Hamlib.RIG_VFO_CURR)

            if current_freq == freq_a and self.current_vfo != 11:
                self.current_vfo = 11  # VFO A
                logging.info("Current VFO inferred as A")
            elif current_freq == freq_b and self.current_vfo != 12:
                self.current_vfo = 12  # VFO B
                logging.info("Current VFO inferred as B")
            else:
                logging.warning("Unable to infer VFO from frequency comparison")

            return self.current_vfo
            """
        except Exception as e:
            logging.error(f"Failed to get or set VFO: {e}")
            raise


    def get_mode(self):
        """Get the current mode of the rig."""
        mode_mapping = {
            "USB": 1,  # Upper Side Band
            "LSB": 2,  # Lower Side Band
            "AM": 3,   # Amplitude Modulation
            "FM": 4,   # Frequency Modulation
            "CW": 5,   # Continuous Wave
            "RTTY": 6, # Radioteletype
            "WFM": 7,  # Wideband FM
        }
        try:
            mode, width = self.rig.get_mode(Hamlib.RIG_VFO_CURR)
            mode_name = Hamlib.rig_strrmode(mode)  
            mode_number = mode_mapping.get(mode_name, 0)
            return mode_number
        except Exception as e:
            logging.error(f"Failed to get mode: {e}")
            return 0

    def set_mode(self, mode):
        try:
            self.rig.set_mode(mode)
            
        except Exception as e:
            logging.error(f"Failed to set mode: {e}")
            raise
           
class Rig_ClientHandler(asyncio.Protocol):
    """Handles client connections and forwards commands to the RTL-SDR device."""
    VALID_COMMANDS = [CMD_RIG_SET_FREQ, CMD_RIG_GET_FREQ]
    def __init__(self, rig_handler):
        self.rig_handler = rig_handler
        self.transport = None
        self.buffer = b""

    def connection_made(self, transport):
        self.transport = transport
        self.rig_handler.clients.append(self)
        peername = transport.get_extra_info("peername")
        logging.info(f"Rig Client connected: {peername}")
        self.rig_handler.start_monitoring()
        #logging.info("Connection started monitoring frequency.")

        
    def data_received(self, data):
        self.buffer += data
        asyncio.create_task(self.process_buffer())

    async def process_buffer(self):
        async with asyncio.Lock(): 

            while len(self.buffer) >= 3:  
                if self.buffer[:3] == b"END":
                    logging.info("Rig Client END Received")
                    self.transport.close()
                    return  
                if len(self.buffer) >= 5:
                    cmd_id = self.buffer[0]
                    param = struct.unpack(">I", self.buffer[1:5])[0]
                    await self.handle_rig_command(cmd_id, param)
                    self.buffer = self.buffer[5:]  
                else:
                    break 

    def connection_lost(self, exc):
        #logging.info("Rig Client connection closed")
        if self in self.rig_handler.clients:
            self.rig_handler.clients.remove(self)
            logging.info("Rig Client connection closed")

        if not self.rig_handler.clients:
            asyncio.create_task(self.rig_handler.stop_monitoring())
            #logging.info("Connection stop monitoring frequency.")

        if self.transport:
            self.transport.close()
        if exc:
            logging.warning(f"Rig Connection lost with error: {exc}")
        else:
            logging.info("Rig Connection closed cleanly")
    
    async def handle_rig_command(self, cmd_id, param):
        """Handle rig commands from the client."""
        try:
            if cmd_id == CMD_RIG_GET_FREQ:
                freq = await asyncio.to_thread(self.rig_handler.get_freq, Hamlib.RIG_VFO_CURR)
                logging.info(f"Sending current frequency: {freq} Hz")
                self.transport.write(struct.pack(">I", freq))

            elif cmd_id == CMD_RIG_SET_FREQ:
                self.rig_handler.set_freq(Hamlib.RIG_VFO_CURR, int(param))
                logging.info(f"Set rig frequency to {param} Hz")
                self.transport.write(struct.pack(">I", 0))  # Success

            elif cmd_id == CMD_RIG_SET_VFO:
                vfo = self.rig_handler.get_vfo(param)
                if vfo is not None:
                    freq = await asyncio.to_thread(self.rig_handler.get_freq, vfo)
                    logging.info(f"Switched to VFO {param}, Frequency: {freq} Hz")
                    #self.transport.write(struct.pack(">I", freq))
                else:
                    logging.warning(f"Invalid VFO parameter: {param}")
                    self.transport.write(struct.pack(">I", 1))  # Error

            elif cmd_id == CMD_RIG_SET_MODE:
                                
                if param == 1:
                    self.rig_handler.set_mode(Hamlib.RIG_MODE_USB)
                    logging.info(f"Switched to MODE {param}, Mode: USB")
                elif param == 2:
                    self.rig_handler.set_mode(Hamlib.RIG_MODE_LSB)
                    logging.info(f"Switched to MODE {param}, Mode: LSB")
                elif param == 3:
                    self.rig_handler.set_mode(Hamlib.RIG_MODE_AM)
                    logging.info(f"Switched to MODE {param}, Mode: AM")
                elif param == 4:
                    self.rig_handler.set_mode(Hamlib.RIG_MODE_FM)
                    logging.info(f"Switched to MODE {param}, Mode: FM")
                elif param == 5:
                    self.rig_handler.set_mode(Hamlib.RIG_MODE_CW)
                    logging.info(f"Switched to MODE {param}, Mode: CW")

            else:
                logging.warning(f"Unknown command ID: {cmd_id}")
                self.transport.write(struct.pack(">I", 1))  # Unknown command

        except Exception as e:
            logging.error(f"Error handling command {cmd_id}: {e}")
            self.transport.write(struct.pack(">I", 2))  # Error

def get_ip_from_command():
    
    try:
        
        ip = os.popen("hostname -I").read().strip()
        return ip
    except Exception as e:
        return f"Error retrieving IP: {e}"
SERVER_IP = get_ip_from_command()

async def rig_start_server(rig_handler, host, port):
    """Start the RIG server."""
    loop = asyncio.get_running_loop()
    server = await loop.create_server(lambda: Rig_ClientHandler(rig_handler), host, port)
    logging.info(f"RIG Server listening on {host}:{port}")
    async with server:
        await server.serve_forever()
