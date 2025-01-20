#from rig_control_handler import RigControlHandler
import asyncio
import logging
import struct
import numpy as np
import Hamlib
import multiprocessing

logging.basicConfig(level=logging.INFO, format="%(message)s")

CMD_RIG_SET_FREQ = 0x20
CMD_RIG_GET_FREQ = 0x21
SERVER_IP = "192.168.10.216"
RIG_PORT = 5622

class RigControlHandler(asyncio.Protocol):
    def __init__(self):
        self.rig_buffer = asyncio.Queue(maxsize=5)
        self.rig = None
        self.current_freq = None  
        self.clients = []  
        self.frequency_task = None
        
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
        while True:
            try:
                # Get the current frequency from the rig
                current_freq = self.get_freq(Hamlib.RIG_VFO_CURR)
                if current_freq != last_freq:
                    logging.info(f"Frequency changed to {current_freq} Hz")
                    last_freq = current_freq

                    # Notify all connected clients
                    await self.notify_clients(current_freq)
            except Exception as e:
                logging.error(f"Error while monitoring frequency: {e}")
            await asyncio.sleep(0.1)  # Polling interval

    async def notify_clients(self, freq):
        """Send the current frequency to all connected clients."""
        disconnected_clients = []
        for client in self.clients:
            try:
                client.transport.write(struct.pack(">I", freq))
                logging.info(f"Notified client with frequency: {freq} Hz")
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
        """Get the current frequency of the rig."""
        try:
            vfo = int(vfo)  # Ensure VFO is an integer
            freq = self.rig.get_freq(vfo)
            freq = int(freq)  # Convert frequency to integer
            #logging.info(f"Got rig frequency: {freq} Hz")
            return freq
        except Exception as e:
            logging.error(f"Failed to get rig frequency: {e}")
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
            """
            while len(self.buffer) >= 5:
                cmd_id = self.buffer[0]
                param = struct.unpack(">I", self.buffer[1:5])[0]
                await self.handle_rig_command(cmd_id, param)
                self.buffer = self.buffer[5:]
            """
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

            else:
                logging.warning(f"Unknown command ID: {cmd_id}")
                self.transport.write(struct.pack(">I", 1))  # Unknown command

        except Exception as e:
            logging.error(f"Error handling command {cmd_id}: {e}")
            self.transport.write(struct.pack(">I", 2))  # Error

async def rig_start_server(rig_handler, host, port):
    """Start the RIG server."""
    loop = asyncio.get_running_loop()
    server = await loop.create_server(lambda: Rig_ClientHandler(rig_handler), host, port)
    logging.info(f"RIG Server listening on {host}:{port}")
    async with server:
        await server.serve_forever()
