import asyncio
import struct
import logging

SERVER_IP = "222.117.38.98"
RIG_PORT = 5622

CMD_RIG_SET_FREQ = 0x20
CMD_RIG_GET_FREQ = 0x21

class RigClient:
    def __init__(self, server_ip, port):
        self.server_ip = server_ip
        self.port = port
        self.reader = None
        self.writer = None
        self.rig_command_queue = asyncio.Queue(maxsize=1)
        self.rig_response_queue = asyncio.Queue(maxsize=1)
        self.connected = False
        self.frequency = 0

    async def connect(self):
        try:
            self.reader, self.writer = await asyncio.open_connection(self.server_ip, self.port)
            self.connected = True
            print(f"Connected to RIG server.")
            await asyncio.gather(
                self.rig_reader_task(),
                self.rig_command_task(),
            )
        except Exception as e:
            print(f"Failed to connect to RIG server: {e}")
            self.connected = False

    async def rig_reader_task(self):
        try:
            while self.connected:
                data = await self.reader.readexactly(4)
                if not self.rig_command_queue.empty():
                    await self.rig_response_queue.put(data)
                else:
                    self.frequency = struct.unpack(">I", data)[0]
                    print(f"Frequency notification received: {self.frequency} Hz")
        except asyncio.CancelledError:
            logging.info("RIG reader task cancelled.")
        except Exception as e:
            logging.error(f"Error in RIG reader task: {e}")

    async def rig_command_task(self):
        try:
            while self.connected:
                if not self.rig_command_queue.empty():
                    command, response_event = await self.rig_command_queue.get()
                    self.writer.write(command)
                    await self.writer.drain()
                    print(f"Command sent: {command.hex()}")
                    try:
                        response = await asyncio.wait_for(self.rig_response_queue.get(), timeout=3.0)
                        self.frequency = struct.unpack(">I", response)[0]
                        print(f"Command response: {self.frequency} Hz")
                        response_event.set()
                    except asyncio.TimeoutError:
                        print(f"No response received from RIG server.")
                        response_event.set()
        except asyncio.CancelledError:
            print(f"RIG command task cancelled.")
        finally:
            print(f"RIG command task stopped.")
            
    async def send_command(self, cmd_id, param):
        if not self.connected:
            print(f"RIG server is not connected.")
            return None
        command = struct.pack(">BI", cmd_id, param)
        response_event = asyncio.Event()
        await self.rig_command_queue.put((command, response_event))
        await response_event.wait()
        if not self.rig_response_queue.empty():
            response = await self.rig_response_queue.get()
            return response
        print(f"No response received from RIG server.")
        return None
        
    async def get_rig_frequency(self):
        """Send CMD_RIG_GET_FREQ and wait for response."""
        cmd_id = CMD_RIG_GET_FREQ
        param = 0
        command = struct.pack(">BI", cmd_id, param)
        response_event = asyncio.Event()

        await self.rig_command_queue.put((command, response_event))
        print(f"Sending command to RIG server: cmd_id={cmd_id}, param={param}")

        try:
            await asyncio.wait_for(response_event.wait(), timeout=3.0)
            print(f"Frequency received: {self.frequency} Hz")
            return self.frequency
        except asyncio.TimeoutError:
            print(f"Failed to get frequency: No response from server.")
            return None

    async def disconnect(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.connected = False
        print(f"Disconnected from RIG server.")

    async def shutdown(self):
        print(f"Shutting down RIG client...")
        tasks = asyncio.all_tasks()
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print(f"Task {task.get_name()} cancelled.")

def run_rig_client():
    rig_client = RigClient(SERVER_IP, RIG_PORT)
    asyncio.run(rig_client.connect())
"""
async def main():
    rig_client = RigClient(SERVER_IP, RIG_PORT)
    try:
        await rig_client.connect()
    except asyncio.CancelledError:
        logging.info("Main task cancelled.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        await rig_client.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("RIG client stopped manually.")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
"""
            

