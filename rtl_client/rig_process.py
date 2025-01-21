import asyncio
import struct
from constants import SERVER_IP, RIG_PORT

CMD_RIG_SET_FREQ = 0x20
CMD_RIG_GET_FREQ = 0x21

rig_command_queue = asyncio.Queue(maxsize=1)
rig_response_queue = asyncio.Queue(maxsize=1)
rig_connected = False
is_connected = False
rig_reader = None
rig_writer = None
tasks = []  
      
async def handle_rig_connection():
    global rig_reader, rig_writer, rig_connected
    try:
        rig_reader, rig_writer = await asyncio.open_connection(SERVER_IP, RIG_PORT)
        rig_connected = True
        print("RIG server Connected.")
        await asyncio.sleep(0.5) 
        asyncio.create_task(rig_reader_task())
        asyncio.create_task(rig_command_task())
        asyncio.create_task(rig_notification_task())
           
    except Exception as e:
        print(f"Failed to connect to RIG server: {e}")
        rig_connected = False

async def rig_reader_task():
    """Reads data from the RIG server and routes it appropriately."""
    global rig_reader, rig_connected, rig_frequency
    try:
        while rig_connected:
            try:
                # Read 4 bytes from the server
                data = await rig_reader.readexactly(4)

                # Check if a command is pending
                if not rig_command_queue.empty():
                    # Route to the command response queue
                    await rig_response_queue.put(data)
                    print("Command response received.")
                else:
                    # Treat as a notification
                    freq = struct.unpack(">I", data)[0]
                    rig_frequency = freq
                    print(f"Received frequency notification: {freq} Hz")

            except asyncio.IncompleteReadError as e:
                print(f"Incomplete read error: {e}. Connection may have been closed.")
                rig_connected = False
                break

            except Exception as e:
                print(f"Error in rig_reader_task: {e}")

    except asyncio.CancelledError:
        print("RIG reader task canceled.")
        
async def rig_command_task():
    """Processes commands and waits for responses."""
    global rig_writer, rig_connected, rig_frequency
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
                    print(f"Sent command: {command.hex()}")

                    # Wait for the response
                    try:
                        response = await asyncio.wait_for(rig_response_queue.get(), timeout=3.0)
                        rig_frequency = struct.unpack(">I", response)[0]
                        print(f"Command response: {rig_frequency} Hz")
                    except asyncio.TimeoutError:
                        print("No response received from RIG server.")
                        if response_event:
                            response_event.set()
                        continue

                    # Trigger the response event
                    if response_event:
                        response_event.set()

            except Exception as e:
                print(f"Error in rig_command_task: {e}")
                if response_event:
                    response_event.set()

            await asyncio.sleep(0.05)

    except asyncio.CancelledError:
        print("RIG command task canceled.")
    finally:
        print("RIG command task stopped.")
        
async def rig_notification_task():
    """Processes frequency change notifications."""
    global rig_frequency
    while rig_connected:
        try:
            # Wait for a frequency update
            freq = rig_frequency
            #print(f"Processing notification: {freq} Hz")
            await asyncio.sleep(0.1)  # Simulate notification processing
        except Exception as e:
            print(f"Error in rig_notification_task: {e}")
   
async def send_rig_command(cmd_id, param):
    global rig_connected
    if not rig_connected:
        print("RIG server is not connected.")
        return None

    command = struct.pack(">BI", cmd_id, param)
    response_event = asyncio.Event()
    print(f"Sending command to RIG server: cmd_id={cmd_id}, param={param}")
    
    await rig_command_queue.put((command, response_event))
    await response_event.wait() 
    
    if not rig_response_queue.empty():
        response = await rig_response_queue.get()
        print(f"Received response from RIG server: {response}")
        return response

    print("No response received from RIG server.")
    return None
      
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
     
async def get_rig_frequency():
    """Send CMD_RIG_GET_FREQ and wait for response."""
    global rig_command_queue

    # Prepare command and enqueue it
    cmd_id = CMD_RIG_GET_FREQ
    param = 0
    command = struct.pack(">BI", cmd_id, param)
    response_event = asyncio.Event()

    await rig_command_queue.put((command, response_event))
    print(f"Sending command to RIG server: cmd_id={cmd_id}, param={param}")

    # Wait for the response or timeout
    try:
        await asyncio.wait_for(response_event.wait(), timeout=3.0)
        print(f"Frequency received: {rig_frequency} Hz")
    except asyncio.TimeoutError:
        print("Failed to get frequency: No response from server.")
  
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
      
