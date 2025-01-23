import asyncio
import logging
import multiprocessing


def start_iq_server():
    from IQ_Server import iq_start_server, RtlTcpHandler, SERVER_IP, IQ_PORT

    async def iq_main():
        rtl_handler = RtlTcpHandler()
        rtl_handler.initialize_device()
        await iq_start_server(rtl_handler, SERVER_IP, IQ_PORT)

    asyncio.run(iq_main())

def start_rig_server():
    from Rig_Server import rig_start_server, RigControlHandler, SERVER_IP, RIG_PORT

    async def rig_main():
        rig_handler = RigControlHandler()
        rig_handler.initialize_device()
        await rig_start_server(rig_handler, SERVER_IP, RIG_PORT)

    asyncio.run(rig_main())

def start_audio_server():
    from Audio_Server_udp import audio_start_server, AudioHandler, SERVER_IP, AUDIO_PORT

    async def audio_main():
        audio_handler = AudioHandler()
        audio_handler.audio_task = asyncio.create_task(audio_handler.start_audio_capture())
        await audio_start_server(audio_handler, SERVER_IP, AUDIO_PORT)

    asyncio.run(audio_main())

if __name__ == "__main__":
    print("Starting servers using multiprocessing...")
    try:
        processes = []
       
        processes.append(multiprocessing.Process(target=start_iq_server, name="IQ Server"))
        processes.append(multiprocessing.Process(target=start_rig_server, name="Rig Server"))
        processes.append(multiprocessing.Process(target=start_audio_server, name="Audio Server"))

        
        for process in processes:
            process.start()

       
        for process in processes:
            process.join()

    except KeyboardInterrupt:
        print("Shutting down servers...")
        for process in processes:
            process.terminate()
        print("All servers stopped.")
