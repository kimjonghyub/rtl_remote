import multiprocessing
import subprocess
from iq_client import run_iq_client
from rig_client import run_rig_client

def run_audio_client_as_subprocess():
    subprocess.Popen(["python3", "audio_client.py"])

def main():
    
    iq_process = multiprocessing.Process(target=run_iq_client)
    rig_process = multiprocessing.Process(target=run_rig_client)

   
    iq_process.start()
    rig_process.start()

   
    run_audio_client_as_subprocess()

   
    iq_process.join()
    rig_process.join()

if __name__ == "__main__":
    main()
