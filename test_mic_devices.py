import sounddevice as sd
import numpy as np
import time

def callback(indata, frames, time_info, status):
    rms = float(np.sqrt(np.mean(indata[:, 0].astype(np.float64) ** 2)))
    bar = int(rms * 500)
    filled = chr(9608) * min(bar, 60)
    print(f'RMS: {rms:.6f}  [{filled}]', end='\r')

input_devices = [1, 5, 9, 11, 16, 22]

for dev_idx in input_devices:
    try:
        info = sd.query_devices(dev_idx)
        if info['max_input_channels'] < 1:
            continue
        name = info['name']
        print(f'\n--- Device {dev_idx}: {name} ---')
        print('Clap or talk now...')
        with sd.InputStream(device=dev_idx, channels=1, dtype='float32',
                            samplerate=16000, blocksize=1024,
                            callback=callback):
            time.sleep(4)
    except Exception as e:
        print(f'  Device {dev_idx} failed: {e}')

print('\n\nDone. Note which device responded to your claps.')