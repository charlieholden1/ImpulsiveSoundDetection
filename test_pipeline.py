# test_pipeline.py
from pathlib import Path
from impulsive_sound_detection.mqtt_bridge import MQTTBridge
from impulsive_sound_detection.pipeline import DetectionPipeline

bridge = MQTTBridge(broker_host="127.0.0.1", node_id="node_test")
bridge.connect()

pipeline = DetectionPipeline(
    mqtt_bridge=bridge,
    classifier_mode="yamnet",   # swap to "cnn" once you have the .keras file
)

# Point at any .wav file you have - doesn't have to be a gunshot
wav = Path(r"C:\Github\ImpulsiveSoundDetection\test_spike.wav")
results = pipeline.run_on_file(wav, visualize=False)

print(f"{len(results)} detections found")
for r in results:
    print(r.to_json())

bridge.disconnect()