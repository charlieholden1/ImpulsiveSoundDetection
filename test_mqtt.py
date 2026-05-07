import time
from impulsive_sound_detection.mqtt_bridge import MQTTBridge
from impulsive_sound_detection.classifier import ClassificationResult

bridge = MQTTBridge(broker_host="127.0.0.1", node_id="node_test")
bridge.connect()
print("Connected.")

# Fake a detection result
result = ClassificationResult(
    timestamp=time.time(),
    onset_index=0,
    label="Gunshot, gunfire",
    confidence=0.94,
    is_suspicious=True,
)

bridge.publish_detection(result)
result2 = ClassificationResult(
    timestamp=time.time(),
    onset_index=0,
    label="Gunshot, gunfire",
    confidence=0.91,
    is_suspicious=True,
    node_id="node_2",       # different node
)
bridge.publish_detection(result2)
bridge.publish_rms(rms=0.42, baseline=0.06, threshold=0.18, is_trigger=True)
print("Published detection + RMS.")
time.sleep(2)
bridge.disconnect()
print("Done.")