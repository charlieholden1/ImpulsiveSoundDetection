"""
simulate_live.py – Feed a WAV file through the live pipeline in real time.

Simulates what the RPi nodes will do with a real microphone, but reads
audio from a WAV file instead of sounddevice.  Feeds audio at the real
sample rate so Stage 1 RMS trigger and Stage 2 YAMNet run exactly as
they would live.

Usage
-----
    python simulate_live.py                        # uses test_spike.wav
    python simulate_live.py --wav path/to/file.wav
    python simulate_live.py --wav myfile.wav --loop # loop forever
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np

# ── Optional MQTT ─────────────────────────────────────────────────────
def get_bridge(use_mqtt: bool, broker: str, node_id: str):
    if not use_mqtt:
        return None
    from impulsive_sound_detection.mqtt_bridge import MQTTBridge
    bridge = MQTTBridge(broker_host=broker, node_id=node_id)
    bridge.connect()
    print(f"MQTT connected to {broker} as {node_id}")
    return bridge


def main():
    parser = argparse.ArgumentParser(
        description="Simulate live mic stream from a WAV file."
    )
    parser.add_argument(
        "--wav",
        default="test_spike.wav",
        help="Path to WAV file (default: test_spike.wav)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the WAV file continuously (Ctrl+C to stop)",
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=2.0,
        help="Stage 1 energy threshold multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,          # change from 1024 to 512
        help="Samples per feed call (default: 512)",
    )
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Publish detections to MQTT broker",
    )
    parser.add_argument(
        "--broker-host",
        default="127.0.0.1",
        help="MQTT broker host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--node-id",
        default="node_sim",
        help="Node ID for MQTT (default: node_sim)",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"ERROR: WAV file not found: {wav_path}")
        print("Create one with:  python test_spike.py")
        sys.exit(1)

    # ── Load audio ────────────────────────────────────────────────────
    from impulsive_sound_detection.data_loader import load_wav
    from impulsive_sound_detection.stream_monitor import StreamMonitor
    from impulsive_sound_detection.pipeline import DetectionPipeline
    from impulsive_sound_detection.dashboard import LiveDashboard
    from impulsive_sound_detection import config

    waveform, sr = load_wav(wav_path)
    duration = len(waveform) / sr
    print(f"\nLoaded: {wav_path.name}  ({duration:.2f}s, {sr}Hz, {len(waveform)} samples)")

    # ── Build pipeline ────────────────────────────────────────────────
    bridge = get_bridge(args.mqtt, args.broker_host, args.node_id)

    monitor = StreamMonitor(
        sample_rate=sr,
        energy_multiplier=args.threshold_multiplier,
        node_id=args.node_id,          # already correct
    )
    pipeline = DetectionPipeline(
        monitor=monitor,
        mqtt_bridge=bridge,
        classifier_mode="yamnet",
    )
    # The classifier also needs node_id:
    pipeline.classifier._node_id = args.node_id   # add this line
    dashboard = LiveDashboard(enable_colour=True)
    dashboard.print_banner()

    print(f"  Loading YAMNet model … ", end="", flush=True)
    pipeline.classifier._ensure_model()
    print("done.\n")

    pipeline.start_inference_worker(dashboard=dashboard)

    # ── Stream loop ───────────────────────────────────────────────────
    block_size = args.block_size
    samples_per_sec = sr
    run = True

    print(f"  Streaming {wav_path.name} at real-time speed  "
          f"(threshold ×{args.threshold_multiplier})  Ctrl+C to stop\n")

    try:
        first_loop = True
        while run:
            if first_loop:
                monitor.reset()   # clear state only on the very first pass
                first_loop = False
            offset = 0
            total  = len(waveform)
            start  = time.monotonic()

            while offset < total:
                end   = min(offset + block_size, total)
                chunk = waveform[offset:end]

                # Feed Stage 1 (with optional MQTT RMS publishing)
                monitor.feed(chunk, mqtt_bridge=bridge)

                # Update dashboard meter
                if monitor._rms_history:
                    rms      = monitor._rms_history[-1]
                    baseline = monitor._rolling_mean()
                    threshold = baseline * monitor.energy_multiplier
                    dashboard.update_meter(rms, baseline, threshold)

                offset += block_size

                # Throttle to real-time speed
                elapsed   = time.monotonic() - start
                expected  = offset / sr
                sleep_for = expected - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            if not args.loop:
                run = False
            else:
                print("\n  [loop] Restarting …")
                time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop_inference_worker()
        if bridge:
            bridge.disconnect()
        elapsed = time.monotonic() - start
        dashboard.print_shutdown(elapsed)

    # Print summary
    results = pipeline.results
    print(f"\n{'='*55}")
    print(f"  Total detections : {len(results)}")
    print(f"  Suspicious       : {sum(1 for r in results if r.is_suspicious)}")
    print(f"  Non-suspicious   : {sum(1 for r in results if not r.is_suspicious)}")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r.to_json()}")


if __name__ == "__main__":
    main()