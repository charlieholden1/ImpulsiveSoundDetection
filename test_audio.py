"""
test_audio.py – Generate synthetic test WAV files for pipeline testing.

Generates three waveforms:
  spike.wav    – Sharp energy spike (original test_spike.wav behaviour)
  sine.wav     – Gradual sine wave at 440 Hz with a loud burst
  sawtooth.wav – Sawtooth wave with a sharp transient

Each file is 5 seconds at 16 kHz mono float32, matching YAMNet's
expected input format.

Usage
-----
    python test_audio.py              # generates all three
    python test_audio.py --type spike # generates just one
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR        = 16_000   # 16 kHz – matches YAMNet / pipeline requirement
DURATION  = 5        # seconds
SAMPLES   = SR * DURATION
OUT_DIR   = Path(__file__).parent  # repo root


# ── Waveform generators ───────────────────────────────────────────────

def make_spike() -> np.ndarray:
    """5 seconds of near-silence with a hard 0.975s energy spike at t=2s.

    Designed to guarantee Stage 1 RMS trigger fires at the spike.
    """
    wav = np.zeros(SAMPLES, dtype=np.float32)

    # Background noise floor so rolling baseline isn't zero
    rng = np.random.default_rng(42)
    wav += rng.normal(0, 0.002, SAMPLES).astype(np.float32)

    # Hard square-wave spike at t=2s, 0.975s long (one YAMNet window)
    spike_start  = SR * 2
    spike_len    = int(SR * 0.975)
    wav[spike_start : spike_start + spike_len] += 0.85

    return np.clip(wav, -1.0, 1.0)


def make_sine() -> np.ndarray:
    """5 seconds of quiet 440 Hz sine that builds to a loud burst at t=2s.

    Simulates a gradual sound event — Stage 1 should trigger when the
    amplitude crosses the rolling baseline threshold.
    """
    t   = np.linspace(0, DURATION, SAMPLES, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(7)

    # Quiet background noise floor
    wav = rng.normal(0, 0.002, SAMPLES).astype(np.float32)

    # Envelope: ramps up from t=1.5s, peaks at t=2.5s, fades by t=4s
    envelope = np.zeros(SAMPLES, dtype=np.float32)
    ramp_start = int(SR * 1.5)
    peak_start = int(SR * 2.0)
    peak_end   = int(SR * 3.0)
    fade_end   = int(SR * 4.0)

    # Ramp up
    ramp_len = peak_start - ramp_start
    envelope[ramp_start:peak_start] = np.linspace(0, 0.8, ramp_len)
    # Sustain
    envelope[peak_start:peak_end] = 0.8
    # Fade out
    fade_len = fade_end - peak_end
    envelope[peak_end:fade_end] = np.linspace(0.8, 0, fade_len)

    # 440 Hz sine modulated by envelope
    wav += (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)

    return np.clip(wav, -1.0, 1.0)


def make_sawtooth() -> np.ndarray:
    """5 seconds of sawtooth wave with a sharp transient at t=2s.

    Sawtooth waves have strong harmonic content — interesting for YAMNet
    classification.  The sudden onset is designed to trigger Stage 1.
    """
    t   = np.linspace(0, DURATION, SAMPLES, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(13)

    # Background noise floor
    wav = rng.normal(0, 0.002, SAMPLES).astype(np.float32)

    # 220 Hz sawtooth: t mod (1/f) scaled to [-1, 1]
    freq = 220.0
    period = SR / freq
    sawtooth = (2.0 * ((np.arange(SAMPLES) % period) / period) - 1.0).astype(np.float32)

    # Amplitude envelope: silent until t=2s, hard onset, sustain, sharp cutoff
    envelope = np.zeros(SAMPLES, dtype=np.float32)

    onset_start  = int(SR * 2.0)
    onset_end    = int(SR * 2.02)   # 20ms attack
    sustain_end  = int(SR * 3.5)
    release_end  = int(SR * 3.6)   # 100ms release

    # Hard attack
    attack_len = onset_end - onset_start
    envelope[onset_start:onset_end] = np.linspace(0, 0.75, attack_len)
    # Sustain
    envelope[onset_end:sustain_end] = 0.75
    # Short release
    release_len = release_end - sustain_end
    envelope[sustain_end:release_end] = np.linspace(0.75, 0, release_len)

    wav += sawtooth * envelope

    return np.clip(wav, -1.0, 1.0)


# ── Generator registry ────────────────────────────────────────────────

GENERATORS = {
    "spike":    (make_spike,    "spike.wav",    "Hard square-wave energy spike at t=2s"),
    "sine":     (make_sine,     "sine.wav",     "Gradual 440 Hz sine burst, peak at t=2-3s"),
    "sawtooth": (make_sawtooth, "sawtooth.wav", "220 Hz sawtooth with sharp onset at t=2s"),
}


def generate(name: str) -> Path:
    fn, filename, description = GENERATORS[name]
    out_path = OUT_DIR / filename
    wav = fn()
    sf.write(str(out_path), wav, SR, subtype="FLOAT")
    duration_s = len(wav) / SR
    peak       = float(np.abs(wav).max())
    rms_all    = float(np.sqrt(np.mean(wav ** 2)))
    print(f"  [{name:>8}]  {filename:<16}  {description}")
    print(f"             {duration_s:.1f}s  peak={peak:.3f}  rms={rms_all:.5f}  → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test WAV files for ISD pipeline testing."
    )
    parser.add_argument(
        "--type",
        choices=list(GENERATORS.keys()),
        default=None,
        help="Generate only one type (default: generate all).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: repo root next to this script).",
    )
    args = parser.parse_args()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [args.type] if args.type else list(GENERATORS.keys())

    print(f"\nGenerating {len(targets)} test WAV file(s) → {OUT_DIR}\n")
    paths = []
    for name in targets:
        paths.append(generate(name))

    print(f"\nDone. Run with simulate_live.py, e.g.:")
    for p in paths:
        print(f"  python simulate_live.py --wav {p.name} "
              f"--threshold-multiplier 1.5 --mqtt "
              f"--broker-host 127.0.0.1 --node-id node_sim --loop")


if __name__ == "__main__":
    main()