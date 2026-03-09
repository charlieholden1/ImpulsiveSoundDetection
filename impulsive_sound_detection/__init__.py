"""
Robust Impulsive Sound Detection System
========================================

A modular production-ready system for monitoring continuous audio streams
and identifying suspicious impulsive sounds (bangs, glass breaking, gunshots)
while ignoring routine school noise.

Modules
-------
config         – Global constants, paths, and tunable parameters.
data_loader    – Data discovery, loading, and segment extraction.
augmentor      – Audio augmentation pipeline (audiomentations).
stream_monitor – Real-time energy-based trigger (Stage 1).
classifier     – YAMNet-based classification and filtering (Stage 2).
visualizer     – Matplotlib waveform + onset visualisation.
pipeline       – End-to-end orchestration of Stages 1 & 2.
dashboard      – ANSI-coloured terminal UI for live presentations.
live_stream    – Live microphone capture via sounddevice.
gui            – customtkinter graphical dashboard with live plots.
"""

__version__ = "1.2.0"
