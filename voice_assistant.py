"""
YogaCare AI - Phase 3: Voice Assistant Module
Provides asynchronous, non-blocking text-to-speech audio feedback, intelligent speech
rate-limiting, priority safety alerts, and milestone announcements for yoga practitioners.
"""

import logging
import queue
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger("YogaCareAI.VoiceAssistant")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VoiceAssistant:
    """
    Non-blocking, background-threaded audio coaching assistant for YogaCare AI.

    Features:
    - Runs in a detached worker thread so webcam FPS is never degraded while speaking.
    - Rate-limits speech (e.g. minimum 3.5s between cues) to avoid auditory fatigue.
    - Prevents repeating the exact same cue repeatedly.
    - High-priority interrupts for critical safety warnings.
    - Windows native sound effects for milestone achievements.
    """

    def __init__(
        self,
        min_interval_seconds: float = 3.5,
        repeat_cooldown_seconds: float = 8.0,
        voice_rate: int = 165,
        volume: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            min_interval_seconds: Minimum seconds between any two spoken cues.
            repeat_cooldown_seconds: Minimum seconds before repeating the exact same cue.
            voice_rate: Speech speed (words per minute, default 165).
            volume: Volume level [0.0, 1.0].
            enabled: Master audio toggle.
        """
        self.min_interval = min_interval_seconds
        self.repeat_cooldown = repeat_cooldown_seconds
        self.voice_rate = voice_rate
        self.volume = volume
        self.enabled = enabled

        self._last_speech_time = 0.0
        self._last_spoken_text = ""
        self._last_spoken_text_time = 0.0

        # Thread-safe queue and worker thread
        self._speech_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background daemon loop that initializes TTS engine and consumes speech queue."""
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.ole32.CoInitialize(None)
            except Exception:
                pass

        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.voice_rate)
            engine.setProperty("volume", self.volume)
            # Pick a natural voice if available
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
        except Exception as e:
            logger.warning("Failed to initialize pyttsx3 engine: %s. Audio feedback disabled.", e)
            if sys.platform.startswith("win"):
                try:
                    import ctypes
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass
            return

        while not self._stop_event.is_set():
            try:
                # Wait for next text item with timeout
                item = self._speech_queue.get(timeout=0.2)
                if item is None:
                    break

                text, is_priority = item
                if self.enabled and engine:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        logger.debug("TTS playback error: %s", err)
                        try:
                            engine = pyttsx3.init()
                            engine.setProperty("rate", self.voice_rate)
                            engine.setProperty("volume", self.volume)
                        except Exception:
                            pass

                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug("Unexpected error in speech worker: %s", e)

        # Cleanup
        if engine:
            try:
                engine.stop()
            except Exception:
                pass

        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.ole32.CoUninitialize()
            except Exception:
                pass

    def speak(self, text: str, priority: bool = False) -> bool:
        """
        Queues a coaching message to be spoken aloud.

        Args:
            text: Coaching instruction (e.g. "Bend your front knee deeper").
            priority: If True, bypasses standard interval throttling (for safety warnings or guided introductions).

        Returns:
            True if queued, False if throttled or disabled.
        """
        if not self.enabled or not text or not text.strip():
            return False

        clean_text = text.strip().lstrip(">").strip()
        now = time.time()

        # Check safety/priority flags
        is_safety = "safety" in clean_text.lower() or priority

        if not is_safety:
            # Check interval throttling
            if now - self._last_speech_time < self.min_interval:
                return False

            # Check duplicate phrase throttling
            if clean_text == self._last_spoken_text and (now - self._last_spoken_text_time < self.repeat_cooldown):
                return False

        # Clear queue on priority safety/guidance alerts so it is spoken immediately
        if is_safety and not self._speech_queue.empty():
            try:
                while not self._speech_queue.empty():
                    self._speech_queue.get_nowait()
                    self._speech_queue.task_done()
            except Exception:
                pass

        self._last_speech_time = now
        self._last_spoken_text = clean_text
        self._last_spoken_text_time = now

        self._speech_queue.put((clean_text, is_safety))
        return True

    def speak_guidance(self, text: str) -> bool:
        """Queues a guided instruction or breathing cue with priority so practitioner hears it immediately."""
        return self.speak(text, priority=True)

    def play_milestone_chime(self) -> None:
        """Plays an uplifting audio tone when target hold duration is achieved."""
        if not self.enabled or not sys.platform.startswith("win"):
            return

        def _play():
            try:
                import winsound
                # Upward arpeggio chime (C5 -> E5 -> G5)
                winsound.Beep(523, 120)
                time.sleep(0.04)
                winsound.Beep(659, 120)
                time.sleep(0.04)
                winsound.Beep(784, 250)
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()

    def stop(self) -> None:
        """Signals worker thread to terminate and cleans up."""
        self._stop_event.set()
        self._speech_queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        logger.info("Voice Assistant shut down cleanly.")
