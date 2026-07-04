"""
Pygame visualizer: renders a pulsing "mood blob" that changes color and motion
based on detected emotion, pitch, and energy.

This module exposes a MoodVisualizer class used by live_mood_ring.py.
It's deliberately simple (a breathing circle) so it's easy to restyle.
"""

import math
import pygame

WIDTH, HEIGHT = 640, 640

# Color per emotion (RGB). Extend/tune these however you like.
EMOTION_COLORS = {
    "neutral": (150, 150, 160),
    "calm": (90, 160, 230),
    "happy": (255, 200, 60),
    "sad": (100, 90, 200),
    "angry": (230, 70, 60),
    "fearful": (140, 80, 160),
    "disgust": (110, 150, 90),
    "surprised": (255, 140, 90),
    "unknown": (120, 120, 120),
}


class MoodVisualizer:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Mood Ring for Your Voice")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 28)

        self.current_color = EMOTION_COLORS["neutral"]
        self.target_color = EMOTION_COLORS["neutral"]
        self.current_label = "neutral"
        self.pulse_phase = 0.0
        self.energy_level = 0.0  # 0-1, controls pulse amplitude
        self.pitch_level = 0.0   # 0-1, controls pulse speed

    def update_state(self, label: str, energy_norm: float, pitch_norm: float):
        self.target_color = EMOTION_COLORS.get(label, EMOTION_COLORS["unknown"])
        self.current_label = label
        self.energy_level = energy_norm
        self.pitch_level = pitch_norm

    def _lerp_color(self, c1, c2, t):
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def render_frame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        # Smoothly transition color toward target
        self.current_color = self._lerp_color(self.current_color, self.target_color, 0.08)

        # Pulse animation: speed driven by pitch, amplitude driven by energy
        speed = 0.05 + self.pitch_level * 0.15
        self.pulse_phase += speed
        base_radius = 120
        amplitude = 40 + self.energy_level * 120
        radius = base_radius + amplitude * (0.5 + 0.5 * math.sin(self.pulse_phase))

        self.screen.fill((15, 15, 20))
        pygame.draw.circle(
            self.screen, self.current_color,
            (self.width // 2, self.height // 2), int(radius)
        )

        label_surface = self.font.render(self.current_label.capitalize(), True, (240, 240, 240))
        label_rect = label_surface.get_rect(center=(self.width // 2, self.height - 60))
        self.screen.blit(label_surface, label_rect)

        pygame.display.flip()
        self.clock.tick(30)
        return True

    def close(self):
        pygame.quit()
