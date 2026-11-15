"""Compact terminal dial: purple load arc, cyan needle, muted remaining scale."""
import math

from rich.align import Align
from rich.text import Text


class LoadGauge:
    def __init__(self, percent, *, height=6, needle_percent=None):
        self.percent = max(0, min(100, percent))
        self.needle = self.percent if needle_percent is None else max(0, min(100, needle_percent))
        self.height = height

    def __rich_console__(self, console, options):
        if options.max_width < 16:
            yield Text(f'GPU {self.percent:.0f}%', style='bold #56d4dd')
            return
        columns = min(25, options.max_width)
        rows = max(2, self.height - 1)
        pixel_width, pixel_height = columns * 2, rows * 4
        center_x, center_y = (pixel_width - 1) / 2, pixel_height - 1
        radius_x, radius_y = center_x - 1, pixel_height - 2
        pixels = {}

        def mark(x, y, kind):
            point = (round(x), round(y))
            if 0 <= point[0] < pixel_width and 0 <= point[1] < pixel_height:
                pixels[point] = max(kind, pixels.get(point, 0))

        for step in range(401):
            fraction = step / 400
            angle = math.pi * (1 - fraction)
            x = center_x + radius_x * math.cos(angle)
            y = center_y - radius_y * math.sin(angle)
            kind = 2 if fraction <= self.needle / 100 else 1
            mark(x, y, kind)
        for fraction in (0, .25, .5, .75, 1):
            angle = math.pi * (1 - fraction)
            for inset in (0, 1, 2):
                mark(center_x + (radius_x - inset) * math.cos(angle),
                     center_y - (radius_y - inset) * math.sin(angle), 1)
        angle = math.pi * (1 - self.needle / 100)
        for step in range(101):
            length = .8 * step / 100
            mark(center_x + radius_x * length * math.cos(angle),
                 center_y - radius_y * length * math.sin(angle), 3)
        for dx in (-1, 0, 1):
            mark(center_x + dx, center_y, 3)

        # Braille provides 2x4 dots per terminal cell without extra dependencies.
        dots = ((0, 0, 1), (0, 1, 2), (0, 2, 4), (1, 0, 8),
                (1, 1, 16), (1, 2, 32), (0, 3, 64), (1, 3, 128))
        styles = {0: '', 1: '#48556b', 2: '#b39df3', 3: 'bold #56d4dd'}
        for row in range(rows):
            line = Text()
            for column in range(columns):
                bits = kind = 0
                for dx, dy, bit in dots:
                    value = pixels.get((column * 2 + dx, row * 4 + dy), 0)
                    if value:
                        bits |= bit
                        kind = max(kind, value)
                line.append(chr(0x2800 + bits) if bits else ' ', style=styles[kind])
            yield Align.center(line)
        label = f'GPU {self.percent:3.0f}%'
        left = (columns - len(label)) // 2
        caption = Text('0' + ' ' * (left - 1), style='#8493a5')
        caption.append(label, 'bold #56d4dd')
        caption.append(' ' * max(0, columns - left - len(label) - 3) + '100', '#8493a5')
        yield Align.center(caption)
