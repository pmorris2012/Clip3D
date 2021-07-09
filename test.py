import math
from vispy import app, gloo
from vispy.color import Color

class Canvas(app.Canvas):
    def __init__(self, *args, **kwargs):
        app.Canvas.__init__(self, *args, **kwargs)
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.color = 'white'

    def on_key_press(self, event):
        modifiers = [key.name for key in event.modifiers]
        print('Key pressed - text: %r, key: %s, modifiers: %r' % (
            event.text, event.key.name, modifiers))

    def on_key_release(self, event):
        modifiers = [key.name for key in event.modifiers]
        print('Key released - text: %r, key: %s, modifiers: %r' % (
            event.text, event.key.name, modifiers))

    def on_mouse_press(self, event):
        self.print_mouse_event(event, 'Mouse press')

    def on_mouse_release(self, event):
        self.print_mouse_event(event, 'Mouse release')

    def on_mouse_move(self, event):
        self.print_mouse_event(event, 'Mouse move')

    def on_mouse_wheel(self, event):
        self.print_mouse_event(event, 'Mouse wheel')

    def print_mouse_event(self, event, what):
        modifiers = ', '.join([key.name for key in event.modifiers])
        print('%s - pos: %r, button: %s, modifiers: %s, delta: %r' %
              (what, event.pos, event.button, modifiers, event.delta))

    def on_draw(self, event):
        gloo.clear(color=True)

    def on_timer(self, event):
        # Animation speed based on global time.
        t = event.elapsed
        c = Color(self.color).rgb
        # Simple sinusoid wave animation.
        s = abs(0.5 + 0.5 * math.sin(t))
        self.context.set_clear_color((c[0] * s, c[1] * s, c[2] * s, 1))
        self.update()

canvas = Canvas(keys='interactive')
#canvas.measure_fps()
canvas.show()
app.run()
