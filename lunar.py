from math import sqrt

from prompt_toolkit import prompt, HTML
from prompt_toolkit import Application
from prompt_toolkit.application import get_app

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.containers import VSplit, HSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout

def make_dashed_title(text):
    """Return a callable that centers `text` within the current
    terminal width, padding with '-' instead of spaces."""
    def _get_text():
        width = get_app().output.get_size().columns
        if len(text) + 2 >= width:  # not enough room to pad nicely
            return text
        total_pad = width - len(text) - 2  # -2 for a space on each side of text
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        return f"{'-' * left_pad} {text} {'-' * right_pad}"
    return _get_text

class EmptyResError( Exception ):
    def __init__(self, msg):
        self.msg = msg
        super().__init__( self.msg )

    def __str__(self):
        return f'{self.msg}'        

class LunarCrashError( Exception ):
    def __init__(self, msg):
        self.msg = msg
        super().__init__( self.msg )

    def __str__(self):
        return f'{self.msg}'        

class LunarLandingException( Exception ):
    def __init__(self, msg):
        self.msg = msg
        super().__init__( self.msg )

    def __str__(self):
        return f'{self.msg}'        
    
class LunarModel:
    def __init__(self, name, altitude, speed):
        self.name      = name
        self.altitude  = altitude
        self.speed     = speed
        self.fuel_res  = 120
        self.fuel_warn = self.fuel_res // 3
        self.fuel_alrt = self.fuel_res // 6
        self.fuel_acc  = 5
        self.tolerance = 3
        self.adjust    = 3
        self.status    = 'OK'

    def burn( self, fuel ):
        self.status    = 'OK'
        if fuel > self.fuel_res:
            self.status = f'Not enough fuel in reserve!'
            raise EmptyResError( self.status )
        #
        gamma = self.fuel_acc - fuel
        self.fuel_res -= fuel
        self.altitude -= self.speed + gamma // 2
        self.speed    += gamma
        #
        if self.altitude <= 0 or ( abs(self.altitude) < self.adjust and abs(self.speed) > self.tolerance ) :
            self.status = f'Crashed at speed {self.speed}'
            raise LunarCrashError( self.status )
        if abs(self.altitude) < self.adjust and abs(self.speed) < self.tolerance :
            self.status = f'Safely Landed. Congratulations!'
            raise LunarLandingException( self.status )
        if self.fuel_res < 2:
            self.status = f'Out of fuel. Will crash at speed { int(sqrt(self.speed*self.speed + self.altitude + self.altitude))}'
            raise LunarCrashError( self.status )

class LunarView:
    def __init__( self, model ):
        self.model   = model
        self.buf     = Buffer( multiline=False )
        self.w_input = Window( height=1, content=BufferControl( buffer=self.buf ) )
        self.root    = HSplit([
            HSplit([
                Window( height=1, 
                        content= FormattedTextControl( text=make_dashed_title(f'Model: {self.model.name}') ) ),
                Window( height=1,
                        content=FormattedTextControl(
                            text= lambda: f'Altitude  : {self.model.altitude}' if self.model.altitude > 100 else HTML(f'<ansiyellow>Altitude  : {self.model.altitude}</ansiyellow>') )),
                Window( height=1, content=FormattedTextControl( text= lambda: f'Speed     : {self.model.speed}' ) ),
                Window( height=1,
                        content=FormattedTextControl(
                            text = lambda: HTML(f'<ansired>Fuel Res. : {self.model.fuel_res}</ansired>') if
                            self.model.fuel_res <= self.model.fuel_alrt else
                            (HTML(f'<ansiyellow>Fuel Res. : {self.model.fuel_res}</ansiyellow>') if
                             self.model.fuel_res <= self.model.fuel_warn else
                             f'Fuel Res. : {self.model.fuel_res}' ) ) ),
                Window( height=1, content=FormattedTextControl( text= lambda: f'Status    : {self.model.status}' ) ),
            ]),
            Window( height=1, char='-' ),
            Window( height=1, content=FormattedTextControl( text=f'Enter fuel mass (or C-q to quit): ' ) ),
            self.w_input
        ])
        self.layout = Layout( self.root, focused_element=self.w_input )
        

m = LunarModel( 'HP25', 500, 50 )
v = LunarView( m )
#    
kb = KeyBindings()

@kb.add('c-q')
def exit_(event):
    """
    Pressing Ctrl-Q will exit the user interface.

    Setting a return value means: quit the event loop that drives the user
    interface and return this value from the `Application.run()` call.
    """
    event.app.exit()

@kb.add("enter")
def _(event):
    text       = v.buf.text.strip()

    if not text:
        return  # ignore empty input, keep asking

    try:
        value = int(text)
    except ValueError:
        return

    try:
        v.model.burn( value )
    except LunarCrashError:
        event.app.exit()
    except EmptyResError:
        pass
    except LunarLandingException:
        event.app.exit()
    
    v.buf.text = ""  # clear the input line for the next entry



app = Application( layout=v.layout, key_bindings=kb, full_screen=False )
app.run()
