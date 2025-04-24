from gpiozero import Motor
import curses

# Define left and right motors
left_motor = Motor(forward=16, backward=17)
right_motor = Motor(forward=18, backward=13)

# Movement functions
def left():
    left_motor.backward()
    right_motor.forward()

def right():
    left_motor.forward()
    right_motor.backward()

def forward():
    left_motor.forward()
    right_motor.forward()

def reverse():
    left_motor.backward()
    right_motor.backward()

def stop():
    left_motor.stop()
    right_motor.stop()

# Mapping keys to actions
actions = {
    curses.KEY_UP:    forward,
    curses.KEY_DOWN:  reverse,
    curses.KEY_LEFT:  left,
    curses.KEY_RIGHT: right,
}

# Main control loop using curses
def main(window):
    next_key = None
    while True:
        curses.halfdelay(1)
        if next_key is None:
            key = window.getch()
        else:
            key = next_key
            next_key = None
        if key != -1:
            curses.halfdelay(3)
            action = actions.get(key)
            if action is not None:
                action()
            next_key = key
            while next_key == key:
                next_key = window.getch()
            stop()

curses.wrapper(main)
