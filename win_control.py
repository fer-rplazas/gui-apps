import pywinauto
import pywinauto.mouse as mouse
from pywinauto.timings import Timings

import pyautogui


Timings.fast()
Timings.after_setcursorpos_wait = 0
Timings.after_click_wait = 0

path = "C:\\UsersData\\Johan_v08\\OptTestingTool.exe"
app = pywinauto.Application(backend="uia").connect(
    path=path
)  # .connect(handle='OPT Production Tool v1.0.0.8')

btn_r = app.window().Pane64
btn_r_coords = btn_r.rectangle().mid_point()

btn_l = app.window().Pane84
btn_l_coords = btn_l.rectangle().mid_point()


def get_coords():
    global btn_r_coords, btn_l_coords
    btn_l_coords = btn_l.rectangle().mid_point()
    btn_r_coords = btn_r.rectangle().mid_point()


def clicks():
    mouse.click(coords=btn_l_coords)
    mouse.click(coords=btn_r_coords)


def clicks_alt():
    coords = pyautogui.position()
    pyautogui.click(x=btn_l_coords.x, y=btn_l_coords.y)
    pyautogui.click(x=btn_r_coords.x, y=btn_r_coords.y)
    pyautogui.moveTo(coords)


# clicks()
# clicks_alt()
