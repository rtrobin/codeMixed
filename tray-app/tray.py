import PySimpleGUI as sg
import pyautogui

def main():
    # menu_def = ['BLANK', ['&Open', '---', '&Save', ['1', '2', ['a', 'b']], '!&Properties', 'E&xit']]
    menu_def = ['BLANK', ['!self use', '---', 'E&xit']]

    # tray = sg.SystemTray(menu=menu_def, filename=r'icon.png')
    tray = sg.SystemTray(menu=menu_def, data_base64=sg.DEFAULT_BASE64_ICON)

    while True:  # The event loop
        menu_item = tray.read(timeout=1000*60)

        if menu_item == 'Exit':
            break
        elif menu_item == sg.EVENT_SYSTEM_TRAY_ICON_DOUBLE_CLICKED:
            break

        pyautogui.press('ctrl')

    tray.close()

if __name__ == '__main__':
    main()
