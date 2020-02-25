from tkinter import *

#set window
canvas_width = 800
canvas_height = 600

master = Tk()

#clear canvas and print direction
def up():
    w.delete("all")
    w.create_text(canvas_width / 2,
              canvas_height / 2,
              font=("Purisa", 42),
              text="FORWARD")
              
def down():
    w.delete("all")
    w.create_text(canvas_width / 2,
              canvas_height / 2,
              font=("Purisa", 42),
              text="BACKWARD")
              
def left():
    w.delete("all")
    w.create_text(canvas_width / 2,
              canvas_height / 2,
              font=("Purisa", 42),
              text="LEFT")
              
def right():
    w.delete("all")
    w.create_text(canvas_width / 2,
              canvas_height / 2,
              font=("Purisa", 42),
              text="RIGHT")
              


#keyboard handler
def key(event):
    if event.char == 'w':
        up()
    elif event.char == 's':
        down()
    elif event.char == 'a':
        left()
    elif event.char == 'd':
        right()

#create window        
w = Canvas(master,
           width=canvas_width, 
           height=canvas_height)
w.pack()

#initialize window text
w.create_text(canvas_width / 2,
              canvas_height / 2,
              font=("Purisa", 42),
              text="Which direction?")

#listen to keyboard              
w.focus_set()
w.bind("<Key>", key)

mainloop()