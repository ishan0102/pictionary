from tkinter import *
import json
import tkinter

# create application
app = Tk()
app.geometry("400x400")

# add a canvas to the app
canvas = Canvas(app, bg='white')
canvas.pack(anchor='nw', fill='both', expand=1)

# get x and y coordinates of mouse click
def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y


# draw a line based on the clicks
def draw_smth(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y), 
                      fill='black', 
                      width=4)
    lasx, lasy = event.x, event.y


# call above functions on mouse clicks
canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)

# trying to save and load to/from json: below code is from 
# https://stackoverflow.com/questions/63025797/how-to-python-tkinter-saving-canvas-object-by-dump-all-canvas-object 
json_file = tkinter.StringVar()
var = tkinter.IntVar()

def json_save():
    # enter file name
    file_name_pop_up()

    with open(json_file.get() + ".json", 'w') as f:
        for item in canvas.find_all():
            print(json.dumps({
                'type': canvas.type(item),
                'coords': canvas.coords(item),
                'options': {key:val[-1] for key,val in canvas.itemconfig(item).items()}
            }), file=f)


def json_load():
    # if you want to clear the canvas first, uncomment below line
    canvas.delete('all')
    funcs = {
        'arc': canvas.create_arc,
        #'bitmap' and 'image' are not supported
        #'bitmap': self.c.create_bitmap,
        #'image': self.c.create_image,
        'line': canvas.create_line,
        'oval': canvas.create_oval,
        'polygon': canvas.create_polygon,
        'rectangle': canvas.create_rectangle,
        'text': canvas.create_text,
         # 'window' is not supported
    }

    # enter file name
    file_name_pop_up()

    with open(json_file.get() + ".json") as f:
        for line in f:
            item = json.loads(line)
            if item['type'] in funcs:
                funcs[item['type']](item['coords'], **item['options'])


# this function was adapted from https://www.tutorialspoint.com/creating-a-popup-message-box-with-an-entry-field-in-tkinter 
def file_name_pop_up():
    # Create a Toplevel window
    top = Toplevel(app)
    top.geometry("300x100")

    label = tkinter.Label(top, text = 'Enter File Name', font=('calibre',10, 'bold'))
    label.pack()

    # Create an Entry Widget in the Toplevel window
    entry = Entry(top, textvariable = json_file, width= 25)
    entry.pack()

    # Create a Button Widget in the Toplevel Window
    button= Button(top, text="Ok", command = lambda:set_file_name(entry, top))
    button.pack(pady=5, side= TOP)

    button.wait_variable(var)
    return

def set_file_name(entry, top):
    var.set(1)
    top.destroy()


menubar = Menu(app)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Save", command=json_save)
filemenu.add_command(label="Open", command=json_load)
menubar.add_cascade(label="File", menu=filemenu)

# stay in this loop indefinitely
app.config(menu=menubar)
app.mainloop()