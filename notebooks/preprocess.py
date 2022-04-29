import ndjson
import numpy as np
from rdp import rdp

"""
Expects a word (string) and json file in the following format:

{
  "lines": [
    {
      "points": [
        {
          "x": 214.39999389648438,
          "y": 216
        },
        ...
        {
          "x": 211.39999389648438,
          "y": 213
        }
      ],
      "brushColor": "#000",
      "brushRadius": 2.5
    },
    ...
    {
      "points": [
        {
          "x": 183.39999389648438,
          "y": 277
        },
        ...
        {
          "x": 451.3999938964844,
          "y": 231
        }
      ],
      "brushColor": "#000",
      "brushRadius": 2.5
    }
  ],
  "width": 750,
  "height": 500
}

Returns an dictionary with the following format:

{"word":"cat",
 "drawing":[
   [
     [130,113,99,109,76,64,55,48,48,51,59,86,133,154,170,203,214,217,215,208,186,176,162,157,132],
     [72,40,27,79,82,88,100,120,134,152,165,184,189,186,179,152,131,114,100,89,76,0,31,65,70]
   ],
   ...
   [
     [107,106],
     [96,113]
   ]
 ]
}

"""
def simplify_drawings(word, sketch):
    # convert each line to a list of 2 lists: x, y
    drawing = [[[int(point['x']) for point in line['points']], 
            [int(point['y']) for point in line['points']]] 
           for line in sketch['lines']]

    # align the drawing to the top-left corner, to have min value of zero
    # uniformly scale the drawing, to have a maximum value of 255
    x_vals = [lines[0] for lines in drawing]
    x_vals = list(np.concatenate(x_vals).flat)
    y_vals = [lines[1] for lines in drawing]
    y_vals = list(np.concatenate(y_vals).flat)
    x_min, y_min = min(x_vals), min(y_vals)
    x_max, y_max = max(x_vals), max(y_vals)
    drawing = [[list(np.array((lines[0] - x_min) * 255/(x_max - x_min), int)), list(np.array((lines[1] - y_min) * 255/(y_max - y_min), int))] for lines in drawing]

    # resample all strokes with 1 pixel spacing
    drawing = [[lines[0][::2], lines[1][::2]] for lines in drawing]

    # simplify all stroke using Ramer-Douglas-Peucker
    simplified = []
    for stroke in drawing:
        array = [[stroke[0][i], stroke[1][i]] for i in range(len(stroke[0]))]
        simplified_stroke = rdp(array, epsilon = 2.0)
        x_vals = [point[0] for point in simplified_stroke]
        y_vals = [point[1] for point in simplified_stroke]
        simplified.append([x_vals, y_vals])

    dictionary = {"word": word, "drawing": simplified}

    return dictionary