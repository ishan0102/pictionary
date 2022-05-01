import { createRef, useState } from 'react';
import CanvasDraw from 'react-canvas-draw';

import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';

import categories from './data/categories.json';

const DEBUG = true;
const ENDPOINT = DEBUG ? 'http://localhost:8000' : '';


export default function Canvas() {
  let canvas = createRef();
  let uuid = uuidv4();
  const [prediction, setPrediction] = useState(null);
  const [guess, setGuess] = useState(null);

  const handleSave = async (data) => {
    const res = await axios.post(`${ENDPOINT}/save`, { uuid: uuid, strokes: data });
    setPrediction({class: res.data.class, probability: res.data.probability});
  }

  return (
    <div className="flex flex-col items-center justify-center">
      { guess &&
        <div class="absolute bottom-0 left-0 bg-orange-100 border-orange-400 text-orange-700 px-8 py-3 m-3 rounded-xl" role="alert">
          <p class="text-2xl font-normal">Draw {(/[aeiou]/.test(guess[0])) ? 'an' : 'a'} {guess}</p>
        </div>
      }

      { prediction &&
        <div class="absolute bottom-0 right-0 bg-blue-100 border-blue-500 text-blue-700 px-8 py-3 m-3 rounded-xl" role="alert">
          <p class="text-2xl font-normal">{prediction.probability < 0.7 ? '🤔' : '😎'} I'm {Math.round(prediction.probability * 10000) / 100}% confident that it's a {prediction.class}</p>
        </div>
      }
      {/* Top row of buttons */}
      <div className="flex flex-row justify-center">
        {/* Undo button */}
        <button
          className="block text-xl m-2"
          onClick={() => {
            // get random category from categories.json
            while (true) {
              const category = categories[Math.floor(Math.random() * categories.length)];
              if (category !== guess) {
                setGuess(category);
                return;
              }
            }
          }}
        >
          Play
        </button>

        {/* Undo button */}
        <button
          className="block text-xl m-2"
          onClick={() => {
            canvas.undo();
          }}
        >
          Undo
        </button>

        {/* Clear button */}
        <button
          className="block text-xl m-2"
          onClick={() => {
            canvas.eraseAll();
          }}
        >
          Clear
        </button>
      </div>

      {/* Canvas object */}
      <CanvasDraw
        className="border-4 border-black shadow-2xl m-4"
        ref={canvasDraw => (canvas = canvasDraw)}
        hideInterface
        hideGrid
        lazyRadius={0}
        brushRadius={2.5}
        brushColor={"#000"}
        canvasWidth={750}
        canvasHeight={500}
      />

      {/* Save button */}
      <button
        class="block accent text-3xl"
        onClick={() => {
          handleSave(canvas.getSaveData());
        }}
      >
        Submit
      </button>
    </div>
  );
}
