import { createRef, useEffect, useState } from 'react';
import CanvasDraw from 'react-canvas-draw';

import axios from 'axios';

import categories from './data/categories.json';


export default function Canvas() {
  let canvas = createRef();
  let endpointInput  = createRef();

  const [prediction, setPrediction] = useState(null);
  const [guess, setGuess] = useState(null);
  const [ENDPOINT, SET_ENDPOINT] = useState('http://localhost:8000');
  const [visibleUrlBanner, setVisibleUrlBanner] = useState(false);

  const handleSave = async (data) => {
    const res = await axios.post(`${ENDPOINT}/save`, { strokes: data });
    setPrediction({ class: res.data.class, probability: res.data.probability });
  }

  useEffect(() => {
    // 3 seconds prompt of the URL banner
    console.log(`Set endpoint to: "${ENDPOINT}"`);
    const timer = setInterval(() => {
      setVisibleUrlBanner(false);
    }, 3000);

    return () => {
      clearInterval(timer);
      setVisibleUrlBanner(true);
    }
  }, [ENDPOINT]);


  return (
    <div className="flex flex-col items-center justify-center">

      <div className="w-full max-w-sm">
        <div className="flex items-center">
          <input ref={endpointInput} className="font-mono appearance-none bg-transparent border w-full text-gray-700 mr-3 py-1 px-2 leading-tight focus:outline-none" type="text" placeholder="http://localhost:8000" />
          <button 
            className="block text-xl m-2"
            onClick={() => SET_ENDPOINT(endpointInput.current.value)}
          >
            Set
          </button>
        </div>
      </div>

      <div 
        className={`absolute duration-700 top-0 right-0 bg-green-100 border-green-400 text-green-700 px-8 py-3 m-3 rounded-xl ${visibleUrlBanner ? 'opacity-1' : 'opacity-0'}`} role="alert">
        Endpoint: 
        <p className="font-bold">
          {ENDPOINT}
        </p>
      </div>

      {guess &&
        <div className="absolute bottom-0 left-0 bg-orange-100 border-orange-400 text-orange-700 px-8 py-3 m-3 rounded-xl" role="alert">
          <p className="text-2xl font-normal">Draw {(/[aeiou]/.test(guess[0])) ? 'an' : 'a'} {guess}</p>
        </div>
      }

      {prediction &&
        <div className="absolute bottom-0 right-0 bg-blue-100 border-blue-500 text-blue-700 px-8 py-3 m-3 rounded-xl" role="alert">
          <p className="text-2xl font-normal">{prediction.probability < 0.7 ? '🤔' : '😎'} I'm {Math.round(prediction.probability * 10000) / 100}% confident that it's a {prediction.class}</p>
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

      {/* Submit button */}
      <button
        className="block accent text-3xl"
        onClick={() => {
          handleSave(canvas.getSaveData());
        }}
      >
        Submit
      </button>
    </div>
  );
}
