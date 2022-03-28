import {createRef} from 'react';
import CanvasDraw from 'react-canvas-draw';

export default function Canvas() {
  var canvas = createRef();

  return (
    <div className="flex flex-col items-center justify-center">
      {/* Title */}
      <h1 className="text-8xl font-extrabold m-8">
        Pictionary
      </h1>

      {/* Description */}
      <text className="text-4xl font-normal mb-8">
        Draw something in the canvas and let our neural network try to guess what it is!
      </text>

      {/* Top row of buttons */}
      <div className="flex flex-row justify-center">
        {/* Undo */}
        <button
          className="block text-xl m-2"
          onClick={() => {
            canvas.undo();
          }}
        >
          Undo
        </button>

        {/* Clear canvas */}
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
        className="border-4 border-black m-4"
        ref={canvasDraw => (canvas = canvasDraw)}
        hideInterface
        hideGrid
        lazyRadius={0}
        brushRadius={2.5}
        brushColor={"#000"} 
        canvasWidth={750} 
        canvasHeight={500}
      />

      {/* Save canvas */}
      <button
        class="block accent text-3xl"
        onClick={() => {
          localStorage.setItem(
            "savedDrawing",
            canvas.getSaveData()
          );
        }}
      >
        Save
      </button>
    </div>
  );
}
