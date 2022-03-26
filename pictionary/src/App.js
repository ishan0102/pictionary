import CanvasDraw from "react-canvas-draw";

function App() {
  return (
    <div className="flex flex-col items-center justify-center">
      <h1 className="text-8xl font-extrabold mt-8">
        Pictionary
      </h1>
      <CanvasDraw
        className="border-4 border-black m-16"
        hideInterface={true}
        lazyRadius={0}
        brushRadius={2.5}
        brushColor={"#000"} 
        canvasWidth={750} 
        canvasHeight={500}
        hideGrid={true}
      />
      <text className="text-4xl font-normal">
        Draw something in the canvas and let our neural network try to guess what it is!
      </text>
    </div>
  );
}

export default App;
