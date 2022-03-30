import Canvas from './Canvas';

export default function App() {
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

      {/* Canvas */}
      <Canvas />
    </div>
  );
}
