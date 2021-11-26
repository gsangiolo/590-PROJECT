import './App.css';
import UploadImage from './UploadImage';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src='./img/100128.jpg' alt='logo'/>
        <p>
          Explore Galaxy Identification with Our Tools Below!
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
      <body>
	  <p>
	    Search Bar Here! Search for images!
	  </p>
	  <p>
	    Browse examples from our gallery -- selector for classes
	  </p>
	  <p>
	    Got an image? Upload it here to classify it! (Make the model input field an autocomplete)
	  </p>
	  <UploadImage />
	  <p>
	    Browse Model Options
	  </p>
        </body>
    </div>
  );
}

export default App;
