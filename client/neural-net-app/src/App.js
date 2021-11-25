import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
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
	  Got an image? Upload it here to classify it!
	</p>
	<p>
	  Selector for which model to use
	</p>
      </body>
    </div>
  );
}

export default App;
