import './App.css';
import UploadImage from './UploadImage';
import ImageSearch from './ImageSearch';
import galaxy from './img/galaxy.png';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <p>
          Explore Galaxy Classification with Our Tools Below!
        </p>
		<img src={galaxy} alt='logo'/>
      </header>
      <body>
	  <p>
	    Got an image? Upload it here to classify it!
	  </p>
	  <UploadImage />
	  <br/>
	  <br/>
	  <p>
	    Search for images! See what images we have in our gallery, and test them on our predictor!
	  </p>
	  <ImageSearch />
	  <br/>
	  <br/>
<div className="Footer" style={{backgroundColor: "lightgrey", fontSize: 10}}>
        <p>
			By George Sangiolo (gss59@georgetown.edu), Hanna Born (hkb9@georgetown.edu), and Yiming Yu (yy628@georgetown.edu)
		</p>
		<p>
			Background Image from http://cdn.onlinewebfonts.com/svg/img_537674.png
		</p>
		<p>
			Data and Image Gallery from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
		</p>
		</div>
        </body>
    </div>
  );
}

export default App;
