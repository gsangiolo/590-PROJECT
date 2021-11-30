import React, { Component } from 'react';
import axios from 'axios';
import Autocomplete from "@material-ui/lab/Autocomplete";
import { TextField } from "@mui/material";


class ImageSearch extends Component {
	API_BASE_URL = 'http://20.124.189.70:8000'
	API_SEARCH_URL = this.API_BASE_URL + '/images/all';
	API_GET_IMAGE_URL = this.API_BASE_URL + '/images/id';
	API_RANDOM_IMAGE_URL = this.API_BASE_URL + '/images/random';
	API_RANDOM_CLASS_URL = this.API_BASE_URL + '/images/class';
	
	state: {
		name: 'React',
		imageData: null,
		imageKeys: [],
		imageSearchId: '',
		prefix: '',
		imageListIds: null,
		open: false,
		randomClassId: null
	};
	
	// 1. Button to get random image, optional parameter for class selection
	// 2. Dropdown selector for classes
	// 3. Image display
	
	componentDidMount = () => {
		axios.get(this.API_SEARCH_URL)
		.then(res => {
			this.setState({imageListIds: res.data.images});
		})
		.catch(err => {
			console.log(err);
		});
	}
	
	handleRandomImage = () => {
		/*
		axios.get(this.API_RANDOM_IMAGE_URL)
		.then(res => {
			this.setState({imageData: res.data});
			console.log(res);
		})
		.catch(err => {
			console.log(err);
		});
		*/
		this.setState({imageData: null})
		this.setState({imageData: this.API_RANDOM_IMAGE_URL + '?hash=' + Date.now()});
	};
	
	handleRandomClassImage = (e) => {
		e.preventDefault();
		this.setState({imageData: null})
		this.setState({imageData: this.API_RANDOM_CLASS_URL + '?img_class=' + this.state.randomClassId + '&hash=' + Date.now()});
	};
	
	handleSubmit = (e) => {
		e.preventDefault();
		this.setState({imageData: null})
		this.setState({imageData: this.API_GET_IMAGE_URL + '?image_id=' + this.state.imageSearchId + '&hash=' + Date.now()});
	};
	
	handleChange = (e) => {
		this.setState({
			[e.target.id]: e.target.value
		});
	};
	
	setOpen = (e) => {
		this.setState({open:e});
	};
	
	setInputValue = (e) => {
		this.setState({imageSearchId:e});
	};
	
	render() {
		return (
			<div className='ImageSearch'>
				<form onSubmit={this.handleSubmit}>
					<p>
					{
						this.state && 
						<Autocomplete
							open={this.state.open}
							onOpen={() => {
							  // only open when in focus and inputValue is not empty
							  if (this.state.imageSearchId) {
								this.setOpen(true);
							  }
							}}
							onClose={() => this.setOpen(false)}
							inputValue={this.state.imageSearchId}
							onInputChange={(e, value, reason) => {
							  this.setInputValue(value);

							  // only open when inputValue is not empty after the user typed something
							  if (!value) {
								this.setOpen(false);
							  }
							}}
							options={this.state.imageListIds}
							renderInput={(params) => (
							  <TextField {...params} label="Image Search" variant="outlined" />
							)}
						  />
					}
					</p>
					<input type='submit'/>
				</form>
				<br/>
				{
					this.state && 
					<TextField value={this.state.randomClassId} id='randomClassId' onChange={this.handleChange} label="Random Image by Class" variant="outlined" />
				}
				<br/>
				<br/>
				<button onClick={this.handleRandomClassImage}>
					Random Class Image!
				</button>
				<br/>
				<br/>
				<button onClick={this.handleRandomImage}>
					Random Image!
				</button>
				<br />
				{
					this.state && this.state.imageData && 
					<img src={this.state.imageData} alt='image'/>
				}
			</div>
		);
	}
}

export default ImageSearch;