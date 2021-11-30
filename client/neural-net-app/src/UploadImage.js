import React, {Component} from 'react';
import axios from 'axios';
import ClipLoader from "react-spinners/ClipLoader";
import Autocomplete from "@material-ui/lab/Autocomplete";
import { TextField } from "@mui/material";

class UploadImage extends Component {
	API_BASE_URL = 'http://20.124.189.70:8000';
	API_URL = this.API_BASE_URL + '/predict';
	
	
	state = {
		model: '',
		image: null,
		showModal: false,
		predictResult: null,
		modelOptions: null,
		areModelOptionsReady: false
	};
	
	componentDidMount = () => {
		console.log(this.isLoading);
		axios.get(this.API_BASE_URL + '/models-list')
		.then(res => {
			this.setState({modelOptions: res.data.result});
			this.setState({areModelOptionsReady: true});
			console.log(this.state.modelOptions);
		})
		.catch(err => {
			console.log(err);
		});
	}
	
	handleChange = (e) => {
		this.setState({
			[e.target.id]: e.target.value
		})
	};
	
	handleImageChange = (e) => {
		this.setState({
			image: e.target.files[0]
		})
	};
	
	handleSubmit = (e) => {
		e.preventDefault();
		let form_data = new FormData();
		form_data.append('image', this.state.image, this.state.image.name);
		form_data.append('model', this.state.model);
		let url = this.API_URL;
		axios.post(url, form_data, {
			headers: {
				'content-type': 'multipart/form-data'
			}
		})
		.then(res => {
			console.log(res.data);
			this.setState({showModal: true});
			var data = res.data.result;
			this.setState({predictResult: data});
		})
		.catch(err => {
			console.log(err);
		});
		
	};
	
	setOpen = (e) => {
		this.setState({open:e});
	};
	
	setInputValue = (e) => {
		this.setState({model:e});
	};
	
	render() {
		return (
			<div className="ImageSubmit">
				<form onSubmit={this.handleSubmit}>
					<p>
						<Autocomplete
							open={this.state.open}
							onOpen={() => {
							  // only open when in focus and inputValue is not empty
							  if (this.state.model) {
								this.setOpen(true);
							  }
							}}
							onClose={() => this.setOpen(false)}
							inputValue={this.state.model}
							onInputChange={(e, value, reason) => {
							  this.setInputValue(value);

							  // only open when inputValue is not empty after the user typed something
							  if (!value) {
								this.setOpen(false);
							  }
							}}
							options={this.state.modelOptions}
							renderInput={(params) => (
							  <TextField {...params} label="Model Selection" variant="outlined" defaultValue="Select a Model to Use"/>
							)}
						  />
					</p>
					<p>
						<input type="file" id="image" accept="image/png, image/jpeg" onChange={this.handleImageChange} required/>
					</p>
					<input type='submit'/>
					<p>
						Result:
					</p>
					<p>{this.state.predictResult}</p>
				</form>
			</div>
		);
	}
}

export default UploadImage;