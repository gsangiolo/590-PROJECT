import React, {Component} from 'react';
import axios from 'axios';

class UploadImage extends Component {
	
	API_URL = 'http://localhost:5000/predict/'
	
	state = {
		title: '',
		model: '',
		image: null
	};
	
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
		console.log(this.state);
		let form_data = new FormData();
		form_data.append('image', this.state.image, this.state.image.name);
		form_data.append('title', this.state.title);
		form_data.append('model', this.state.model);
		let url = this.API_URL;
		axios.post(url, form_data, {
			headers: {
				'content-type': 'multipart/form-data'
			}
		})
		.then(res => {
			console.log(res.data);
		})
		.catch(err => console.log(err))
	};
	
	render() {
		return (
			<div className="ImageSubmit">
				<form onSubmit={this.handleSubmit}>
					<p>
						<input type="text" placeholder='Title' id='title' value={this.state.title} onChange={this.handleChange} required/>
					</p>
					<p>
						<input type="text" placeholder='Model' id='model' value={this.state.model} onChange={this.handleChange} required/>
					</p>
					<p>
						<input type="file" id="image" accept="image/png, image/jpeg" onChange={this.handleImageChange} required/>
					</p>
					<input type='submit'/>
				</form>
			</div>
		);
	}
}

export default UploadImage;