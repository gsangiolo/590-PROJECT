# 590-PROJECT

By George Sangiolo, Hanna Born, and Yiming Yu

Here is our project to cap off our semester in ANLY 590 (Neural Networks) at Georgetown University!

Included is:

1. Code to build Keras/Tensorflow models that can predict Galaxy Classification/Features. Models we used include: Logistic Regression, Simple DFF (3 layers), Tuned DFF (288 layers), and CNN.

2. A Backend API (written with FastAPI) to deploy our models. This includes features such as predicting classifications of user-submitted galaxy images, and displaying images that we used for training. The Backend can be run with the following command in the root of this project:
`
uvicorn controller:app --host 0.0.0.0 --port 8000
`

3. A Frontend Client (written with React.js) to allow the user to interact with our models in a limited way. The Frontend can be run with the following command in /client/neural-net-app:
`
npm start
`

Note that to run the frontend, the API URLs must be changed in UploadImage.js and ImageSearch.js
