# Import flask
from flask import Flask, render_template,request

# Load the model
# We can use either joblib or pickle model file but we have to follow their respective syntaxes while loading the model
# Here we will use joblib as the syntax is easier
# I have saved the joblib model file in the folder named model
# Let's import joblib and import the model

import joblib

app=Flask(__name__)

# Loading the model
model=joblib.load('Model\Diabetic_80.pkl')

# Now that we have loaded the model so render the html file

@app.route('/')
def dbts():
    return render_template('dbts.html')

@app.route('/input', methods=['post'])
def input():
    preg=request.form.get('preg')
    plas=request.form.get('plas')
    pres=request.form.get('pres')
    skin=request.form.get('skin')
    test=request.form.get('test')
    mass=request.form.get('mass')
    pedi=request.form.get('pedi')
    age=request.form.get('age')
    result=model.predict([[preg,plas,pres,skin,test,mass,pedi,age]])

    if result[0]==1:
        res='The person is diabetic'
    else:
        res='The person is not diabetic'
    # print(data)
    # return 'data received'
    return render_template('predict.html', out=res)
app.run(debug=True)

# Success- The result is shown in the terminal not on the webpage
# To overcom this problem we will create another html file named predict
# Our return statement will render this file and it will show a proper message on the webpage
# Syntax- return render_template(data=data)- Variable name can be anything
# When we render a variable to an html file then we writre that variable inside the {{}} in the html file.
# The example of above html file is predict.html

# Now that we have done with webpage mapping and other stuff but the problem is that we can only access the data on local machine.
# Now I have to make it accissible to the world.
# Now where deployment comes into picture.
# For deployment I have to work with servers and here we are working on HEROKU and AWS
# I ahve created an account on HEROKU.
# Now to reach HEROKU we have to do some steps here.
# Let's take an example- I send this file to my friend and he tries to run it on his machine but the problem is that it will not run on his machine.
# Because a proper library should be installed on my friends machine otherwise it will say this model is not found or that model is not found.
# To overcome this problem whatever project we are doing, we create a requirements.txt file that contains all the libraries to run  the module.
# We can create the requirements.txt file by the running the following command in the terminal inside the project folder
# SYNTAX- pip freeze>requirements.txt
# Now I can send the module in zip file to my friend and to install all the libraries in one go he has to run the following command.
# pip install -r requirements.txt
# This is how it is done in the real time.
# So the most important thing is that requirements.txt file should be there in the project/modle
# I have created requirements.txt inside the ML Deployment folder
# Let's create a virtual environment-- Read more about it in the virtual environment file saved in this folder
# To create a virtual environment- conda create -n <env_name> python==<version>
# To change the virtual environment- conda activate <env_name>
# To get the list of the libraries installed on the virtual environment- pip list
# I have created the virtual environment named Deployment
# Read a bit about vertical and horizontal scaling