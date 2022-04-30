from flask import Flask,render_template,request
import joblib

# App intialisation
app=Flask(__name__)

# Loadong the model
model=joblib.load('loanpredictor.pkl')

# Render the file to HTML page

@app.route('/')
def loan():
    return render_template('loaninput.html')

@app.route('/input', methods=['post'])
def input():
    amt=request.form.get('amt')
    term=request.form.get('term')
    crd=request.form.get('crd')
    gnd=request.form.get('gnd')
    mrd=request.form.get('mrd')
    edu=request.form.get('edu')
    slf=request.form.get('slf')
    ppt=request.form.get('ppt')
    inc=request.form.get('inc')
    result=model.predict([[amt,term,crd,gnd,mrd,edu,slf,ppt,inc]])

    if result[0]==1:
        res='The person is eligible for loan'
    else:
        res='The person is not eligible for loan'
    return render_template('prediction.html', out=res)

app.run(debug=True)
