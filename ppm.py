from flask import Flask,render_template,jsonify,request
import numpy as np
import pickle

with open('dt_reg1.pkl','rb') as f:
    model=pickle.load(f)
    
    
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('ppm_data.html')

@app.route('/ppm',methods=['POST'])
def prediction():
    Data_No=int(request.form['Data_no'])
    Differential_pressure=float(request.form['dp'])
    Flow_rate=float(request.form['fr'])
    Time=float(request.form['time'])
    Dust_feed=float(request.form['df'])
    Dust=float(request.form['dust'])
    
    user_data=np.array([Data_No,Differential_pressure,Flow_rate,Time,Dust_feed,Dust],ndmin=2)
    result=float(model.predict(user_data))
    print(result)
    
    
    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True)
