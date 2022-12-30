
# import lib
from flask import Flask, render_template, url_for, redirect, request, session, flash, logging, abort
from functools import wraps
from flaskext.mysql import MySQL
import pandas
import sklearn 
import pickle
import os
from flask_mail import Mail, Message
from passlib.hash import sha256_crypt
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import random, string
from random import  choice
from datetime import datetime
import gc
from sklearn import svm
from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import *

from wtforms.fields.html5 import EmailField
from wtforms.fields.html5 import DateField
from sklearn import neural_network, ensemble
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from wtforms import Form,StringField,TextAreaField,PasswordField,validators,IntegerField
from wtforms import  BooleanField, DateTimeField,   SelectField
from wtforms.validators import InputRequired, DataRequired, Email, Required, Length, ValidationError

# save vir env to app
app = Flask(__name__)

#if __name__ == "__main__":
#    app.run(debug=True)

# config the database connection
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = os.environ.get('MYSQL_DATABASE_PASSWORD')
app.config['MYSQL_DATABASE_DB'] = 'pd'


# config the email
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'identify2021.noreply@gmail.com'
app.config['MAIL_PASSWORD'] =  'aaff01aaii2021@'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# save the data & model into folders & config them
app.secret_key = '4zJW=nyT[Bk:4uuY'
UPLOAD_FOLDER = './static/data'   # .csv
UPLOADED_ITEMS_DEST = './static/model'  # .sav
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOADED_ITEMS_DEST'] = UPLOADED_ITEMS_DEST

def connection(): # establech the database conncetion and return the conncetion and cursor
    mysql = MySQL()
    mysql.init_app(app)
    conn = mysql.connect()
    cursor =conn.cursor()
    return conn ,cursor

def email(subject,recipient , html): # gets the subject ,recipient and html body of email and retuen the email massaage to be sent
    return Message(subject, sender = app.config['MAIL_USERNAME'] , recipients = [recipient], html= html)




class NationalID(Form):
    patientID = IntegerField('NationalID', [validators.NumberRange(min=1000000000, max=9999999999, message="National ID/Iqama must be 10 digits")])

@app.route("/")
def index():
    #session['logged_in'] = True
    #session['role'] = 'medical specialist'
    #session['role'] = 'admin'
    #session['role'] = 'registered user'
    #session['username'] = 'roomomm1' # admin
    #session['username'] = '1089888711' # ru
    #session['username'] = 'NouraAlRoomi' # ms
    return render_template('index.html')

def logingout():
    if 'logged_in' in session:
        session.pop('logged_in')
        session.pop('role')
        session.pop('username')

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return render_template('404.html')

    return wrap
    
@app.route("/logout")
@login_required
def logout():
    logingout()
    return render_template('index.html')
    
@app.route("/diagnosishistory", methods=['GET', 'POST'])
@login_required
def diagnosishistory():
    #unneededresult=cursor.execute("SELECT testresult,diseasetype,accuracy from result as r,model as m where  m.modelid=r.modelid and resultid= %s",(resultID))
    #data = cursor.fetchone()
    
    name='Reem Alassaf'
    usertype=session['role']
    if usertype=='medical specialist':
        form = NationalID(request.form)
        if request.method == 'POST' and form.validate():


            session['national']=form.patientID.data
            # Commit to DB
            #mysql.connection.commit()

            # Close connection
             #cur.close()



            return redirect(url_for('diagnosishistory'))

        elif session.get('national') is not None:

            # Execute query
            #cursor.execute("SELECT * FROM RESULT WHERE NationalID= %s", (patientID))
            try:
                conn, cursor= connection()
                cursor.execute("SELECT ResultID, date, diseasetype,testresult ,accuracy FROM RESULT as r, Model as m WHERE m.modelID=r.modelID and NationalID= %s", (session['national']))

                #session['result']= cursor.fetchall();
                result= cursor.fetchall()


                # might need changing
                session['national']='';
            finally:
                cursor.close()
                conn.close()
            
            return render_template('MSdiagnoseHistory.html', result=result,form=form,name=name,usertype=usertype)

        else:
            return render_template('MSdiagnoseHistory.html', form=form,name=name,usertype=usertype)

    elif usertype=='registered user':

        RUID=session['username']
        # Execute query
        # cursor.execute("SELECT * FROM RESULT WHERE NationalID= %s", (patientID))
        try:
            conn, cursor= connection()
            cursor.execute("SELECT ResultID, date, diseasetype,testresult ,accuracy FROM RESULT as r, Model as m WHERE m.modelID=r.modelID and NationalID= %s", (RUID))

        #session['result']= cursor.fetchall();
            result= cursor.fetchall()


        # might need changing
            session['national']='';
        finally:
            cursor.close()
            conn.close()
            
        return render_template('MSdiagnoseHistory.html', result=result,name=name,usertype=usertype)
    else:
        return render_template('404.html')


# Rebuild Model

@app.route('/rebuildModel')
@login_required
def display():
    if session['role'] == 'admin':
        try:
            conn, cur = connection()
            
            
            # fetch the temp models
            cur.execute('select ModelName, ModelType, TrainingPercent, Accuracy, TotalInstances, TestInstances from tempmodel')
            tempModels = cur.fetchall()
            
            
            # fetch the active models
            cur.execute("select ModelName, DiseaseType, TrainingPercent, Accuracy, TotalInstances, TestInstances from model where Active = '1'")
            activeModels = cur.fetchall()
            
            
            if 'updateM' in session:
                flash("Success! Diagnostic Model has been Updated", 'info')
                session.pop('updateM')
                
        finally:
            cur.close()
            conn.close()
        return render_template('adminRebuildModel.html', tempModels = tempModels, activeModels = activeModels)
    else:
        return render_template('404.html')


# Generate Model

@app.route('/generateModel', methods=['POST'])
def generateModel():
   # Start first if  (General, no need to add more)
    if request.method == 'POST':
        training = request.form['trainingP']
        test = (100 - int(training))/100
        disease = request.form.get('diseaseList')
        file = request.files['dataFile']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        savedFile = './static/data/' + filename
       
       #.................................................
       #...........ADD THE DISEASE MODEL HERE ..........
       #.................................................
       
       
        # Start second if (Which has the best technique for each disease, add the new techniques )
        
        if(disease == 'Diabetes'):
            names = [1,2,3,4]   # All Attributes
            items1=[1,2,3]  # Attributes without class
            items=[4]   # Class Attribute
            
            data = pandas.read_csv(savedFile, names=names)   # read .csv file
            
            train, test = train_test_split(data, test_size=test , random_state = 0) # divid the data
            
            x_data, x_class = train.filter(items1), train.filter(items)   # save data for training
            
            y_data, y_class = test.filter(items1), test.filter(items)   # save data for testing
            
             # define the technique with the best parameters
             
            clf= sklearn.neural_network.MLPClassifier(solver='lbfgs',hidden_layer_sizes=(5), learning_rate='constant', learning_rate_init=0.3, random_state=0)
           
        elif(disease == 'CKD'):
            names= [1,2,3]
            items1=[1,2]
            items=[3]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf= sklearn.neural_network.MLPClassifier(solver='lbfgs', activation = 'identity', hidden_layer_sizes=(100), random_state=0)
            
        elif(disease == 'CHD'):
            names= [1,2,3,4,5,6,7,8]
            items1=[1,2,3,4,5,6,7]
            items=[8]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf=sklearn.ensemble.RandomForestClassifier(random_state=9,min_samples_leaf=4,min_samples_split=20,max_depth=15)
        
        elif(disease == 'RA'):
            names= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
            items1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
            items=[27]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf=sklearn.svm.SVC(kernel='rbf',C=25,gamma=0.001)
     
        elif(disease == 'Schizophrenia'):
            names= [1,2,3,4,5,6,7,8,9]
            items1=[1,2,3,4,5,6,7,8]
            items=[9]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 11)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
        
            clf=sklearn.svm.SVC(C= 10,gamma='scale',kernel= 'rbf')
            
        elif(disease == 'Asthma'):
            names= [1,2,3,4,5,6,7,8,9,10]
            items1=[1,2,3,4,5,6,7,8,9]
            items=[10]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf=sklearn.svm.SVC(C= 10, gamma= 0.0001, kernel= 'rbf')
            
        elif(disease == 'TC'):
            names= [1,2,3,4,5,6,7,8]
            items1=[1,2,3,4,5,6,7]
            items=[8]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 42)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf= sklearn.neural_network.MLPClassifier(solver='adam', activation = 'tanh', hidden_layer_sizes=(100), learning_rate='constant', alpha=0.01)

        elif(disease == 'Hypothyroidism'):
            names= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            items1=[2,4,7,9,10]
            items=[19]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            #print(x_class)
            estimators= []
            model1 = KNeighborsClassifier()
            estimators.append(('knn1',model1))
            model2 = sklearn.svm.SVC(C=9, gamma=1, kernel='linear',probability=True)
            estimators.append(('svm1',model2))
            model3 = sklearn.svm.SVC(C=9, gamma=1, kernel='rbf',probability=True)
            estimators.append(('svm2',model3))
            clf=sklearn.ensemble.VotingClassifier(estimators,voting='soft')
        elif(disease == 'PC'):
            names= [1,2,3,4,5,6,7,9,10]
            items1=[3,4,5,6]
            items=[10]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            #print(x_class)
            x_class = x_class.values.ravel()
            clf=sklearn.svm.SVC(C=100, gamma=0.1)
        elif(disease == 'MS'):
            names= [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
            items1=[3,6,7,8,15,16,20]
            items=[22]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            #print(x_class)
            x_class = x_class.values.ravel()
            clf=sklearn.svm.SVC(C=2, gamma=0.1, kernel='sigmoid')
        elif(disease == 'Alzheimer'):
            names= [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            items1=[1,2,3,4,5,6,7,8,9,10,11,12,13]
            items=[14]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf=SVC(C=4, gamma=1, random_state=0)   
        elif(disease == 'Glaucoma'):
            names= [1,2,3,4,5,6,7,8]
            items1=[1,2,3,4,5,6,7]
            items=[8]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf= KNeighborsClassifier(metric='manhattan', n_neighbors=17)
        elif(disease == 'Lung Cancer'):
            names= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            items1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            items=[16]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            clf=sklearn.svm.SVC(C= 15, gamma=0.001 , kernel= 'rbf')
        elif(disease == 'ADHD'):
            names= [1,2,3,4,5,6,7,8,9]
            items1=[1,2,3,4,5,6,7,8]
            items=[9]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            rf=RandomForestClassifier(n_estimators=95,max_depth=6,min_samples_split=5,max_leaf_nodes=48,bootstrap= 'True',random_state=1,max_features =6)
            clf=AdaBoostClassifier(n_estimators=50, base_estimator=rf, learning_rate=1, random_state=1)
            
        elif(disease == 'Breast Cancer'):
            names= [1,2,3,4,5,6]
            items1=[1,2,3,4,5]
            items=[6]
            data = pandas.read_csv(savedFile, names=names)
            train, test = train_test_split(data, test_size=test , random_state = 1)
            x_data, x_class = train.filter(items1), train.filter(items)
            y_data, y_class = test.filter(items1), test.filter(items)
            level0=list()
            level0.append(('svc', SVC(C=0.1,gamma=1,kernel="linear", random_state = 1)))
            level0.append (('lr', LogisticRegression(C=1,max_iter=2500,penalty="l1", solver="saga",random_state= 1)))
            level1=LogisticRegression(C=1.623776739188721, max_iter=100, solver='lbfgs',random_state = 1)
            clf=sklearn.ensemble.StackingClassifier(estimators=level0,final_estimator=level1)

        # End first if
        
        # run the model and save it in .sav format
        clf.fit(x_data, x_class)
        currentDate = datetime.now()
        modelName = currentDate.strftime("%Y-%m-%d %H.%M.%S") + '.sav'
        
        
        # save the model name in the database file (dump)
        pickle.dump(clf, open('./static/model/'+modelName, 'wb'))
        loaded_model = pickle.load(open('./static/model/'+modelName, 'rb'))
        result = loaded_model.score(y_data, y_class)
        prd=loaded_model.predict(y_data)
 
        
        # save the accuracy with the attibutes used in training and testing in DB
        acc = str(float(result))
        totalInst = str(len(data))
        testInst = str(len(y_data))
        
        try:
            conn, cur = connection()
            cur.execute("insert into tempmodel (ModelName, ModelType, TrainingPercent, Accuracy, TotalInstances, TestInstances) values('"+modelName+"','"+disease+"','"+training+"','"+acc+"','"+totalInst+"','"+testInst+"')")
            conn.commit()
        finally:
            cur.close()
            conn.close()
        flag = 'T'
        #ctypes.windll.user32.MessageBoxW(0, disease+" model is generated successfully", "", 0)
        flash("Model is Generated Successfully", 'info')
        return redirect(url_for('display'))
    
# Update Model

@app.route('/updateModel', methods=['POST'])
def updateModel():
    if request.method == 'POST':
        aModelName = request.form['mName']
        aModelName = aModelName[12:]
        try:
            conn , cur = connection()
            cur.execute("select modelType, TrainingPercent, Accuracy, TotalInstances, TestInstances from tempmodel where ModelName ='"+aModelName+"'")
            result = cur.fetchall()
            if cur.rowcount == 0 :
                session['updateM'] = True
                return redirect(url_for('display'))
            resRow = result[0]
            aDisease = resRow[0]
            aTraining = str(resRow[1])
            aAcc = str(resRow[2])
            aTotalInst = str(resRow[3])
            aTestInst = str(resRow[4])
            cur.execute("update model set Active = '0' where DiseaseType = '"+aDisease+"'")
            conn.commit()
            cur.execute("insert into model (ModelName, DiseaseType, Accuracy, TotalInstances, TestInstances, TrainingPercent, Active) values('"+aModelName+"','"+aDisease+"','"+aAcc+"','"+aTotalInst+"','"+aTestInst+"','"+aTraining+"', '1')")
            conn.commit()
        
            cur.execute("select ModelName from tempmodel where ModelName != '"+aModelName+"' and ModelType = '"+aDisease+"'")
            result2 = cur.fetchall()
            #print(result2)
            for row in result2:
                os.remove(os.path.join(app.config['UPLOADED_ITEMS_DEST'], row[0]))
            
            cur.execute("delete from tempmodel where ModelType = '"+aDisease+"'")
            conn.commit()
		
        finally:
            cur.close()
            conn.close()
        session['updateM'] = True
        return redirect(url_for('display'))

def send_confirmation_email(user_email, name , html): # funtion that sends email confirmtion email with token to cofrim email address
    confirm_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    # create the token to cofrim email address
    confirm_url = url_for(
        'confirm_email',
        token=confirm_serializer.dumps(user_email, salt='email-confirmation-salt'),
        _external=True)
    #split the user name to first and last name
    FLname = name.split(" ")
    # render the change_email_confirmation html template with the user first name and token  
    html = render_template( html,confirm_url=confirm_url,user=FLname[0] )
    # send the email to user with the change_email_confirmation html template
    mail.send(email('Confirm Your Email Address', user_email, html))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(405)
def page_not_found(e):
    return render_template('404.html'), 404

#@app.errorhandler(500)
#def page_not_found(e):
#    return render_template('404.html'), 404


class Email(Form):
    uemail = StringField('Email', [DataRequired(),validators.Email() ])#,validators.length(min=10, max=50, message="Please enter your email address.") ] )

@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
        form = Email(request.form)
    #try:
        if request.method == 'GET':
            # return render_template('forgot.html')
            return render_template('forgot.html',form=form)
        
        elif request.method == 'POST' and form.validate():
            uemail= form.uemail.data
            conn , cursor = connection()
            
            q = "SELECT username ,name ,password ,Email,  EmailConfirmed   FROM PD.Account WHERE Email = '"+ uemail +"'"
            cursor.execute(q)
            data = cursor.fetchone()
            
            if data is not None:
                username = data[0]
                name= data[1]
                passw = data[2]
                useremail = data[3]
                emailfalg = data[4]
                if emailfalg == '1':
                    tpassword = password_gen()
                    temppass=sha256_crypt.encrypt(str(tpassword))
                
                    TPdate=str(date.today())
                    cursor.execute("UPDATE Account SET TempPassFlag ='1', Password = '"+ temppass +"', TempPassDate = '"+ TPdate +"' WHERE username = '"+ username +"'")
                    conn.commit()
                    flash("You can enter your Account know")
                    FLname = name.split(" ")
                    html = render_template('forgetpass_email.html', user=FLname[0], temp= tpassword, form=uemail)
                    msg = email("Reset password", useremail, html )
                    mail.send(msg)
                else:
                    error = "Email unconfrimed, please confirm your new email address (link sent to you in email)."
                    send_confirmation_email(useremail, name , 'change_email_confirmation.html')
                    return render_template("forgot.html", form=form, error=error)
                form1 = Login(request.form)
                return render_template("login.html", form =form1)
            #   
            else:
                error="Email does not exist"
                return render_template("forgot.html", form=form, error=error)
        else:
            #error="Email does not exist"
            return render_template("forgot.html", form=form)



class Login(Form):
    ID = StringField('Username' )#,[validators.required()])
    password= PasswordField('Password')#, [validators.required()])#,validators.Length(min=1, max=255)])

@app.route('/login', methods=['GET', 'POST'])
def login():
        form = Login(request.form)
        
        if request.method == 'GET':
            return render_template('login.html', form =form )


        elif request.method == 'POST' and form.validate():
            ID  = form.ID.data
            password = form.password.data
            if ID.strip() == "" or password.strip() == "":
                error="All fields must be felld" 
                return render_template('login.html', form=form, error=error)
            conn , cursor = connection()
            q = "SELECT name, password ,role FROM PD.Account WHERE username = '"+ ID +"'"
            cursor.execute(q)
            data = cursor.fetchone()
            if data is not None:
                name = data[0]
                #print (name)
                ps = data[1]
                #print(ps)
                Role = data[2]
                #print (Role)
                v = sha256_crypt.verify(password, ps)
                #print(v)
                if ( v == True ):

                    if (Role == 'admin') :
                        session['logged_in']=True
                        session['username']=ID
                        session['role'] = Role
                        return redirect(url_for('profile'))

                    elif (Role == 'registered user'):
                        session['logged_in']=True
                        session['username']=ID
                        session['role'] = Role
                        return redirect(url_for('profile') )

                    elif (Role == 'medical specialist'):
                        session['logged_in']=True
                        session['username']=ID
                        session['role'] = Role
                        return redirect(url_for('profile'))

                else:
                    return render_template('login.html', error="Invalid username or password" , form=form)
            else:
                error="Invalid username or password" 
                return render_template('login.html', form=form, error=error)

        
        elif request.method == 'POST':
                error="Invalid username or password"
                return render_template('login.html',  form=form)


@app.route('/idcheck', methods=['POST'])
def IDcheck():
    try:
        conn , cursor = connection()
        PID = request.form['NationalID']
        q = "SELECT Name ,Gender , BirthDate FROM PD.Patient where NationalID = '" +  PID + "' "
        cursor.execute(q)
        if cursor.rowcount > 0:
            data = cursor.fetchone()
            dbname= data[0]
            dbgender= str(data[1])
            dbbdate = data[2]
            return render_template("MSdiagnose.html",patient = "True", add = "False",PID = PID,  name =dbname , gender= dbgender, bdate =  dbbdate )
    finally:
        cursor.close()
        conn.close()
    return render_template("MSdiagnose.html",patient = "False", PID = PID)
    

@app.route('/addPatient', methods=['POST'])
def addpatient():
    try:
        conn , cursor = connection()
        PID = request.form['NPID']
        dbname = request.form['name']
        dbbdate = request.form['birthdate']
        dbgender = request.form['gender']
        q = "INSERT INTO `PD`.`Patient` (`NationalID`, `Name`, `BirthDate`, `Gender`) VALUES ('"+PID +"', '"+dbname+"', '"+dbbdate+"', '"+dbgender+"')"
        cursor.execute(q)
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    
    return render_template("MSdiagnose.html",patient = "True", add = "True", PID = PID, name =dbname , gender= dbgender, bdate =  dbbdate )

#.................................................
#...........ADD THE DISEASE METHOD HERE ..........
#.................................................

@app.route('/diagnosis')
def dmDiagnosis():
    #session['role'] = 'Medical Specialist'
    Role = ""
    if 'role' in session:
        Role = session['role']
    
    if Role == 'medical specialist':
        return render_template("MSdiagnose.html")
    if Role == 'registered user':    
        return render_template("RUdiagnosis.html")
    if Role == 'admin':
        return render_template("404.html")
    return render_template("diagnose.html" )

@app.route('/CKDdiagnosis', methods=['POST'])
def ckddiagnosis():
    try:
       	conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'CKD' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName =   data[1]
        accuracy =   data[2]
        accuracy = float(accuracy)*100
        #print(modleID,modelName)
        loadedmodel = pickle.load(open('./static/model/' +modelName,'rb')) # **check Path**
        
        crt = request.form['crt']
        bun = request.form['bun']
        
        #print(cbt,slg,hct,mpv,wbc)
        prd=loadedmodel.predict([[float(bun) ,float(crt)]])
        #prd=loadedmodel.predict([[float(slg) ,float(hct),float(mpv)]])
        #print(float(cbt),float(slg) ,float(hct),float(mpv),float(wbc))
        #print(prd[0])
        if prd[0] == "CKD":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Chronic Kidney Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
                #ID = "1089888711"
            elif session['role'] == 'registered user':
                ID = session['username']
                #ID = "1089888711"
            #ID = "1089888711" #**get it form the session**
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            #print(RID)
            #session['result_id'] = RID
            
            
              # ** Reems route **
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)
    
@app.route('/DMdiagnosis', methods=['POST'])
def dmdiagnosis():
    try:
       	conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'Diabetes' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName =   data[1]
        accuracy =   data[2]
        accuracy = float(accuracy)*100
        #print(modleID,modelName)
        loadedmodel = pickle.load(open('./static/model/' +modelName,'rb')) # **check Path**
        #cbt = request.form['cbt']
        slg = request.form['slg']
        hct = request.form['hct']
        mpv = request.form['mpv']
        #wbc = request.form['wbc']
        #print(cbt,slg,hct,mpv,wbc)
        prd=loadedmodel.predict([[float(slg) ,float(hct),float(mpv)]])
        #prd=loadedmodel.predict([[float(slg) ,float(hct),float(mpv)]])
        #print(float(cbt),float(slg) ,float(hct),float(mpv),float(wbc))
        #print(prd[0])
        if prd[0] == "DM":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Diabetes Mellitus"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
                #ID = "1089888711"
            elif session['role'] == 'registered user':
                ID = session['username']
                #ID = "1089888711"
            #ID = "1089888711" #**get it form the session**
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            #print(RID)
            #session['result_id'] = RID
            
            
              # ** Reems route **
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)
    
@app.route('/CHDdiagnosis', methods=['POST'])
def CHDdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'CHD' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName =   data[1]
        accuracy =   data[2]
        accuracy = float(accuracy)*100
        print(modelName)
        print('./static/model/' +modelName,'rb')

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

        print(loadedmodel)

        #ADDD WE NEED to add ATTRUBUTES
        genderCHD = request.form['genderCHD']
        ageCHD = request.form['ageCHD']
        bunCHD = request.form['bunCHD']
        dbiliCHD = request.form['dbiliCHD']
        creatCHD = request.form['creatCHD']
        albuminCHD = request.form['albuminCHD']
        tbiliCHD = request.form['tbiliCHD']

        if genderCHD == "Male":
            genderCHD = 0
        else: genderCHD = 1

        print(genderCHD,ageCHD,bunCHD,dbiliCHD,creatCHD,albuminCHD,tbiliCHD)
        prd=loadedmodel.predict([[float(genderCHD) ,float(ageCHD), float(bunCHD), float(dbiliCHD) ,float(creatCHD), float(albuminCHD), float(tbiliCHD)]])
        print(float(genderCHD) ,float(ageCHD), float(bunCHD), float(dbiliCHD) ,float(creatCHD), float(bunCHD), float(dbiliCHD))
        print(prd[0])
        if prd[0] == "1":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Coronary Heart Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
                #ID = "1089888711"
            elif session['role'] == 'registered user':
                ID = session['username']
                #ID = "1089888711"
            #ID = "1089888711" #**get it form the session**
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            #print(RID)
            #session['result_id'] = RID
            
            
              # ** Reems route **
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)



@app.route('/RAdiagnosis', methods=['POST'])
def RAdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'RA' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName =   data[1]
        accuracy =   data[2]
        accuracy = float(accuracy)*100
       # print(modelName)
       # print('./static/model/' +modelName,'rb')

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

        print(loadedmodel)

        #ADDD WE NEED to add ATTRUBUTES
        genderRA = request.form['genderRA']
        ageRA = request.form['ageRA']
        albuminRA = request.form['albuminRA']
        AlkPhosRA = request.form['AlkPhosRA']
        bunRA = request.form['bunRA']
        ClRA = request.form['ClRA']
        CO2RA = request.form['CO2RA']
        creatRA = request.form['creatRA']
        dbiliRA = request.form['dbiliRA']
        GGTPRA = request.form['GGTPRA']
        hgbRA = request.form['hgbRA']
        hctRA = request.form['hctRA']
        KRA = request.form['KRA']
        LDHRA = request.form['LDHRA']
        MCHRA = request.form['MCHRA']
        MCHCRA = request.form['MCHCRA']
        MCVRA = request.form['MCVRA']
        MPVRA = request.form['MPVRA']
        NaRA = request.form['NaRA']
        PltRA = request.form['PltRA']
        RBCRA = request.form['RBCRA']
        RDWRA = request.form['RDWRA']
        SGOTRA = request.form['SGOTRA']
        SGPTRA = request.form['SGPTRA']
        TbiliRA = request.form['TbiliRA']
        TProteinRA = request.form['TProteinRA']

        if genderRA == "Male":
            genderRA = 0
        else: genderRA = 1

        #print(genderRA,ageRA,albuminRA,AlkPhosRA,bunRA,ClRA,CO2RA,creatRA,dbiliRA,GGTPRA,hgbRA,hctRA,KRA,LDHRA,MCHRA,MCHCRA,MCVRA,MPVRA,NaRA,PltRA,RBCRA,RDWRA,SGOTRA,SGPTRA,TbiliRA,TProteinRA)
        prd=loadedmodel.predict([[float(genderRA) ,float(ageRA), float(albuminRA), float(AlkPhosRA) ,float(bunRA), float(ClRA), float(CO2RA),float(creatRA),float(dbiliRA),float(GGTPRA),float(hgbRA),float(hctRA),float(KRA),float(LDHRA),float(MCHRA),float(MCHCRA),float(MCVRA),float(MPVRA),float(NaRA),float(PltRA),float(RBCRA),float(RDWRA),float(SGOTRA),float(SGPTRA),float(TbiliRA),float(TProteinRA)]])
        #print(float(genderRA) ,float(ageRA), float(albuminRA), float(AlkPhosRA) ,float(bunRA), float(ClRA), float(CO2RA),float(creatRA),float(dbiliRA),float(GGTPRA),float(hgbRA),float(hctRA),float(KRA),float(LDHRA),float(MCHRA),float(MCHCRA),float(MCVRA),float(MPVRA),float(NaRA),float(PltRA),float(RBCRA),float(RDWRA),float(SGOTRA),float(SGPTRA),float(TbiliRA),float(TProteinRA))
        #print(prd[0])
        if prd[0] == "1":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Rheumatoid Arthritis Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
                #ID = "1089888711"
            elif session['role'] == 'registered user':
                ID = session['username']
                #ID = "1089888711"
            #ID = "1089888711" #**get it form the session**
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)


# The new Schizophrenia Diagnosis

@app.route('/Schizophreniadiagnosis', methods=['POST'])
def Schizophreniadiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'Schizophrenia' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName =   data[1]
        accuracy =   data[2]
        accuracy = float(accuracy)*100
       # print(modelName)
       # print('./static/model/' +modelName,'rb')

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

        #print(loadedmodel)

        #ADD WE NEED to add ATTRUBUTES
        genderSchizophrenia = request.form['genderSchizophrenia']
        neuSchizophrenia = request.form['neuSchizophrenia']
        lymSchizophrenia = request.form['lymSchizophrenia']
        hgbSchizophrenia = request.form['hgbSchizophrenia']
        hctSchizophrenia = request.form['hctSchizophrenia']
        mchcSchizophrenia = request.form['mchcSchizophrenia']
        rdwSchizophrenia = request.form['rdwSchizophrenia']
        ureaSchizophrenia = request.form['ureaSchizophrenia']

        if genderSchizophrenia == "Male":
            genderSchizophrenia = 0
        else:
            genderSchizophrenia = 1

        #print("----------------------genderSchizophrenia,neuSchizophrenia,lymSchizophrenia,hgbSchizophrenia,hctSchizophrenia,mchcSchizophrenia,rdwSchizophrenia,ureaSchizophrenia")
        #print(genderSchizophrenia,neuSchizophrenia,lymSchizophrenia,hgbSchizophrenia,hctSchizophrenia,mchcSchizophrenia,rdwSchizophrenia,ureaSchizophrenia)
        
       # print(float(genderSchizophrenia), float(neuSchizophrenia), float(lymSchizophrenia), float(hgbSchizophrenia), float(hctSchizophrenia), float(mchcSchizophrenia), float(rdwSchizophrenia), float(ureaSchizophrenia))
        
        prd=loadedmodel.predict([[float(genderSchizophrenia), float(neuSchizophrenia), float(lymSchizophrenia), float(hgbSchizophrenia), float(hctSchizophrenia), float(mchcSchizophrenia), float(rdwSchizophrenia), float(ureaSchizophrenia)]])

        # End of Attributes Adding


        #print(prd[0])
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"
        disease = "Schizophrenia Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
            elif session['role'] == 'registered user':
                ID = session['username']
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)



# The new Thyroid Cancer Diagnosis

@app.route('/tcdiagnosis', methods=['POST'])
def tcdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'TC' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName =      data[1]
        accuracy =     data[2]
        accuracy = float(accuracy)*100
       # print(modelName)
       # print('./static/model/' +modelName,'rb')

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

       # print(loadedmodel)

        #ADDD WE NEED to add ATTRUBUTES
        genderThyroid = request.form['genderThyroid']
        ageThyroid = request.form['ageThyroid']
        HematocritThyroid = request.form['HematocritThyroid']
        MCHCThyroid = request.form['MCHCThyroid']
        MPVThyroid = request.form['MPVThyroid']
        RBCThyroid = request.form['RBCThyroid']
        WBCThyroid = request.form['WBCThyroid']

        if genderThyroid == "Male":
            genderThyroid = 0
        else: genderThyroid = 1

        #print(genderThyroid,ageThyroid,HematocritThyroid,MCHCThyroid,MPVThyroid,RBCThyroid,WBCThyroid)
        prd=loadedmodel.predict([[float(genderThyroid),float(ageThyroid),float(HematocritThyroid),float(MCHCThyroid),float(MPVThyroid),float(RBCThyroid),float(WBCThyroid)]])
        #print(float(genderThyroid),float(ageThyroid),float(HematocritThyroid),float(MCHCThyroid),float(MPVThyroid),float(RBCThyroid),float(WBCThyroid))
        
        #print(prd[0])
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"
        disease = "Thyroid Cancer Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
            elif session['role'] == 'registered user':
                ID = session['username']
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)




# The new Asthma  Diagnosis

@app.route('/asdiagnosis', methods=['POST'])
def asdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'Asthma' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName =      data[1]
        accuracy =     data[2]
        accuracy = float(accuracy)*100
       # print(modelName)
       # print('./static/model/' +modelName,'rb')

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

       # print(loadedmodel)

        #ADDD WE NEED to add ATTRUBUTES
        genderAsthma = request.form['genderAsthma']
        ageAsthma = request.form['ageAsthma']
        BasophilsAsthma = request.form['BasophilsAsthma']
        HematocritAsthma = request.form['HematocritAsthma']
        HemoglobinAsthma = request.form['HemoglobinAsthma']
        MCHAsthma = request.form['MCHAsthma']
        MCHCAsthma = request.form['MCHCAsthma']
        MPVAsthma = request.form['MPVAsthma']
        WBCAsthma = request.form['WBCAsthma']

        if genderAsthma == "Male":
            genderAsthma = 0
        else: genderAsthma = 1

     #   print(genderAsthma,ageAsthma,BasophilsAsthma,HematocritAsthma,HemoglobinAsthma,MCHAsthma,MCHCAsthma,MPVAsthma,WBCAsthma)
        prd=loadedmodel.predict([[float(genderAsthma),float(ageAsthma),float(BasophilsAsthma),float(HematocritAsthma),float(HemoglobinAsthma),float(MCHAsthma),float(MCHCAsthma),float(MPVAsthma),float(WBCAsthma)]])
      #  print(float(genderAsthma),float(ageAsthma),float(BasophilsAsthma),float(HematocritAsthma),float(HemoglobinAsthma),float(MCHAsthma),float(MCHCAsthma),float(MPVAsthma),float(WBCAsthma))
        
       # print(prd[0])
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"
        disease = "Asthma Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
            elif session['role'] == 'registered user':
                ID = session['username']
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)

#OUR MODELS 2021

@app.route('/Hypodiagnosis', methods=['POST'])
def Hypodiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'Hypothyroidism' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName = data[1]
        accuracy = data[2]
        accuracy = float(accuracy)*100

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

        #genderHypo = request.form['genderHypo']
        ageHypo = request.form['ageHypo']
        BPSHypo = request.form['BPSHypo']
        #HtHypo = request.form['HtHypo']
        #WeiHypo = request.form['WeiHypo']
        #BMIHypo = request.form['BMIHypo']
        #BSAHypo = request.form['BSAHypo']
        #TempHypo = request.form['TempHypo']
        RespHypo = request.form['RespHypo']
        MCVHypo = request.form['MCVHypo']
        POHypo = request.form['POHypo']
        #HemogHypo = request.form['HemogHypo']
        #HemHypo = request.form['HemHypo']
        #Ft4Hypo = request.form['Ft4Hypo']
        #Ft3Hypo = request.form['Ft3Hypo']
        #VDHypo = request.form['VDHypo']
        
        """
        if genderHypo == "Male":
            genderHypo = 1
        else:
            genderHypo = 0
        """
        prd=loadedmodel.predict([[float(ageHypo),float(BPSHypo),float(RespHypo),float(MCVHypo),float(POHypo)]])
        
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"
        disease = "Hypothyroidism Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
            elif session['role'] == 'registered user':
                ID = session['username']
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)
    
@app.route('/PCdiagnosis', methods=['POST'])
def PCdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'PC' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName = data[1]
        accuracy = data[2]
        accuracy = float(accuracy)*100

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)


        PCperimeter = request.form['PCperimeter']
        PCarea = request.form['PCarea']
        PCsmoothness = request.form['PCsmoothness']
        PCcompactness = request.form['PCcompactness']        
        """
        if genderHypo == "Male":
            genderHypo = 1
        else:
            genderHypo = 0
        """
        prd=loadedmodel.predict([[float(PCperimeter),float(PCarea),float(PCsmoothness),float(PCcompactness)]])
        
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"
        disease = "Prostate Cancer Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
            elif session['role'] == 'registered user':
                ID = session['username']
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)
    
@app.route('/MSdiagnosis', methods=['POST'])
def MSdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'MS' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName = data[1]
        accuracy = data[2]
        accuracy = float(accuracy)*100

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)


        MSheight = request.form['MSheight']
        MSSSS = request.form['MSSSS']
        MSRTSA = request.form['MSRTSA']
        MSLSC = request.form['MSLSC']
        MSRDSA = request.form['MSRDSA']   
        MSRDSC = request.form['MSRDSC']   
        MSLONSD = request.form['MSLONSD'] 
        """
        if genderHypo == "Male":
            genderHypo = 1
        else:
            genderHypo = 0
        """
        prd=loadedmodel.predict([[float(MSheight),float(MSSSS),float(MSRTSA),float(MSLSC),float(MSRDSA),float(MSRDSC),float(MSLONSD)]])
        
        if prd[0] == 1:
            result = "Positive"
        else:
            result = "Negative"
        disease = "Multiple Sclerosis Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
            elif session['role'] == 'registered user':
                ID = session['username']
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)

@app.route('/ADdiagnosis', methods=['POST'])
def ADdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'Alzheimer' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName = data[1]
        accuracy = data[2]
        accuracy = float(accuracy)*100
        #print(modelName)
        #print('./static/model/' +modelName,'rb')

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

        print(loadedmodel)

        genderAD = request.form['genderAD']
        AgeAD = request.form['AgeAD']
        PulseAD = request.form['PulseAD']
        Respiratory_RateAD = request.form['Respiratory_RateAD']
        BP_DiastolicAD = request.form['BP_DiastolicAD']
        wbcAD = request.form['wbcAD']
        rbcAD = request.form['rbcAD']
        HemoglobinAD = request.form['HemoglobinAD']
        HematocritAD = request.form['HematocritAD']
        MCVAD = request.form['MCVAD']
        MCHAD = request.form['MCHAD']
        RDWAD = request.form['RDWAD']
        MPVAD = request.form['MPVAD']

        if genderAD == "Male":
            genderAD = 1
        else: 
            genderAD = 2

        prd=loadedmodel.predict([[float(genderAD), float(AgeAD),float(PulseAD),float(Respiratory_RateAD),float(BP_DiastolicAD),float(wbcAD),float(rbcAD),
        float(HemoglobinAD),float(HematocritAD),float(MCVAD),float(MCHAD),float(RDWAD),float(MPVAD)]])
        if prd[0] == "1":
                result = "Positive"
        else:
            result = "Negative"
        disease = "Alzheimer's Disease"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']
                #ID = "1089888711"
            elif session['role'] == 'registered user':
                ID = session['username']
                #ID = "1089888711"
            #ID = "1089888711" #**get it form the session**
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            #print(RID)
            #session['result_id'] = RID
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)

@app.route('/ADHDdiagnosis', methods=['POST'])
def ADHDdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'ADHD' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName = data[1]
        accuracy = data[2]
        accuracy = float(accuracy)*100

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

        genderADHD = request.form['genderADHD']
        ageADHD = request.form['ageADHD']
        indADHD = request.form['indADHD']
        inatADHD = request.form['inatADHD']
        hypADHD = request.form['hypADHD']
        vqADHD = request.form['vqADHD']
        perfADHD = request.form['perfADHD']
        fqADHD = request.form['fqADHD']

        if genderADHD == "Male":
            genderADHD = 1
        else:
            genderADHD = 2

        prd=loadedmodel.predict([[float(ageADHD) ,float(indADHD), float(inatADHD), float(hypADHD) ,float(vqADHD), float(perfADHD),float(fqADHD)]])
        if prd[0] == "1":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Attention Deficit Hyperactivity Disorder (ADHD) "
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']

            elif session['role'] == 'registered user':
                ID = session['username']
                
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)



@app.route('/BCdiagnosis', methods=['POST'])
def BCdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'Breast Cancer' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName = data[1]
        accuracy = data[2]
        accuracy = float(accuracy)*100

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

        print(loadedmodel)

        AgeBC = request.form['AgeBC']
        BMIBC = request.form['BMIBC']
        GlucoseBC = request.form['GlucoseBC']
        HOMABC = request.form['HOMABC']
        ResistinBC = request.form['ResistinBC']

        prd=loadedmodel.predict([[float(AgeBC) ,float(BMIBC), float(GlucoseBC), float(HOMABC) ,float(ResistinBC)]])
        if prd[0] == "1":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Breast Cancer"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']

            elif session['role'] == 'registered user':
                ID = session['username']
                
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)

@app.route('/Glaucomadiagnosis', methods=['POST'])
def Glaucomadiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'Glaucoma' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName = data[1]
        accuracy = data[2]
        accuracy = float(accuracy)*100

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)
        
        atG = request.form['atG']
        eanG = request.form['eanG']
        mhciG = request.form['mhciG']
        vasiG = request.form['vasiG']
        vargG = request.form['vargG']
        varsG = request.form['varsG']
        tmiG = request.form['tmiG']

        prd=loadedmodel.predict([[float(atG) ,float(eanG), float(mhciG) ,float(vasiG), 
        float(vargG),float(varsG),float(tmiG)]])
        if prd[0] == "1":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Glaucoma "
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']

            elif session['role'] == 'registered user':
                ID = session['username']
                
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)



@app.route('/LCdiagnosis', methods=['POST'])
def LCdiagnosis():
    try:
        conn , cursor = connection()
        q = "SELECT ModelID, ModelName , Accuracy FROM PD.Model where  DiseaseType = 'Lung Cancer' and Active='1'"
        cursor.execute(q)
        data = cursor.fetchone()
        modleID = data[0]
        modelName = data[1]
        accuracy = data[2]
        accuracy = float(accuracy)*100

        pathForModel=open('./static/model/'+ modelName,'rb')
        loadedmodel = pickle.load(pathForModel)

        print(loadedmodel)

        genderLC = request.form['genderLC']
        ageLC = request.form['ageLC']
        smokingLC = request.form['smokingLC']
        yellow_FingersLC = request.form['yellow_FingersLC']
        anxietyLC = request.form['anxietyLC']
        peer_PressureLC = request.form['peer_PressureLC']
        chronic_DiseaseLC = request.form['chronic_DiseaseLC']
        fatigueLC = request.form['fatigueLC']
        allergyLC = request.form['allergyLC']
        wheezingLC = request.form['wheezingLC']
        alcoholLC = request.form['alcoholLC']
        coughingLC = request.form['coughingLC']
        shortness_of_BreathLC = request.form['shortness_of_BreathLC']
        swallowing_DifficultyLC = request.form['swallowing_DifficultyLC']
        chest_PainLC = request.form['chest_PainLC']

        if genderLC == "Male":
            genderLC = 1
        else:
            genderLC = 2

        prd=loadedmodel.predict([[float(genderLC), float(ageLC) ,float(smokingLC), float(yellow_FingersLC) ,float(anxietyLC), float(wheezingLC),
        float(peer_PressureLC),float(chronic_DiseaseLC),float(fatigueLC),float(allergyLC), float(coughingLC), float(alcoholLC),
        float(shortness_of_BreathLC),float(swallowing_DifficultyLC),float(chest_PainLC)]])
        if prd[0] == "1":
            result = "Positive"
        else:
            result = "Negative"
        disease = "Lung Cancer"
        if 'role' in session:
            if session['role'] == 'medical specialist':
                ID = request.form['NID']

            elif session['role'] == 'registered user':
                ID = session['username']
                
            q ="INSERT INTO `PD`.`Result` ( `date`, `TestResult`, `NationalID`, `ModelID`) VALUES ('"+str(date.today())+"', '"+result+"', '"+ID+"', '"+str(modleID)+"')"
            cursor.execute(q)
            conn.commit()
            RID = cursor.lastrowid
            
    finally:
        cursor.close()
        conn.close()
    return render_template('diagnoseRR.html',result=result,disease=disease,acc=accuracy)

@app.route('/deleteaccount')
@login_required
def delUser():
    if session['role'] == 'registered user':
        try:
            conn , cursor = connection()
            uname = session['username'] #**get it form the session**
            #uname = "1087282393" #**get it form the session**
            q= "SELECT Email ,Name FROM PD.Account where username = '" +  uname + "' "
            cursor.execute(q)
            data = cursor.fetchone()
            dbemail = data[0]
            #print(dbemail)
            dbname = data[1]
            q = "DELETE FROM account WHERE username = '"+  uname +"'"
            cursor.execute(q)
            conn.commit()
            FLname = dbname.split(" ")
            # render the Deleteuser_email html template with the user first name   
            html = render_template('Deleteuser_email.html', user=FLname[0])
            # send the email to user with the Deleteuser_email html template
            msg = email("Account Deleted",dbemail,   html )
            mail.send(msg)
            flash("Account successfully deleted.", "info")
            logingout()
        finally:
            # close the Database connection 
            cursor.close()
            conn.close()    
        return render_template("Emailconfirm.html" ,title ="Account Deleted" )
    else:
        return render_template("404.html")

@app.route('/confirm/<token>') # this route handles email address confirmation tokens requests
def confirm_email(token):
    try:
        # extract the email address from the token 
        confirm_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        email = confirm_serializer.loads(token, salt='email-confirmation-salt', max_age=86400)
    except:  # check if the token is invalde or has expired
        flash('The confirmation link is invalid or has expired.', 'info')
        # render Emailconfirm html the will disply the token info message 
        return render_template("Emailconfirm.html",title ="Email Confirmation")
    try:
        # get the Database connection
        conn , cursor = connection()
        # fetch the username and EmailConfirmed flag vaule of the emial confirming user account
        cursor.execute("SELECT username , EmailConfirmed FROM PD.Account where email = '" +  email + "' ")
        data = cursor.fetchone()
        uname = data[0]
        econfirmed = data[1]
        if econfirmed == '1': # check if the email has been confrimed
            flash('Account already confirmed. Please login.', 'info')
        else:
        # update emial confirming user account by setting the EmailConfirmed to 1 'confirmed' and ConfirmedEmailOn with with current timestamp
            q = "UPDATE PD.Account SET EmailConfirmed = '1' , ConfirmedEmailOn = '" +  str(datetime.now()) + "' WHERE username = '" +  uname + "' "
            cursor.execute(q)
            conn.commit()
            flash('Thank you for confirming your email address!','info')
        # render Emailconfirm html the will disply the token info message
    finally:
        # close the Database connection
        cursor.close()
        conn.close()
    return render_template("Emailconfirm.html" ,title ="Email Confirmation" )


@app.route('/video')
def video():
    return render_template("video.html")
    
def validate_pass(form, field):
    letter_flag = False
    number_flag = False
    for i in field.data:
        if i.isalpha():
            letter_flag = True
        if i.isdigit():
            number_flag = True
    if not(letter_flag) or not(number_flag):
        raise ValidationError('Password should be at least 8 alphanumeric characters.')

def validate_name(form, field):
    number_flag = False
    for i in field.data:
        if i.isdigit():
            number_flag = True
    if number_flag:
        raise ValidationError('Numbers are not allowed in name.')

def validate_id(form, field):
    if not(field.data.isdigit()) or len(field.data) != 10 :
        raise ValidationError('National ID/Iqama must be 10 digits.')
        

def validate_date(form, field):        
        bdate = field.data
        print(bdate)
        today = date.today()
        years_ago = date.today() - relativedelta(years=115)
        if bdate >= today or years_ago >= bdate :
            raise ValidationError('Invalid date for date of birth.')

class Register(Form):
    name = StringField('Name',[ DataRequired(), validate_name, validators.length(min=3, max=40)])
    ID =  StringField('National ID/Iqama', [DataRequired(), validate_id]) #validators.length(min=10, max=10, 
    uemail = EmailField('Email', [DataRequired(),validators.Email() ])
    gender = SelectField(choices=[ ('',''), ('Female','Female'),('Male', 'Male')], validators = [DataRequired()] )
    #DateTimeField('Date', validators=[DateRange( min=datetime(1900, 1, 1), max=datetime(2019, 1, 1))] ) 
    birthdate = DateField('Date of birth', [validators.DataRequired(),validate_date], format="%Y-%m-%d")
    password = PasswordField('Password', [ validators.DataRequired(), validators.length(min=8, message="Password should be at least 8 alphanumeric characters."), validate_pass, validators.EqualTo('cpassword', message='Passwords do not match.')])
    cpassword = PasswordField('Confirm Password') 
    
    
        
        
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    
        form = Register(request.form)
        
        if request.method == 'GET':
            return render_template('register.html', form =form)


        elif request.method == 'POST' and form.validate():
            name = form.name.data
            ID = form.ID.data
            uemail = form.uemail.data
            gender = form.gender.data
            birthdate = str(form.birthdate.data)
            password =  sha256_crypt.encrypt(str(form.password.data))
            #print(birthdate)
            conn , cursor = connection()


            x =  "SELECT username FROM Account WHERE username = '"+ ID +"'"
            #print(x)
            conn , cursor = connection()
            cursor.execute(x)
            data = cursor.fetchone()
            x =  "SELECT username FROM Account WHERE email = '"+ uemail +"'"
            #print(x)
            cursor.execute(x)
            data1 = cursor.fetchone()
            if data is not None:
                #if int(cursor.rowcount) > 0 :
                #print("1")
                error="The ID is already taken"
                return render_template('register.html', form=form, error=error)
            elif data1 is not None:
                error="Email already exists in another account"
                return render_template('register.html', form=form, error=error)
            else:
                role = "registered user"
                d= """INSERT INTO `PD`.`Account` (`username`, `Name`, `Email`, `Password`, `Role` , `EmailConfirmed`, `EmailConfirmationSentOn` ) VALUES ('"""+ID+"""', '"""+name+"""', '"""+uemail+"""', '"""+password+"""', '"""+role+"""','0','"""+str(datetime.now())+"')"""
                cursor.execute(d)
                conn.commit()
                x =  "SELECT NationalID FROM Patient WHERE NationalID = '"+ ID +"'"
                cursor.execute(x)
                data = cursor.fetchone()
                if data is not None:
                    f = "UPDATE `PD`.`Patient` SET `Name`='"+name+"', `BirthDate`='"+birthdate+"', `Gender`='"+gender+"' WHERE `NationalID`='"+ID+"'"
                else:
                    f = """INSERT INTO `PD`.`Patient` (`NationalID`, `Name`, `BirthDate`, `Gender`) VALUES ('"""+ID+"""', '"""+name+"""', '"""+birthdate+"""', '"""+gender+"')"""
                    
                cursor.execute(f)
                conn.commit()
                #print(uemail)
                send_confirmation_email(uemail,name,'createaccount_email.html' )
                #html = render_template('CRemail.html', user=name)
                #msg = sendemail("Register", uemail, html )
                #mail.send(msg)
                session['logged_in']=True
                session['username']=ID
                session['role'] =role 
                return redirect(url_for('profile'))
                cursor.close()
                conn.close()

        elif request.method == 'POST':
            error="All field must be felld"
            return render_template('register.html', form=form, error=error)


@app.route('/profile')
@login_required# this route handles profile disply requests 
def profile():
    try:
        #uname = "mosalah"
        uname = session['username']
        
        #uname = "1089888711" #**get it form the session**
        #uname ="saahmad"
        # get the Database connection
        conn , cursor = connection()
        # fetch the profile info of the logged in user account: the name, email and role of logged in user
        cursor.execute("SELECT name , email ,role , TempPassFlag FROM PD.Account where username = '" +  uname + "' ")
        data = cursor.fetchone()
        dbname = data[0]
        dbemail = data[1]
        dbrole = data[2]
        dbtemppassflg = data[3]
        # chech the role of the logged in user to determian his/her profile page
        if dbrole == 'medical specialist':
            page = 'MSprofile.html'
        elif dbrole == 'admin':
            page = 'admin.html'
        elif dbrole == 'registered user':
            page = 'RUprofile.html'
        # fetch the profile info of the logged in pateint record: the gender and birth date
            q = "SELECT Gender , BirthDate FROM PD.Patient where NationalID = '" +  uname + "' "
            cursor.execute(q)
            data = cursor.fetchone()
            dbgender= str(data[0])
            dbbdate = data[1]
    finally:
        # close the Database connection
        cursor.close()
        conn.close()
    
    if dbrole == 'registered user': # check if the user is a registered user
        if 'passerror' in request.args: # check if a password  update error exist
            # get the password update error flag value
            passerror = request.args.get('passerror')
            #render the user profile template with the profile info and password update error flag
            return render_template(page, passerror = passerror , name=dbname , email = dbemail, gender = dbgender, bdate= dbbdate   )
        elif 'emailerror' in request.args: # check if a email updete error exist
            # get the email error flag value
            emailerror = request.args.get('emailerror')
            # get the entered old and new email values 
            if 'olde' in request.args:
                olde =request.args.get('olde')
                newe =request.args.get('newe')
                #render the user profile template with the profile info and the email from old, email update error flag and new email inputted values
                return render_template(page, emailerror = emailerror, name=dbname , email = dbemail , gender = dbgender, bdate= dbbdate ,oldemail = olde,newemail= newe)
            #render the user profile template with the profile info and the email from old, email update error flag and new email inputted values
            #render the user profile template with the profile info and the email update error flag 
            return render_template(page, emailerror = emailerror, name=dbname , email = dbemail , gender = dbgender, bdate= dbbdate)
        elif dbtemppassflg == '1':
            flash("Please update your temporary password", 'error')
            return render_template(page, passerror = "True" , name=dbname , email = dbemail, gender = dbgender, bdate= dbbdate   )
        else: 
            #render the user profile template with the profile info only
            return render_template(page , name=dbname , email = dbemail , gender = dbgender, bdate= dbbdate)
    else:
        
        if 'passerror' in request.args:
            # get the password update error flag value
            passerror = request.args.get('passerror')
            #render the user profile template with the profile info and password update error flag
            return render_template(page, passerror = passerror , name=dbname , email = dbemail) 
    
        elif 'emailerror' in request.args:
            # get the email error flag value
            emailerror = request.args.get('emailerror')
            # get the entered old and new email values
            if 'olde' in request.args:
                olde =request.args.get('olde')
                newe =request.args.get('newe')
                #render the user profile template with the profile info and the email from old, email update error flag and new email inputted values
                return render_template(page, emailerror = emailerror, name=dbname , email = dbemail ,oldemail = olde,newemail= newe )
            #render the user profile template with the profile info and the email from  update error flag 
            return render_template(page, emailerror = emailerror, name=dbname , email = dbemail )
        elif dbtemppassflg == '1':
            flash("Please update your temporary password", 'error')
            return render_template(page, passerror = "True" , name=dbname , email = dbemail   )
        else:
            #render the user profile template with the profile info only
            return render_template(page , name=dbname , email = dbemail)

@app.route('/passwordchange', methods=['POST']) # this route handles password change requests
def changepassword():
    #if 'username' in session:
    uname = session['username']
    #else:
    #    uname = "1089888711"

    #uname  = "mosalah" #**get it form the session**
    #uname = "1083375777"
    #uname ="saahmad"
    #takes the inputted old password and new password from the change password form
    frmoldpass = request.form['oldpassword']
    frmnewpass = request.form['newpassword']
    try:
        # get the Database connection 
        conn , cursor = connection()
        # fetch the hashed password, email and name of the logged in user account
        q = "SELECT Password, Email ,Name FROM PD.Account where username = '" +  uname + "' "
        cursor.execute(q)
        data = cursor.fetchone()
        dboldpass = data[0]
        dbemail = data[1]
        dbname = data[2]
        if not(sha256_crypt.verify(frmoldpass, dboldpass)): #check if the old password match the hashed password of the user account
            flash("Old password is incorrect", 'error')
        elif (frmoldpass == frmnewpass): #check if the new password match the old password of the user account
            flash("Old password matches the new password", 'error')
        else: # if not any of the above then update user accout password with the hashed new password
            # hash the new password
            hashedpassword = sha256_crypt.hash(frmnewpass)
            q = "UPDATE PD.Account SET Password = '" +  hashedpassword + "',TempPassFlag = '0', TempPassDate = NULL  WHERE username = '" +  uname + "' "
            cursor.execute(q)
            conn.commit()
            #split the user name to first and last name
            FLname = dbname.split(" ")
            # render the change_password_email html template with the user first name   
            html = render_template('change_password_email.html', user=FLname[0])
            # send the email to user with the change_password_email html template
            msg = email("Changed password",dbemail,   html )
            mail.send(msg)
            flash("Password is successfully changed.",'info')
            # return with no errors and update confirmation message
            return redirect(url_for('profile' , passerror = "False"))
            
    #except MySQL.Error as error:
     #   print(error)
 
    finally:
        # close the Database connection
        cursor.close()
        conn.close()
    
    # return with the error, the error message and the entered values 
    return redirect(url_for('profile' , passerror = "True"))

@app.route('/emailchange' , methods=['POST']) # this route handles email change requests
def changeemail():
    #if 'username' in session:
    uname = session['username']
    #else:
    #    uname = "1089888711"
    #uname  = "mosalah" #**get it form the session**
    #uname = "1083375777"
    #uname ="saahmad"
    #takes the inputted old email and new email  from the change email form
    frmoldemail = request.form['oldemail'].lower()
    frmnewemail = request.form['newemail'].lower()  
    try:
        # get the Database connection 
        conn , cursor = connection()
        # fetch the email and name of the logged in user account
        q = "SELECT Email ,Name FROM PD.Account where username = '" +  uname + "' "
        cursor.execute(q)
        data = cursor.fetchone()
        dbemail = data[0]
        dbname = data[1]
        #check if the new email is exist in anthor user account
        q = "SELECT Email  FROM PD.Account where Email = '" +  frmnewemail + "' "
        cursor.execute(q)
        data = cursor.fetchone()
        if frmoldemail != dbemail: #check if the old email match the email of the user account
            flash("Old email is incorrect", 'error')
        elif frmoldemail == frmnewemail: #check if the new email match the old email of the user account
            flash("New email matches the old email", 'error')
        elif cursor.rowcount > 0 : #check if the new email is exist in anthor user account
            flash("New email already exists in another account", 'error')
        else: # if not any of the above then update email but keep it unvalidated and send an emial confirmation email to the user new email also set the EmailConfirmationSentOn with current timestamp
            q = "UPDATE PD.Account SET Email = '" +  frmnewemail + "' , EmailConfirmationSentOn = '" +  str(datetime.now()) + "' ,EmailConfirmed = '0' WHERE username = '" +  uname + "' "
            cursor.execute(q)
            conn.commit()
            # send an emial confirmation email to the user new email 
            send_confirmation_email(frmnewemail,dbname,'change_email_confirmation.html' )
            flash("Email updated, please confirm your new email address (link sent to new email).",'info')
            # return with no errors and update confirmation message
            return redirect(url_for('profile' , emailerror = "False"))
    finally:
        # close the Database connection 
        cursor.close()
        conn.close()
    # return with the error, the error message and the entered values 
    return redirect(url_for('profile' , emailerror = "True", olde=frmoldemail ,newe=frmnewemail )   )


            

@app.route('/profileupdate' , methods=['POST']) # this route handles the registered user profie update requests
def updeteprofile():
    uname = session['username']
    #uname = "1083375777" #**get it form the session**
    # get the user profile info from the profile form
    frmname = request.form['profile-name']
    frmgender = request.form['profile-gender']
    frmbdate = request.form['birth-date']
    try:
        # get the Database connection 
        conn , cursor = connection()
        # update the user patient record with new profile info name, gender and birth date
        q = "UPDATE PD.Patient SET name = '" + frmname  + "' , Gender  = '" +  frmgender + "' , BirthDate= '"+ frmbdate +"' WHERE  NationalID= '" +  uname + "' "
        cursor.execute(q)
        conn.commit()
        # update the user account record with new name 
        q = "UPDATE PD.Account SET name = '" + frmname  + "'  WHERE username = '" +  uname + "' "
        cursor.execute(q)
        conn.commit() 
        flash("Profile successfully updated.",'info')
    finally:
        # close the Database connection 
        cursor.close()
        conn.close()
    # return with update confirmation message
    return redirect(url_for('profile'))


@app.route('/manageUser')
@login_required
def displayUsers():
    if session['role'] == 'admin':
        try:
            conn , cur = connection()
            cur.execute('select username, name, role, email from account')
            usersList = cur.fetchall()
        finally:
            # close the Database connection
            cur.close()
            conn.close()
        if 'removeerror' in request.args:
            removeerror = request.args.get('removeerror')
            return render_template('adminManageUser.html', usersList = usersList, registeredName = "Khawla", removeerror = "False")
        if 'adderror' in request.args:
            adderror = request.args.get('adderror')
            if 'name' in request.args:
                name =request.args.get('name')
                email =request.args.get('email')
                role =request.args.get('role')
                return render_template('adminManageUser.html', usersList = usersList, registeredName = "Khawla" ,adderror = "True",name=name ,email=email, role=  role) 
            return render_template('adminManageUser.html', usersList = usersList, registeredName = "Khawla", adderror = "False")
        return render_template('adminManageUser.html', usersList = usersList, registeredName = "Khawla")
    else:
        return render_template("404.html")


    
@app.route('/removeUser', methods=['POST'])
def removeUser():
    try:
        conn , cur = connection()
        Users = request.form.getlist("Users")
        for user in Users:
            q = "SELECT Email ,Name FROM PD.Account where username = '" +  user + "' "
            cur.execute(q)
            data = cur.fetchone()
            dbemail = data[0]
            dbname = data[1]
            q = "DELETE FROM account WHERE username = '"+  user +"'"
            cur.execute(q)
            conn.commit()
            FLname = dbname.split(" ")
            # render the Deleteuser_email html template with the user first name
            html = render_template('Deleteuser_email.html', user=FLname[0])
            # send the email to user with the Deleteuser_email html template
            msg = email("Account Deleted",dbemail,   html )
            mail.send(msg)
            #print(user)
    finally:
        cur.close()
        conn.close()
    flash("User(s) successfully removed", 'info')
    return redirect(url_for('displayUsers', removeerror = "False"))
    
@app.route('/addUser', methods=['POST'])
def addUser():
    #print('h')
    try:
        conn , cur = connection()
        #if request.method == 'POST':
        newName = request.form['inputName']
        newEmail = request.form['inputEmail']
        newRole = request.form.get('roleSelect')
        tempPass = password_gen()
        username = username_gen(newName.lower())
        hashedTempPass = sha256_crypt.hash(tempPass)
        q = "SELECT Email  FROM PD.Account where Email = '" +  newEmail + "' "
        cur.execute(q)
        reteunedrow = cur.rowcount
    finally:
        cur.close()
        conn.close()

    if reteunedrow > 0 : #check if the new email is exist in anthor user account
        flash("New email already exists in another account", 'error')
    else:
        try:
            conn , cur = connection()
            q ="INSERT INTO `PD`.`Account` (`username`, `Name`, `Email`, `TempPassFlag`, `Password`, `Role`, `TempPassDate`) VALUES ('"+username+"', '"+newName+"', '"+newEmail+"', '1', '"+hashedTempPass +"', '"+newRole+"','"+str(date.today())+"')"
            cur.execute(q)
            conn.commit()

        
            FLname = newName.split(" ")
            # render the Deleteuser_email html template with the user first name
            html = render_template('adduser_email.html', user=FLname[0] , username =username,password= tempPass  )
            # send the email to user with the Deleteuser_email html template
            msg = email("Account Created",newEmail,   html )
            mail.send(msg)
            flash("User successfully added.",'info')
            #print(newName, newEmail, newRole, tempPass, username)
        finally:
            cur.close()
            conn.close()
        return redirect(url_for('displayUsers', adderror = "False"))
    
    return redirect(url_for('displayUsers', adderror = "True", name=newName ,email=newEmail, role=  newRole ))

def password_gen():
    alphabet = string.ascii_letters 
    digits = string.digits
    passwordalphabet = ''.join([choice(alphabet) for _ in range(5)]) 
    passworddigits = ''.join([choice(digits) for _ in range(3)])
    unshuffledpassword = passwordalphabet+passworddigits
    #print(unshuffledpassword)
    l = list(unshuffledpassword)
    random.shuffle(l)
    shuffledpassword = ''.join(l)
    #print(shuffledpassword)
    return shuffledpassword


def username_gen(name):
    FLname = name.split(" ")
    #print(Usernames[2])
    Username = FLname[0]
    UsernamesList = []
    if len(FLname) > 1  :
        Username += FLname[1]
    #print(Username)
    conn , cur = connection()
    cur.execute("select username from account where username LIKE '"+ Username +"%'")
    UsernamesList = [item[0] for item in cur.fetchall()]   
    #print(UsernamesList)
    for x in range(0, 1001):
        if x == 0:
            username = Username
        else:    
            username=Username+str(x)
        if username in UsernamesList:
            continue
        else:    
            User = username
            break
        
    return User

