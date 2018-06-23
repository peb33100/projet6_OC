

from flask import Flask, request, make_response, redirect, render_template
import auxiliary as aux
import pickle

app = Flask(__name__)


infile = open(f"tf_idf_transformation.pyc",'rb')
dico = pickle.load(infile)
infile.close()

infile = open(f"log_reg_ovr_model.pyc",'rb')
model = pickle.load(infile)
infile.close()

infile = open(f"multi_label.pyc",'rb')
label_conversion = pickle.load(infile)
infile.close()

@app.route('/essai')
def essai():
    return " Ceci est un essai"
	
	
@app.route('/tagging',methods=['GET', 'POST'])
def prediction():
	if (request.method=='POST'):
	
		if(request.form['methode']=="Supervisée"):
			a = list()
			input_question = request.form['titre'] + " " + request.form['contenu']
			a.append(input_question)
			answer  = ", ".join(aux.return_tags_from_question(a, dico, model, label_conversion)[0])
			return answer
		
		## Logique du programme
				
	
	return render_template('accueil.html')

@app.errorhandler(404)
def ma_page_404(error):
	response = make_response("le chemin d'accès à l'application de tagging est /tagging", 404)
	return response
	
@app.route('/google')
def redirection_google():
    return redirect('http://www.google.fr')

	
	
if __name__ == "__main__":
	app.run(debug=True)