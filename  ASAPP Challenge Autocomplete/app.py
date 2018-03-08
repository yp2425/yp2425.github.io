from Buildform import InputForm
from flask import Flask, render_template, request, jsonify
from autocomplete_HMM import Autocomplete

app = Flask(__name__)

ac=Autocomplete(model_path = "ngram",
                  n_model=5,
                  n_candidates=5,
                  match_model="start",
                  min_freq=0,
                  punctuations="",
                  lowercase=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        result = ac.predictions(form.Text.data)
    else:
        result = None
    return render_template('view.html', form=form, result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
