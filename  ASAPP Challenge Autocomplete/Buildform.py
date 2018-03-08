from wtforms import Form, FloatField, validators,StringField

class InputForm(Form):
    Text = StringField(validators=[validators.InputRequired()])
    