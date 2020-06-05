from flask_wtf import FlaskForm
from numpy import unicode
from wtforms import StringField, FloatField, SubmitField, SelectField, DecimalField
from wtforms.validators import DataRequired

class DiamondForm(FlaskForm):
    carat = DecimalField('Carat', validators=[DataRequired()])

    cut = SelectField('Cut', validators=[DataRequired()], choices=[('Fair', 'Fair'), ('Good',
                                                        'Good'), ('Very Good', 'Very Good'),
                                                        ('Premium', 'Premium'), ('Ideal', 'Ideal')]
                      )

    color = SelectField('Color', validators=[DataRequired()], choices=[('D', 'D'), ('E', 'E'),
                                                                       ('F', 'F'), ('G', 'G'),
                                                                       ('H', 'H'), ('I', 'I'),
                                                                       ('J', 'J')])

    clarity = SelectField('Clarity', validators=[DataRequired()], choices=[('I1', 'I1'), ('SI2',
                                                                        'SI2'), ('SI1', 'SI1'),
                                                                           ('VS2', 'VS1'),
                                                                           ('VS1', 'VS1'),
                                                                           ('WS2', 'WS2'),
                                                                           ('WS1', 'WS1')])

    depth = DecimalField('Depth', validators=[DataRequired()])
    table = DecimalField('Table', validators=[DataRequired()])
    x = DecimalField('X', validators=[DataRequired()])
    y = DecimalField('Y', validators=[DataRequired()])
    z = DecimalField('Z', validators=[DataRequired()])

    submit = SubmitField('Predict')
