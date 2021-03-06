from flask import Flask, render_template, url_for, flash, redirect, request
from forms import DiamondForm
app = Flask(__name__)
import diamond

app.config['SECRET_KEY'] = '3c566a4460dc4e71703b9cff8ea80c79'

# dummy data
posts = [
    {
        'carat': 1,
        'cut': 'Ideal',
        'color': 'C',
        'clarity': 'C',
        'depth': 30,
        'table': 20,
        'x': 12,
        'y': 12,
        'z': 11,
    },
    {

        'carat': 2,
        'cut': 'Good',
        'color': 'C',
        'clarity': 'C',
        'depth': 10,
        'table': 10,
        'x': 10,
        'y': 8,
        'z': 11,
    }
]

@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = DiamondForm()
    if form.validate_on_submit():
        price_prediction = diamond.output_prediction(form.carat.data, form.cut.data,
                                                     form.color.data, form.clarity.data,
                                                     form.depth.data, form.table.data,
                                                     form.x.data, form.y.data, form.z.data)

        flash(f'Predicting the price of your {form.carat.data} carat diamond', 'success')
        return render_template('results.html', price_prediction=price_prediction)
    return render_template('home.html', form=form)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results')
def results():
    return render_template('results.html', 'carat', 'cut', 'color', 'clarity', 'depth', 'table',
                           'x', 'y', 'z')


if __name__ == '__main__':
    app.run(debug=True)
