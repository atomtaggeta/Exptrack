from flask import Flask, render_template, url_for, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SECRET_KEY'] = 'mysecretkey'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    expenses = db.relationship('Expense', backref='owner', lazy=True)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return '<Expense %r>' % self.id

# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect("/")
    
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect("/")
        else:
            flash("Login failed. Check your credentials and try again.", "danger")
            
    return render_template("login.html")

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect("/login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        new_user = User(username=username, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Account created succesfully! Please login.", "success")
            return redirect("/login")
        except:
            flash("An error occured. Please try again.", "danger")
    return render_template("register.html")


@app.route('/', methods=['POST', 'GET'])
@login_required
def index():
    if request.method == 'POST':
        expense_content = request.form['content']
        expense_category = request.form['category']
        expense_amount = float(request.form['amount'])
        new_expense = Expense(content=expense_content, amount=expense_amount, category=expense_category, user_id=current_user.id)
        
        try:
            db.session.add(new_expense)
            db.session.commit()
            return redirect('/')
        except:
            return 'There is an issue adding your expense'
        
    else:
        # Show expenses for logged-in user
        expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date_created).all()
        return render_template('index.html', expenses=expenses)

@app.route('/delete/<int:id>')
@login_required
def delete(id):
    expense_to_delete = Expense.query.get_or_404(id)
    # Ensure the user can only delete their own expenses
    if expense_to_delete.user_id != current_user.id:
        return "You are not authorized to delete this expense."
    
    try:
        db.session.delete(expense_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that expense'

@app.route('/update/<int:id>', methods=['GET', 'POST'])
@login_required
def update(id):
    
    expense = Expense.query.get_or_404(id)
    if expense.user_id != current_user.id:
        return "You are not authorized to update this expense."
    
    if request.method == 'POST':
        expense.content = request.form['content']
        expense.category = request.form['category']
        expense.amount = float(request.form['amount'])
        
        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue updating your expense'
    else:
        return render_template('update.html', expense=expense)


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)