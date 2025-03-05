from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from flask import Flask, render_template, url_for, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from flask_bcrypt import Bcrypt


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SECRET_KEY'] = 'mysecretkey'
matplotlib.use('Agg')

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
        
        # Check if we have enough data to train the model
        all_expenses = Expense.query.all()
        training_count = len(all_expenses)
        model_trained = training_count >= 10
        
        return render_template('index.html', 
                              expenses=expenses, 
                              model_trained=model_trained,
                              training_count=training_count)
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

@app.route("/analytics")
@login_required
def analytics():
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    
    expense_list = []
    for expense in expenses:
        expense_list.append({
            'description': expense.content,
            'category': expense.category,
            'amount': expense.amount,
            'date': expense.date_created
        })

    total_expenses = len(expense_list)
    if total_expenses == 0:
        return render_template("analytics.html",
                               has_expenses = False,
                               message = "Add some expenses to see analysis")
        
    df = pd.DataFrame(expense_list)
    
    total_spent = df['amount'].sum()
    
    highest_expense = df.loc[df['amount'].idxmax()]

    category_totals = df.groupby('category')['amount'].sum().to_dict()
    
    most_expensive_category = max(category_totals.items(), key=lambda x: x[1])
    
    return render_template("analytics.html",
                           has_expenses=True,
                           total_expenses=total_expenses,
                           total_spent=round(total_spent, 2),
                           highest_expense={
                               'description': highest_expense['description'],
                               'amount': highest_expense['amount'],
                               'category': highest_expense['category']
                           },
                           category_totals = category_totals,
                           most_expensive_category = most_expensive_category)

def train_category_model():
    expenses = Expense.query.all()
    
    
    if len(expenses) < 10:
        return None, None
    
    # extract descriptions and categories
    descriptions = [expense.content for expense in expenses]
    categories = [expense.category for expense in expenses]
    
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    
    model = MultinomialNB()
    model.fit(X, categories)
    
    return vectorizer, model

@app.route("/suggest-category")
@login_required
def suggest_category():
    description = request.args.get('description', '')
    
    if not description:
        return {'success': False, 'message': 'No description provided'}
    
    vectorizer, model = train_category_model()
    
    if not vectorizer or not model:
        return {'success': False, 'message': 'Not enough data to make suggestions'}
    
    X = vectorizer.transform([description])
    
    predicted_category = model.predict(X)[0]
    
    probabilities = model.predict_proba(X)[0]
    confidence = round(max(probabilities) * 100, 2)
    
    return {
        'success': True,
        'suggested_category': predicted_category,
        'confidence': confidence
    }

@app.route("/train_model", methods=["POST"])
@login_required
def train_model():
    expense_description = request.form["expense_description"]
    expense_category = request.form["expense_category"]
    
    new_training_example = Expense(
        content = expense_description,
        category = expense_category,
        amount = 0.00,
        user_id = current_user.id
    )
    
    try:
        db.session.add(new_training_example)
        db.session.commit()
        flash("Training example added successfully!", "success")
    except:
        flash("There was an issue adding your training example", "danger")
    
    return redirect("/")

# @app.route("/model_stats")
# @login_required
# def model_stats():
#     expenses = Expense.query.all()
#     training_count = len(expenses)
#     model_exists = training_count >= 10
    
#     category_distribution = {}
    
#     for expense in expenses:
#         if expense.category in category_distribution:
#             category_distribution[expense.category] += 1
#         else:
#             category_distribution[expense.category] = 1
    
#     return render_template("model_stats.html",
#                            model_exists=model_exists,
#                            training_count=training_count,
#                            category_distribution=category_distribution)

@app.route("/predict_category", methods=["POST"])
@login_required
def predict_category():
    test_description = request.form["test_description"]
    amount = request.form.get("amount", 0.00)
    
    try:
        amount = float(amount)
    except:
        flash("Invalid amount value", "danger")
        return redirect("/")
    
    vectorizer, model = train_category_model()
    
    if not vectorizer or not model:
        flash("Not enough training data to make predictions.", "danger")
        return redirect("/")
    
    X = vectorizer.transform([test_description])
    predicted_category = model.predict(X)[0]
    
    probabilities = model.predict_proba(X)[0]
    confidence = round(max(probabilities)*100, 2)
    
    # Create a new expense with the predicted category
    new_expense = Expense(
        content=test_description,
        category=predicted_category,
        amount=amount,
        user_id=current_user.id
    )
    
    try:
        db.session.add(new_expense)
        db.session.commit()
        flash(f"Added expense with AI-predicted category: {predicted_category} (Confidence: {confidence}%)", "success")
    except:
        flash("There was an issue adding your expense", "danger")
    
    return redirect("/")

def create_category_chart(category_distribution):
    plt.figure(figsize=(8, 6))
    categories = list(category_distribution.keys())
    counts = list(category_distribution.values())
    
    # Generate colors based on the number of categories
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(categories)))
    
    plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Category Distribution in Training Data')
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Convert to base64 string for embedding in HTML
    image_png = buffer.getvalue()
    buffer.close()
    
    chart = base64.b64encode(image_png).decode('utf-8')
    return chart

def create_confidence_chart(vectorizer, model):
    if not vectorizer or not model:
        return None
        
    # Sample words analysis
    try:
        # Get feature names (words) from the vectorizer
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()
            
        # Get model coefficients
        # For MultinomialNB, higher coefficient means stronger predictor for a class
        feature_importances = model.feature_log_prob_
        
        # The classes (categories)
        categories = model.classes_
        
        # Create a visualization of top informative words per category
        plt.figure(figsize=(10, 8))
        
        # Limit to at most 5 categories to keep the chart readable
        max_categories = min(5, len(categories))
        
        # Create subplots based on number of categories
        fig, axes = plt.subplots(max_categories, 1, figsize=(10, max_categories * 3))
        
        # Make axes iterable even if there's only one category
        if max_categories == 1:
            axes = [axes]
        
        for i, category in enumerate(categories[:max_categories]):
            if i >= max_categories:
                break
                
            # Get log probabilities for this category
            category_index = np.where(model.classes_ == category)[0][0]
            log_probs = feature_importances[category_index]
            
            # Sort by importance
            sorted_indices = np.argsort(log_probs)
            
            # Get top 10 most informative words
            top_indices = sorted_indices[-10:]
            top_words = [feature_names[idx] for idx in top_indices]
            top_values = [log_probs[idx] for idx in top_indices]
            
            # Plot horizontal bar chart
            ax = axes[i]
            ax.barh(top_words, top_values, color='skyblue')
            ax.set_title(f'Top words for category: {category}')
            ax.set_xlabel('Log Probability')
        
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Convert to base64 string for embedding in HTML
        image_png = buffer.getvalue()
        buffer.close()
        
        chart = base64.b64encode(image_png).decode('utf-8')
        return chart
    except Exception as e:
        print(f"Error creating confidence chart: {e}")
        return None

@app.route("/model_stats")
@login_required
def model_stats():
    expenses = Expense.query.all()
    training_count = len(expenses)
    model_exists = training_count >= 10
    
    category_distribution = {}
    
    for expense in expenses:
        if expense.category in category_distribution:
            category_distribution[expense.category] += 1
        else:
            category_distribution[expense.category] = 1
    
    # Generate charts using matplotlib
    category_chart = create_category_chart(category_distribution) if category_distribution else None
    
    # If model exists, create visualization of word importance
    vectorizer, model = train_category_model()
    confidence_chart = create_confidence_chart(vectorizer, model) if model_exists else None
    
    return render_template("model_stats.html",
                           model_exists=model_exists,
                           training_count=training_count,
                           category_distribution=category_distribution,
                           category_chart=category_chart,
                           confidence_chart=confidence_chart)

# Add an additional route to show prediction confidence visualization
@app.route("/visualize_prediction", methods=["POST"])
@login_required
def visualize_prediction():
    test_description = request.form["test_description"]
    
    vectorizer, model = train_category_model()
    
    if not vectorizer or not model:
        flash("Not enough training data to make predictions.", "danger")
        return redirect("/")
    
    # Transform the description
    X = vectorizer.transform([test_description])
    
    # Get class predictions and probabilities
    predicted_category = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Match probabilities with categories
    categories = model.classes_
    prob_by_category = {cat: prob for cat, prob in zip(categories, probabilities)}
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(categories, probabilities, color='skyblue')
    plt.title('Category Prediction Confidence')
    plt.xlabel('Category')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Highlight the predicted category
    pred_index = np.where(categories == predicted_category)[0][0]
    plt.bar(pred_index, probabilities[pred_index], color='green')
    
    # Add percentage labels
    for i, prob in enumerate(probabilities):
        plt.text(i, prob + 0.01, f'{prob:.2%}', ha='center')
    
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Convert to base64 string for embedding in HTML
    image_png = buffer.getvalue()
    buffer.close()
    
    prediction_chart = base64.b64encode(image_png).decode('utf-8')
    
    return render_template("prediction_visualization.html",
                          test_description=test_description,
                          predicted_category=predicted_category,
                          confidence=max(probabilities) * 100,
                          prediction_chart=prediction_chart)

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)