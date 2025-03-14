from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
import csv
import pdfkit
import base64
from datetime import datetime, timedelta
from flask import Flask, render_template, url_for, request, redirect, flash, send_file, make_response
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

class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    theme = db.Column(db.String(20), default='light')
    date_format = db.Column(db.String(20), default='MM/DD/YYYY')
    start_page = db.Column(db.String(20), default='expenses')
    currency = db.Column(db.String(3), default='USD')
    currency_position = db.Column(db.String(10), default='before')
    decimal_separator = db.Column(db.String(1), default='.')
    
    # Relationship with User
    user = db.relationship('User', backref='settings', uselist=False)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), nullable=True)
    password = db.Column(db.String(150), nullable=False)
    date_registered = db.Column(db.DateTime, default=datetime.utcnow)
    expenses = db.relationship('Expense', backref='owner', lazy=True)
    training_examples = db.relationship('TrainingExample', backref='owner', lazy=True)
    user_settings = db.relationship('UserSettings', backref='owner', uselist=False)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return '<Expense %r>' % self.id
class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    icon = db.Column(db.String(50), default='tag')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationship with User
    user = db.relationship('User', backref='categories')
    
    def __repr__(self):
        return f'<Category {self.name}>'
class TrainingExample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return '<TrainingExample %r>' % self.id

def format_date(date, format_string='MM/DD/YYYY'):
    """Format a date according to user preferences"""
    if format_string == 'MM/DD/YYYY':
        return date.strftime('%m/%d/%Y')
    elif format_string == 'DD/MM/YYYY':
        return date.strftime('%d/%m/%Y')
    elif format_string == 'YYYY-MM-DD':
        return date.strftime('%Y-%m-%d')
    return date.strftime('%m/%d/%Y')  # Default format


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

# Login manager unauthorized handler
@login_manager.unauthorized_handler
def unauthorized():
    flash("Please log in to access this page.", "info")
    return redirect(url_for("login"))
@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect("/login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form.get("email", "")
        password = request.form["password"]
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        new_user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully! Please login.", "success")
            return redirect("/login")
        except:
            flash("An error occurred. Please try again.", "danger")
    return render_template("register.html")


@app.route('/', methods=['POST', 'GET'])
@login_required
def index():
    if request.method == 'POST':
        expense_content = request.form['content']
        expense_category = request.form['category']
        expense_amount = float(request.form['amount'])
        
        # Get the date from the form
        expense_date_str = request.form.get('date', '')
        
        if expense_date_str:
            # Parse the date string into a datetime object
            try:
                expense_date = datetime.strptime(expense_date_str, '%Y-%m-%d')
                # Set the time to current time
                current_time = datetime.now().time()
                expense_date = datetime.combine(expense_date.date(), current_time)
            except ValueError:
                # If date parsing fails, use current datetime
                expense_date = datetime.now()
        else:
            # If no date provided, use current datetime
            expense_date = datetime.now()
        
        # Create the expense with the specified date
        new_expense = Expense(
            content=expense_content, 
            amount=expense_amount, 
            category=expense_category, 
            date_created=expense_date,
            user_id=current_user.id
        )
        
        try:
            db.session.add(new_expense)
            db.session.commit()
            return redirect('/')
        except:
            return 'There is an issue adding your expense'
        
    else:
        # Show expenses for logged-in user
        expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date_created.desc()).all()
        
        # Check if we have enough data to train the model
        # Count both expenses and training examples
        expenses_count = Expense.query.filter_by(user_id=current_user.id).count()
        training_examples_count = TrainingExample.query.filter_by(user_id=current_user.id).count()
        training_count = expenses_count + training_examples_count
        model_trained = training_count >= 10
        
        # Get user settings for theme
        user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        
        # Get default categories
        default_categories = [
            "Food", "Transportation", "Entertainment", "Bills", 
            "Health", "Education", "Shopping", "Rent", 
            "Utilities", "Insurance", "Gifts", "Travel", 
            "Subscriptions", "Taxes", "Loans", "Entertainment/Leisure", 
            "Sports", "Other"
        ]
        
        # Get user's custom categories
        custom_categories = [cat.name for cat in Category.query.filter_by(user_id=current_user.id).all()]
        
        # Combine all categories, removing duplicates
        all_categories = sorted(set(default_categories + custom_categories))
        
        return render_template('index.html', 
                              expenses=expenses, 
                              model_trained=model_trained,
                              training_count=training_count,
                              user_settings=user_settings,
                              categories=all_categories)

@app.context_processor
def inject_template_helpers():
    """Inject utility functions and user settings into all templates"""
    utilities = {}
    
    # Get user settings if authenticated
    if current_user.is_authenticated:
        user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        if not user_settings:
            user_settings = UserSettings(user_id=current_user.id)
            db.session.add(user_settings)
            db.session.commit()
        utilities['user_settings'] = user_settings
    else:
        utilities['user_settings'] = None
    
    # Date formatter
    def format_date_template(date, user_settings=None):
        if user_settings:
            return format_date(date, user_settings.date_format)
        return date.strftime('%m/%d/%Y')  # Default format
    
    # Add now() and timedelta to the template context
    utilities['now'] = datetime.now
    utilities['timedelta'] = timedelta
    
    utilities['format_currency'] = format_currency
    utilities['format_date'] = format_date_template
    
    return utilities

# Currency formatter
def format_currency(amount, user_settings=None):
    """Format currency values according to user preferences"""
    # Default values
    currency = 'USD'
    position = 'before'
    decimal_separator = '.'
    
    # If user_settings is a dictionary
    if isinstance(user_settings, dict):
        currency = user_settings.get('currency', currency)
        position = user_settings.get('currency_position', position)
        decimal_separator = user_settings.get('decimal_separator', decimal_separator)
    # If user_settings is a UserSettings object
    elif user_settings and hasattr(user_settings, 'currency'):
        currency = user_settings.currency
        position = user_settings.currency_position
        decimal_separator = user_settings.decimal_separator
    
    # Round to 2 decimal places
    amount = round(float(amount), 2)
    
    # Format the number with the appropriate decimal separator
    if decimal_separator == ',':
        formatted_amount = f"{int(amount)},{amount % 1:.2f}".replace('0,', '').replace('.', '')
    else:
        formatted_amount = f"{amount:.2f}"
    
    # Currency symbols
    symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CAD': 'C$',
        'AUD': 'A$',
        'INR': '₹',
        'CNY': '¥'
    }
    
    symbol = symbols.get(currency, '$')
    
    # Apply the position
    if position == 'before':
        return f"{symbol}{formatted_amount}"
    else:
        return f"{formatted_amount}{symbol}"

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
        
        # Process date if provided
        expense_date_str = request.form.get('date', '')
        if expense_date_str:
            try:
                expense_date = datetime.strptime(expense_date_str, '%Y-%m-%d')
                # Keep the original time but update the date
                original_time = expense.date_created.time()
                expense.date_created = datetime.combine(expense_date.date(), original_time)
            except ValueError:
                # If date parsing fails, keep the original date
                pass
        
        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue updating your expense'
    else:
        return render_template('update.html', expense=expense)

@app.route("/delete_category/<int:id>")
@login_required
def delete_category(id):
    category = Category.query.get_or_404(id)
    
    # Make sure the user can only delete their own categories
    if category.user_id != current_user.id:
        flash("You are not authorized to delete this category.", "danger")
        return redirect("/settings")
    
    try:
        # Get the name for the success message
        category_name = category.name
        
        db.session.delete(category)
        db.session.commit()
        flash(f"Category '{category_name}' deleted successfully!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting category: {str(e)}", "danger")
    
    return redirect("/settings#categories")

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

def train_category_model(user_id):
    """
    Train a category prediction model based on user's expense data.
    Combines both regular expenses and specific training examples.
    
    Args:
        user_id (int): The ID of the user to train the model for
        
    Returns:
        tuple: (vectorizer, model) or (None, None) if insufficient data
    """
    # Get expenses for the current user
    expenses = Expense.query.filter_by(user_id=user_id).all()
    
    # Get training examples for the current user
    training_examples = TrainingExample.query.filter_by(user_id=user_id).all()
    
    # Extract descriptions and categories from expenses
    expense_descriptions = [expense.content for expense in expenses]
    expense_categories = [expense.category for expense in expenses]
    
    # Extract descriptions and categories from training examples
    training_descriptions = [example.content for example in training_examples]
    training_categories = [example.category for example in training_examples]
    
    # Combine both sources of data
    all_descriptions = expense_descriptions + training_descriptions
    all_categories = expense_categories + training_categories
    
    # Check if we have enough data to train a meaningful model
    if len(all_descriptions) < 10:
        return None, None
    
    # Create and fit the vectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(all_descriptions)
    
    # Create and fit the model
    model = MultinomialNB()
    model.fit(X, all_categories)
    
    return vectorizer, model

@app.route("/suggest-category")
@login_required
def suggest_category():
    description = request.args.get('description', '')
    
    if not description:
        return {'success': False, 'message': 'No description provided'}
    
    # Pass current user's ID to train model with only their data
    vectorizer, model = train_category_model(current_user.id)
    
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
    
    # Add validation - ensure description isn't empty
    if not expense_description.strip():
        flash("Expense description cannot be empty", "danger")
        return redirect("/")
    
    # Create a new training example (not an expense)
    new_training_example = TrainingExample(
        content = expense_description,
        category = expense_category,
        user_id = current_user.id
    )
    
    try:
        db.session.add(new_training_example)
        db.session.commit()
        flash("Training example added successfully!", "success")
    except:
        flash("There was an issue adding your training example", "danger")
    
    return redirect("/")


@app.route("/predict_category", methods=["POST"])
@login_required
def predict_category():
    test_description = request.form["test_description"]
    amount = request.form.get("amount", 0.00)
    
    # Get the date from the form
    expense_date_str = request.form.get('date', '')
        
    if expense_date_str:
        # Parse the date string into a datetime object
        try:
            expense_date = datetime.strptime(expense_date_str, '%Y-%m-%d')
            # Set the time to current time
            current_time = datetime.now().time()
            expense_date = datetime.combine(expense_date.date(), current_time)
        except ValueError:
            # If date parsing fails, use current datetime
            expense_date = datetime.now()
    else:
        # If no date provided, use current datetime
        expense_date = datetime.now()
    
    try:
        amount = float(amount)
    except:
        flash("Invalid amount value", "danger")
        return redirect("/")
    
    # Pass current user's ID to train model with only their data
    vectorizer, model = train_category_model(current_user.id)
    
    if not vectorizer or not model:
        flash("Not enough training data to make predictions.", "danger")
        return redirect("/")
    
    X = vectorizer.transform([test_description])
    predicted_category = model.predict(X)[0]
    
    probabilities = model.predict_proba(X)[0]
    confidence = round(max(probabilities)*100, 2)
    
    # Create a new expense with the predicted category and specified date
    new_expense = Expense(
        content=test_description,
        category=predicted_category,
        amount=amount,
        date_created=expense_date,
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
    
    # Create a clean pie chart without shadow and with a crisp edge
    plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, 
            shadow=False, wedgeprops={'edgecolor': 'none'})
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Category Distribution in Training Data')
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=False, bbox_inches='tight')
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
    # Get expenses for the current user
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    
    # Get training examples for the current user
    training_examples = TrainingExample.query.filter_by(user_id=current_user.id).all()
    
    # Combined count
    total_training_data = len(expenses) + len(training_examples)
    training_count = total_training_data
    model_exists = training_count >= 10
    
    category_distribution = {}
    
    # Count categories from expenses
    for expense in expenses:
        if expense.category in category_distribution:
            category_distribution[expense.category] += 1
        else:
            category_distribution[expense.category] = 1
    
    # Count categories from training examples
    for example in training_examples:
        if example.category in category_distribution:
            category_distribution[example.category] += 1
        else:
            category_distribution[example.category] = 1
    
    # Generate charts using matplotlib
    category_chart = create_category_chart(category_distribution) if category_distribution else None
    
    # If model exists, create visualization of word importance
    vectorizer, model = train_category_model(current_user.id)
    confidence_chart = create_confidence_chart(vectorizer, model) if model_exists else None
    
    return render_template("model_stats.html",
                          model_exists=model_exists,
                          training_count=training_count,
                          category_distribution=category_distribution,
                          category_chart=category_chart,
                          confidence_chart=confidence_chart)
@app.route("/visualize_prediction", methods=["POST"])
@login_required
def visualize_prediction():
    test_description = request.form["test_description"]
    
    # Pass current user's ID to train model with only their data
    vectorizer, model = train_category_model(current_user.id)
    
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
@app.route("/dashboard")
@login_required
def dashboard():
    # Get user expenses
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    
    # Initialize variables with default values
    total_expenses = 0
    monthly_average = 0
    highest_category = "N/A"
    category_totals = {}
    monthly_spending = {}
    months = []
    monthly_spending_values = []
    category_labels = []
    category_values = []
    
    # Calculate dashboard metrics if there are expenses
    if expenses:
        # Calculate total expenses
        total_expenses = sum(expense.amount for expense in expenses)
        
        # Group expenses by month
        for expense in expenses:
            month = expense.date_created.strftime('%Y-%m')
            if month in monthly_spending:
                monthly_spending[month] += expense.amount
            else:
                monthly_spending[month] = expense.amount
        
        # Calculate the average monthly spending
        monthly_average = total_expenses / len(monthly_spending) if monthly_spending else 0
        
        # Get category distribution for pie chart
        for expense in expenses:
            if expense.category in category_totals:
                category_totals[expense.category] += expense.amount
            else:
                category_totals[expense.category] = expense.amount
        
        # Find highest spending category
        if category_totals:
            highest_category = max(category_totals.items(), key=lambda x: x[1])[0]
        
        # Sort months chronologically and get data for the last 6 months
        sorted_months = sorted(monthly_spending.keys())[-6:] if monthly_spending else []
        
        for month in sorted_months:
            # Convert YYYY-MM to display format (e.g., Jan)
            display_month = datetime.strptime(month, '%Y-%m').strftime('%b')
            months.append(display_month)
            monthly_spending_values.append(monthly_spending[month])
        
        # Get category distribution for the pie chart
        category_labels = list(category_totals.keys())
        category_values = list(category_totals.values())
    
    # Get recent expenses for the table (limit to 5)
    recent_expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date_created.desc()).limit(5).all()
    
    return render_template("dashboard.html",
                           total_expenses=round(total_expenses, 2),
                           monthly_average=round(monthly_average, 2),
                           highest_category=highest_category,
                           expense_count=len(expenses),
                           months=months,
                           monthly_spending=monthly_spending_values,
                           recent_expenses=recent_expenses,
                           category_labels=category_labels,
                           category_values=category_values)

@app.route("/profile")
@login_required
def profile():
    # Get current date for the template
    now = datetime.now
    
    # Get user stats for the Account Statistics section
    expense_count = Expense.query.filter_by(user_id=current_user.id).count()
    total_spent = db.session.query(db.func.sum(Expense.amount)).filter_by(user_id=current_user.id).scalar() or 0
    
    # Get unique categories used by the user
    categories_used = db.session.query(Expense.category).filter_by(user_id=current_user.id).distinct().count()
    
    # Calculate days active (days since registration)
    if current_user.date_registered:
        days_active = (datetime.utcnow() - current_user.date_registered).days
    else:
        days_active = 0
    
    return render_template("profile.html", 
                           now=now,
                           expense_count=expense_count,
                           total_spent=total_spent,
                           categories_used=categories_used,
                           days_active=days_active)

@app.route("/change_password", methods=["POST"])
@login_required
def change_password():
    if request.method == "POST":
        current_password = request.form["current_password"]
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]
        
        # Verify current password
        if not bcrypt.check_password_hash(current_user.password, current_password):
            flash("Current password is incorrect.", "danger")
            return redirect("/profile")
        
        # Verify new passwords match
        if new_password != confirm_password:
            flash("New passwords do not match.", "danger")
            return redirect("/profile")
        
        # Update password
        hashed_password = bcrypt.generate_password_hash(new_password).decode("utf-8")
        current_user.password = hashed_password
        
        try:
            db.session.commit()
            flash("Password changed successfully!", "success")
        except:
            flash("An error occurred while changing your password.", "danger")
            
    return redirect("/profile")

@app.route("/settings")
@login_required
def settings():
    # Get or create user settings
    user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    
    if not user_settings:
        # Create default settings for this user
        user_settings = UserSettings(user_id=current_user.id)
        db.session.add(user_settings)
        db.session.commit()
    
    # Get user's custom categories
    custom_categories = Category.query.filter_by(user_id=current_user.id).all()
    
    # Get default categories that ship with the app
    default_categories = [
        {"name": "Food", "icon": "utensils"},
        {"name": "Transportation", "icon": "car"},
        {"name": "Entertainment", "icon": "film"},
        {"name": "Bills", "icon": "file-invoice"},
        {"name": "Health", "icon": "medkit"},
        {"name": "Education", "icon": "graduation-cap"},
        {"name": "Shopping", "icon": "shopping-bag"},
        {"name": "Rent", "icon": "home"},
        {"name": "Utilities", "icon": "bolt"},
        {"name": "Insurance", "icon": "shield-alt"},
        {"name": "Gifts", "icon": "gift"},
        {"name": "Travel", "icon": "plane"},
        {"name": "Subscriptions", "icon": "calendar-alt"},
        {"name": "Taxes", "icon": "dollar-sign"},
        {"name": "Loans", "icon": "hand-holding-usd"},
        {"name": "Entertainment/Leisure", "icon": "smile"},
        {"name": "Sports", "icon": "running"},
        {"name": "Other", "icon": "tag"}
    ]
    
    # Get all available icons for the category form
    available_icons = [
        {"value": "utensils", "name": "Food (Utensils)"},
        {"value": "car", "name": "Transportation (Car)"},
        {"value": "film", "name": "Entertainment (Film)"},
        {"value": "file-invoice", "name": "Bills (Invoice)"},
        {"value": "medkit", "name": "Health (Medkit)"},
        {"value": "graduation-cap", "name": "Education (Cap)"},
        {"value": "shopping-bag", "name": "Shopping (Bag)"},
        {"value": "home", "name": "Housing (Home)"},
        {"value": "bolt", "name": "Utilities (Bolt)"},
        {"value": "shield-alt", "name": "Insurance (Shield)"},
        {"value": "gift", "name": "Gifts (Gift)"},
        {"value": "plane", "name": "Travel (Plane)"},
        {"value": "calendar-alt", "name": "Subscriptions (Calendar)"},
        {"value": "dollar-sign", "name": "Taxes (Dollar)"},
        {"value": "hand-holding-usd", "name": "Loans (Money)"},
        {"value": "smile", "name": "Entertainment (Smile)"},
        {"value": "running", "name": "Sports (Running)"},
        {"value": "tag", "name": "Other (Tag)"},
        {"value": "coffee", "name": "Coffee"},
        {"value": "pizza-slice", "name": "Food (Pizza)"},
        {"value": "beer", "name": "Drinks (Beer)"},
        {"value": "bus", "name": "Bus"},
        {"value": "subway", "name": "Subway"},
        {"value": "taxi", "name": "Taxi"},
        {"value": "building", "name": "Building"},
        {"value": "tshirt", "name": "Clothing"},
        {"value": "phone", "name": "Phone"},
        {"value": "laptop", "name": "Computer"},
        {"value": "gamepad", "name": "Gaming"},
        {"value": "paw", "name": "Pets"},
        {"value": "baby", "name": "Baby/Children"},
        {"value": "briefcase", "name": "Work"},
        {"value": "tools", "name": "Tools/Repairs"},
        {"value": "book", "name": "Books"},
        {"value": "gym", "name": "Gym"},
    ]
    
    return render_template("settings.html", 
                          settings=user_settings,
                          custom_categories=custom_categories,
                          default_categories=default_categories,
                          available_icons=available_icons)

@app.route("/update_settings", methods=["POST"])
@login_required
def update_settings():
    section = request.form.get("section", "all")
    
    # Get current settings or create if not exists
    user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not user_settings:
        user_settings = UserSettings(user_id=current_user.id)
        db.session.add(user_settings)
    
    if section == "general":
        # Handle general settings only
        user_settings.theme = request.form.get("theme", "light")
        user_settings.date_format = request.form.get("date_format", "MM/DD/YYYY")
        flash("General settings have been updated successfully!", "success")
        
    elif section == "currency":
        # Handle currency settings only
        user_settings.currency = request.form.get("currency", "USD")
        user_settings.currency_position = request.form.get("currency_position", "before")
        user_settings.decimal_separator = request.form.get("decimal_separator", ".")
        flash("Currency settings have been updated successfully!", "success")
    
    elif section == "all":
        # Handle all settings at once
        # General settings
        user_settings.theme = request.form.get("theme", "light")
        user_settings.date_format = request.form.get("date_format", "MM/DD/YYYY")
        
        # Currency settings
        user_settings.currency = request.form.get("currency", "USD")
        user_settings.currency_position = request.form.get("currency_position", "before")
        user_settings.decimal_separator = request.form.get("decimal_separator", ".")
        
        flash("All settings have been updated successfully!", "success")
    
    # Save all changes
    db.session.commit()
    
    # For AJAX requests, return JSON response
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"success": True, "message": "Settings updated successfully"})
    
    # For regular form submissions, redirect back to the settings page
    return redirect("/settings")

@app.route("/update_profile", methods=["POST"])
@login_required
def update_profile():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        
        # Check if username already exists (if it's not the current user)
        existing_user = User.query.filter(User.username == username, User.id != current_user.id).first()
        if existing_user:
            flash("Username already exists. Please choose another.", "danger")
            return redirect("/profile")
        
        # Update user info
        current_user.username = username
        current_user.email = email
        
        try:
            db.session.commit()
            flash("Profile updated successfully!", "success")
        except:
            db.session.rollback()
            flash("An error occurred while updating your profile.", "danger")
            
    return redirect("/profile")

@app.route("/add_category", methods=["POST"])
@login_required
def add_category():
    category_name = request.form.get("category_name")
    category_icon = request.form.get("category_icon", "tag")
    
    # Check if category already exists for this user
    existing_category = Category.query.filter_by(
        name=category_name, 
        user_id=current_user.id
    ).first()
    
    if existing_category:
        flash(f"Category '{category_name}' already exists.", "info")
    else:
        # Create and save the new category
        new_category = Category(
            name=category_name,
            icon=category_icon,
            user_id=current_user.id
        )
        
        try:
            db.session.add(new_category)
            db.session.commit()
            flash(f"Category '{category_name}' added successfully!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred while adding the category: {str(e)}", "danger")
    
    return redirect("/settings")

@app.route("/export/csv")
@login_required
def export_csv():
    # Get user expenses
    expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date_created.desc()).all()
    
    # Get user settings for currency formatting
    user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not user_settings:
        user_settings = UserSettings(user_id=current_user.id)
        db.session.add(user_settings)
        db.session.commit()
    
    # Create a CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['Date', 'Description', 'Category', 'Amount'])
    
    # Write expense data with formatted currency
    for expense in expenses:
        writer.writerow([
            expense.date_created.strftime('%Y-%m-%d'),
            expense.content,
            expense.category,
            format_currency(expense.amount, user_settings)
        ])
    
    # Prepare response
    output.seek(0)
    filename = f"expenses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@app.route("/export/pdf")
@login_required
def export_pdf():
    try:
        # Check if pdfkit/wkhtmltopdf is installed
        import pdfkit
        
        # Get user expenses
        expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date_created.desc()).all()
        
        # Calculate total
        total_spent = sum(expense.amount for expense in expenses)
        
        # Get user settings for currency formatting
        user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
        if not user_settings:
            user_settings = UserSettings(user_id=current_user.id)
            db.session.add(user_settings)
            db.session.commit()
        
        # Create a context with the format_currency function available
        template_context = {
            'expenses': expenses,
            'user': current_user,
            'total_spent': total_spent,
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_settings': user_settings,
            'format_currency': format_currency  # Pass the function to the template
        }
        
        # Render HTML template for PDF
        html = render_template(
            'pdf_template.html',
            **template_context
        )
        
        # Configure pdfkit options
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None
        }
        
        try:
            # First attempt: Try using pdfkit with installed wkhtmltopdf
            pdf = pdfkit.from_string(html, False, options=options)
            
            # Prepare response
            filename = f"expenses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            response = make_response(pdf)
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename={filename}'
            
            return response
            
        except OSError as e:
            # If wkhtmltopdf fails, try WeasyPrint
            try:
                from weasyprint import HTML, CSS
                from weasyprint.text.fonts import FontConfiguration
                
                font_config = FontConfiguration()
                html_obj = HTML(string=html)
                
                # Generate PDF
                pdf_file = io.BytesIO()
                html_obj.write_pdf(pdf_file, font_config=font_config)
                pdf_file.seek(0)
                
                # Prepare response
                filename = f"expenses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                response = make_response(pdf_file.getvalue())
                response.headers['Content-Type'] = 'application/pdf'
                response.headers['Content-Disposition'] = f'attachment; filename={filename}'
                
                return response
                
            except ImportError:
                # If neither is available, fall back to CSV
                flash("PDF generation requires PDF generation libraries. Providing CSV export instead.", "warning")
                return redirect("/export/csv")
                
    except ImportError:
        # If pdfkit is not available at all
        # Try WeasyPrint as an alternative
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            # Get user expenses
            expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date_created.desc()).all()
            
            # Calculate total
            total_spent = sum(expense.amount for expense in expenses)
            
            # Get user settings for currency formatting
            user_settings = UserSettings.query.filter_by(user_id=current_user.id).first()
            if not user_settings:
                user_settings = UserSettings(user_id=current_user.id)
                db.session.add(user_settings)
                db.session.commit()
            
            # Create a context with the format_currency function available
            template_context = {
                'expenses': expenses,
                'user': current_user,
                'total_spent': total_spent,
                'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_settings': user_settings,
                'format_currency': format_currency  # Pass the function to the template
            }
            
            # Render HTML template for PDF
            html_content = render_template(
                'pdf_template.html',
                **template_context
            )
            
            # Configure WeasyPrint
            font_config = FontConfiguration()
            html = HTML(string=html_content)
            
            # Generate PDF
            pdf_file = io.BytesIO()
            html.write_pdf(pdf_file, font_config=font_config)
            pdf_file.seek(0)
            
            # Prepare response
            filename = f"expenses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            response = make_response(pdf_file.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename={filename}'
            
            return response
            
        except ImportError:
            flash("PDF export is not available. Using CSV export instead.", "warning")
            return redirect("/export/csv")
        
    except Exception as e:
        # General error handling
        flash(f"PDF generation failed: {str(e)}. Using CSV export instead.", "danger")
        return redirect("/export/csv")

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)