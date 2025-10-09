from flask import Flask, render_template, request, redirect, url_for, flash, abort, send_file, make_response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import func
from transformers import pipeline
import re
import spacy
from textblob import TextBlob
from collections import defaultdict
from datetime import datetime
import io, csv, os, json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from fpdf import FPDF
import nltk

# Download vader_lexicon if not already present
nltk.download('vader_lexicon')
# ---------------- NLP / Sentiment Tools ---------------- #
sia = SentimentIntensityAnalyzer()
hf_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
nlp = spacy.load("en_core_web_sm")

def extract_aspects_smart(text):
    """Improved aspect extraction that handles contrastive sentences"""
    doc = nlp(text)
    aspects = {}
    
    # Enhanced stop words
    STOP_ASPECTS = {
        'i', 'you', 'it', 'they', 'we', 'what', 'this', 'that', 'which', 'who',
        'there', 'here', 'something', 'everything', 'nothing', 'someone',
        'everyone', 'anyone', 'somebody', 'everybody', 'anybody', 'thing',
        'things', 'stuff', 'item', 'items', 'way', 'time', 'part', 'kind', 'sort',
        'work', 'code', 'line', 'dedication'  # Your excluded terms
    }
    
    # Split on contrastive conjunctions to handle "but" sentences
    sentences = []
    current_sentence = []
    
    for token in doc:
        if token.text.lower() in ['but', 'however', 'although', 'though']:
            if current_sentence:
                sentences.append(' '.join([t.text for t in current_sentence]))
            current_sentence = [token]
        else:
            current_sentence.append(token)
    
    if current_sentence:
        sentences.append(' '.join([t.text for t in current_sentence]))
    
    # If no contrastive conjunctions found, use original sentence
    if len(sentences) <= 1:
        sentences = [sent.text for sent in doc.sents]
    
    # Analyze each segment separately
    for i, sent_text in enumerate(sentences):
        sent_doc = nlp(sent_text)
        sent_scores = sia.polarity_scores(sent_text)
        sent_compound = sent_scores['compound']
        
        # Extract aspects from this segment
        for chunk in sent_doc.noun_chunks:
            aspect = chunk.text.lower().strip()
            
            # Skip invalid aspects
            if (len(aspect) <= 2 or aspect in STOP_ASPECTS or 
                chunk.root.pos_ == 'PRON' or len(aspect.split()) == 1 and len(aspect) < 4):
                continue
            
            # Find adjectives that modify this aspect
            aspect_sentiment = "neutral"
            
            # Strategy 1: Look for adjectives within the noun chunk
            for token in chunk:
                if token.pos_ == 'ADJ':
                    adj_sentiment = sia.polarity_scores(token.text)['compound']
                    if adj_sentiment >= 0.1:
                        aspect_sentiment = "positive"
                    elif adj_sentiment <= -0.1:
                        aspect_sentiment = "negative"
                    break
            
            # Strategy 2: Look for nearby adjectives in the segment
            if aspect_sentiment == "neutral":
                for token in sent_doc:
                    if (token.pos_ == 'ADJ' and 
                        any(child in chunk for child in token.children)):
                        adj_sentiment = sia.polarity_scores(token.text)['compound']
                        if adj_sentiment >= 0.1:
                            aspect_sentiment = "positive"
                        elif adj_sentiment <= -0.1:
                            aspect_sentiment = "negative"
                        break
            
            # Strategy 3: Use segment sentiment with context awareness
            if aspect_sentiment == "neutral":
                # For contrastive segments after "but", be more sensitive to negative
                if i > 0 and any(word in sentences[i-1].lower() for word in ['but', 'however']):
                    if sent_compound <= -0.1:
                        aspect_sentiment = "negative"
                    elif sent_compound >= 0.1:
                        aspect_sentiment = "positive"
                else:
                    if sent_compound >= 0.2:
                        aspect_sentiment = "positive"
                    elif sent_compound <= -0.2:
                        aspect_sentiment = "negative"
            
            aspects[aspect] = aspect_sentiment
    
    return aspects
# ---------------- Flask Setup ---------------- #
from module import db, User, Review, AspectCategory, ProcessingLog, Dataset

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------- Helpers ---------------- #
def is_valid_email(email):
    regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(regex, email) is not None


def highlight_aspects(text, aspects):
    """Return text with aspects highlighted with sentiment color (case-insensitive)."""
    for aspect, sentiment in aspects.items():
        pattern = re.compile(re.escape(aspect), re.IGNORECASE)
        text = pattern.sub(f'<span class="highlight-{sentiment}">\\g<0></span>', text)
    return text


# ---------------- Routes ---------------- #

@app.route("/")
def home():
    return render_template("home.html")


# -------- Signup -------- #
@app.route("/signup", methods=["GET", "POST"])
def signup():
    email_error = username_error = password_error = None

    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"].strip()

        if not email:
            email_error = "Email is required."
        if not username:
            username_error = "Username is required."
        if not password:
            password_error = "Password is required."

        if not email_error and not username_error and not password_error:
            if not is_valid_email(email):
                email_error = "Invalid email format."
            if User.query.filter_by(email=email).first():
                email_error = "Email already registered."
            if User.query.filter_by(username=username).first():
                username_error = "Username already exists."

        if not email_error and not username_error and not password_error:
            new_user = User(
                username=username,
                email=email,
                password=generate_password_hash(password, method="scrypt")
            )
            db.session.add(new_user)
            db.session.commit()
            flash("✅ Account created! Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("signup.html",
                           email_error=email_error,
                           username_error=username_error,
                           password_error=password_error)


# -------- Login -------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    username_error = password_error = None

    if request.method == "POST":
        identifier = request.form["identifier"].strip()
        password = request.form["password"].strip()

        if not identifier:
            username_error = "Username or Email is required."
        if not password:
            password_error = "Password is required."

        if not username_error and not password_error:
            user = User.query.filter(
                (User.username == identifier) | (User.email == identifier)
            ).first()

            if not user:
                username_error = "User not found."
            elif not check_password_hash(user.password, password):
                password_error = "Incorrect password."
            else:
                login_user(user)
                return redirect(url_for("admin" if user.is_admin else "dashboard"))

    return render_template("login.html",
                           username_error=username_error,
                           password_error=password_error)


# -------- User Dashboard -------- #
@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.is_admin:
        return redirect(url_for("admin"))
     # FIX THIS LINE - count datasets for current user only
    datasets_uploaded = Dataset.query.filter_by(user_id=current_user.id).count()  # Fixed
    
    user_reviews = Review.query.filter_by(user_id=current_user.id).all()
    reviews_posted = len(user_reviews)
    reviews_analyzed = sum(1 for r in user_reviews if r.sentiment)
        
    # Calculate sentiment counts - FIXED
    positive_reviews = sum(1 for r in user_reviews if r.sentiment == "positive")
    negative_reviews = sum(1 for r in user_reviews if r.sentiment == "negative")
    neutral_reviews = sum(1 for r in user_reviews if r.sentiment == "neutral")



    return render_template("dashboard.html",
                           datasets_uploaded=datasets_uploaded,
                           reviews_analyzed=reviews_analyzed,
                           reviews_posted=reviews_posted,
                          positive_reviews=positive_reviews,
                           negative_reviews=negative_reviews,
                           neutral_reviews=neutral_reviews)


# -------- Insights -------- #
@app.route("/insights")
@login_required
def insights():
    if current_user.is_admin:
        return redirect(url_for("admin"))
    reviews = Review.query.filter_by(user_id=current_user.id).all()
    for review in reviews:
        aspects = review.aspect_sentiments.get("aspects", {}) if review.aspect_sentiments else {}
        review.highlighted_content = highlight_aspects(review.content, aspects)
    positive = sum(1 for r in reviews if r.sentiment == "positive")
    negative = sum(1 for r in reviews if r.sentiment == "negative")
    neutral = sum(1 for r in reviews if r.sentiment == "neutral")
    return render_template("insights.html",
                           reviews=reviews,
                           positive=positive,
                           negative=negative,
                           neutral=neutral,
                           total_reviews=len(reviews))


# -------- Review & Dataset Upload -------- #
@app.route("/review", methods=["GET", "POST"])
@login_required
def review():
    if current_user.is_admin:
        return redirect(url_for("admin"))
    
    if request.method == "POST":
        # Check if it's a file upload (dataset) or text review
        if 'file' in request.files and request.files['file'].filename:
            # Handle dataset file upload
            file = request.files['file']
            if file.filename == '':
                flash("⚠ Please select a file to upload.", "danger")
                return redirect(url_for("review"))
            
            # Validate file type
            allowed_extensions = {'csv', 'txt', 'json'}
            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                flash("⚠ Please upload a CSV, TXT, or JSON file.", "danger")
                return redirect(url_for("review"))
            
            try:
                # Create dataset record
                dataset_name = f"Dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                new_dataset = Dataset(
                    user_id=current_user.id,
                    name=dataset_name,
                    reviews_count=0  # Will be updated after processing
                )
                db.session.add(new_dataset)
                db.session.commit()
                
                # Process the file and extract reviews
                reviews_count = process_uploaded_file(file, current_user.id, new_dataset.id)
                
                # Update dataset with actual count
                new_dataset.reviews_count = reviews_count
                db.session.commit()
                
                flash(f"✅ Dataset '{dataset_name}' uploaded successfully with {reviews_count} reviews!", "success")
                
            except Exception as e:
                db.session.rollback()
                flash(f"❌ Error processing file: {str(e)}", "danger")
                return redirect(url_for("review"))
                
        else:
            # Handle individual text review
            content = request.form["content"].strip()
            if not content:
                flash("⚠ Review content cannot be empty.", "danger")
                return redirect(url_for("review"))
            
            # Process individual review
            scores = sia.polarity_scores(content)
            compound = scores['compound']
            sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
            confidence = round(abs(compound) * 100, 2)
            hf_result = hf_sentiment(content[:512])
            hf_label = hf_result[0]['label']
            hf_score = round(hf_result[0]['score'] * 100, 2)
            aspects = extract_aspects_smart(content)
            
            new_review = Review(
                user_id=current_user.id,
                content=content,
                sentiment=sentiment,
                confidence=confidence,
                aspect_sentiments={
                    "huggingface_label": hf_label,
                    "huggingface_score": hf_score,
                    "aspects": aspects
                }
            )
            db.session.add(new_review)
            db.session.commit()
            flash("✅ Review submitted successfully.", "success")
        
        return redirect(url_for("review"))
    
    # GET request - show user's reviews
    user_reviews = Review.query.filter_by(user_id=current_user.id).all()
    
    # Calculate sentiment counts for chart
    positive = sum(1 for r in user_reviews if r.sentiment == "positive")
    negative = sum(1 for r in user_reviews if r.sentiment == "negative")
    neutral = sum(1 for r in user_reviews if r.sentiment == "neutral")
    
    # Prepare reviews with highlighted content
    for review in user_reviews:
        aspects = review.aspect_sentiments.get("aspects", {}) if review.aspect_sentiments else {}
        review.highlighted_content = highlight_aspects(review.content, aspects)
    
    return render_template("review.html",
                         reviews=user_reviews,
                         positive=positive,
                         negative=negative,
                         neutral=neutral)

# Dataset processing function handling
def process_uploaded_file(file, user_id, dataset_id):
    """Process uploaded file and create review records"""
    filename = file.filename.lower()
    reviews_count = 0
    
    try:
        if filename.endswith('.csv'):
            # Process CSV file
            content = file.read().decode('utf-8')
            csv_reader = csv.reader(io.StringIO(content))
            
            for row in csv_reader:
                if row and row[0].strip():  # First column contains review text
                    process_single_review(row[0].strip(), user_id, dataset_id)
                    reviews_count += 1
        
        elif filename.endswith('.txt'):
            # Process TXT file (one review per line)
            content = file.read().decode('utf-8')
            lines = content.split('\n')
            
            for line in lines:
                if line.strip():
                    process_single_review(line.strip(), user_id, dataset_id)
                    reviews_count += 1
        
        elif filename.endswith('.json'):
            # Process JSON file
            content = file.read().decode('utf-8')
            data = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'review' in item:
                        process_single_review(item['review'], user_id, dataset_id)
                        reviews_count += 1
                    elif isinstance(item, str):
                        process_single_review(item, user_id, dataset_id)
                        reviews_count += 1
            elif isinstance(data, dict) and 'reviews' in data:
                for review_text in data['reviews']:
                    process_single_review(review_text, user_id, dataset_id)
                    reviews_count += 1
        
        # COMMIT ALL REVIEWS AFTER PROCESSING
        db.session.commit()
        return reviews_count
        
    except Exception as e:
        db.session.rollback()  # Rollback on error
        raise Exception(f"File processing error: {str(e)}")

def process_single_review(content, user_id, dataset_id):
    """Process a single review and save to database"""
    if not content or len(content.strip()) < 3:
        return
    
    # Sentiment analysis
    scores = sia.polarity_scores(content)
    compound = scores['compound']
    sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    confidence = round(abs(compound) * 100, 2)
    
    # HuggingFace analysis
    try:
        hf_result = hf_sentiment(content[:512])
        hf_label = hf_result[0]['label']
        hf_score = round(hf_result[0]['score'] * 100, 2)
    except:
        hf_label = "unknown"
        hf_score = 0.0
    
    # Improved aspect extraction
    aspects = extract_aspects_smart(content)
    
    # Create review record
    new_review = Review(
        user_id=user_id,
        content=content,
        sentiment=sentiment,
        confidence=confidence,
        aspect_sentiments={
            "huggingface_label": hf_label,
            "huggingface_score": hf_score,
            "aspects": aspects,
            "dataset_id": dataset_id  # Track which dataset this came from
        }
    )
    db.session.add(new_review)
    # Note: We don't commit here, we commit after processing all reviews
# -------- Update Old Reviews -------- #
def update_existing_reviews():
    reviews = Review.query.filter(
        (Review.sentiment.is_(None)) |
        (Review.confidence.is_(None)) |
        (Review.aspect_sentiments.is_(None))
    ).all()
    for review in reviews:
        scores = sia.polarity_scores(review.content)
        compound = scores['compound']
        review.sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
        review.confidence = round(abs(compound) * 100, 2)
        hf_result = hf_sentiment(review.content[:512])
        hf_label = hf_result[0]['label']
        hf_score = round(hf_result[0]['score'] * 100, 2)
        aspects = extract_aspects_smart(review.content)
        review.aspect_sentiments = {
            "huggingface_label": hf_label,
            "huggingface_score": hf_score,
            "aspects": aspects
        }
    db.session.commit()


@app.route("/admin/update_sentiments")
@login_required
def update_sentiments():
    if not current_user.is_admin:
        abort(403)
    update_existing_reviews()
    flash("✅ All reviews updated.", "success")
    return redirect(url_for("admin"))


# -------- Profile -------- #
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        bio = request.form.get("bio", "").strip()
        if User.query.filter(User.username == username, User.id != current_user.id).first():
            flash("⚠ Username already taken.", "danger")
            return redirect(url_for("profile"))
        if User.query.filter(User.email == email, User.id != current_user.id).first():
            flash("⚠ Email already registered.", "danger")
            return redirect(url_for("profile"))
        current_user.username = username
        current_user.email = email
        current_user.bio = bio
        db.session.commit()
        flash("✅ Profile updated successfully!", "success")
        return redirect(url_for("profile"))
    return render_template("userprofile.html", user=current_user)



# -------- Admin Dashboard -------- #
@app.route("/admin")
@login_required
def admin():
    if not current_user.is_admin:
        return redirect(url_for("review"))
    
    # Dashboard metrics
    total_users = User.query.count()
    total_datasets = Dataset.query.count()
    total_reviews = Review.query.count()
    
    # Active users today
    today = datetime.utcnow().date()
    active_users_today = User.query.filter(
        func.date(User.created_at) == today
    ).count()
    
    # Sentiment counts for analytics
    positive = Review.query.filter_by(sentiment="positive").count()
    negative = Review.query.filter_by(sentiment="negative").count()
    neutral = Review.query.filter_by(sentiment="neutral").count()
    avg_confidence = round(db.session.query(func.avg(Review.confidence)).scalar() or 0, 2)
    
    # Get reviews with user info
    reviews = db.session.query(Review, User).join(User, Review.user_id == User.id).all()
    for review, user in reviews:
        aspects = review.aspect_sentiments.get("aspects", {}) if review.aspect_sentiments else {}
        review.highlighted_content = highlight_aspects(review.content, aspects)
    
    # Aspect summary for analytics section
    aspect_summary = defaultdict(lambda: {"positive":0,"negative":0,"neutral":0})
    for r in Review.query.filter(Review.aspect_sentiments.isnot(None)).all():
        if isinstance(r.aspect_sentiments, dict):
            for aspect, sentiment in r.aspect_sentiments.get("aspects", {}).items():
                if sentiment in ("positive","negative","neutral"):
                    aspect_summary[aspect][sentiment] += 1
    
    # Get users for users section
    users = User.query.all()
    
    # Get aspects for aspects section
    aspects_list = AspectCategory.query.all()
    
    # Get logs for monitor section
    logs = ProcessingLog.query.order_by(ProcessingLog.created_at.desc()).all()
    
    # Reports data
    top_users = db.session.query(User.username, func.count(Review.id).label("cnt")) \
                 .join(Review, Review.user_id==User.id) \
                 .group_by(User.id).order_by(func.count(Review.id).desc()).limit(5).all()

    # -------- Aspect Analysis Data -------- #
    # Get all reviews for aspect analysis
    all_reviews = Review.query.all()
    
    # Aggregate aspect data from all reviews
    aspect_data = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
    total_aspects_found = 0
    
    for review in all_reviews:
        # Extract aspects from review content
        if review.aspect_sentiments and isinstance(review.aspect_sentiments, dict):
            aspects = review.aspect_sentiments.get("aspects", {})
        else:
            # Extract aspects if not already stored
            aspects = extract_aspects_smart(review.content)
            # Update the review with extracted aspects
            if not review.aspect_sentiments:
                review.aspect_sentiments = {}
            review.aspect_sentiments["aspects"] = aspects
            review.aspect_sentiments["extracted_at"] = datetime.utcnow().isoformat()
        
        # Aggregate the aspect data
        for aspect, sentiment in aspects.items():
            if sentiment in ("positive", "negative", "neutral"):
                aspect_data[aspect][sentiment] += 1
                aspect_data[aspect]["total"] += 1
                total_aspects_found += 1
    
    # Commit aspect data to reviews
    try:
        db.session.commit()
    except:
        db.session.rollback()
    
    # Convert to list and sort by total mentions
    aspect_list = []
    for aspect, counts in aspect_data.items():
        if counts["total"] >= 1:  # Only include aspects mentioned at least once
            aspect_list.append({
                "name": aspect,
                "positive": counts["positive"],
                "negative": counts["negative"], 
                "neutral": counts["neutral"],
                "total": counts["total"],
                "positive_percent": round((counts["positive"] / counts["total"]) * 100, 2) if counts["total"] > 0 else 0,
                "negative_percent": round((counts["negative"] / counts["total"]) * 100, 2) if counts["total"] > 0 else 0,
                "neutral_percent": round((counts["neutral"] / counts["total"]) * 100, 2) if counts["total"] > 0 else 0
            })
    
    # Sort by total mentions (descending)
    aspect_list.sort(key=lambda x: x["total"], reverse=True)
    
    # Prepare data for charts
    chart_aspects = [aspect["name"] for aspect in aspect_list[:10]]  # Top 10 aspects
    chart_positive = [aspect["positive"] for aspect in aspect_list[:10]]
    chart_negative = [aspect["negative"] for aspect in aspect_list[:10]]
    chart_neutral = [aspect["neutral"] for aspect in aspect_list[:10]]
    
    # Get sample reviews for each top aspect
    aspect_reviews = {}
    for aspect in chart_aspects[:5]:
        sample_reviews = []
        for review in all_reviews[:20]:  # Check first 20 reviews for examples
            if review.aspect_sentiments and isinstance(review.aspect_sentiments, dict):
                aspects_in_review = review.aspect_sentiments.get("aspects", {})
            else:
                aspects_in_review = extract_aspects_smart(review.content)
                
            if aspect in aspects_in_review:
                sample_reviews.append({
                    'content': review.content[:100] + '...' if len(review.content) > 100 else review.content,
                    'sentiment': aspects_in_review[aspect],
                    'review_sentiment': review.sentiment
                })
                if len(sample_reviews) >= 3:  # Limit to 3 sample reviews per aspect
                    break
        aspect_reviews[aspect] = sample_reviews
    # -------- End Aspect Analysis Data -------- #

    return render_template("admin_base.html",
                           # Dashboard data
                           total_users=total_users,
                           total_datasets=total_datasets,
                           total_reviews=total_reviews,
                           active_users_today=active_users_today,
                           
                           # Analytics data
                           positive=positive,
                           negative=negative, 
                           neutral=neutral,
                           avg_confidence=avg_confidence,
                           aspect_summary=dict(aspect_summary),
                           
                           # Reviews data
                           reviews=reviews,
                           
                           # Users data
                           users=users,
                           
                           # Aspects data
                           aspects=aspects_list,
                           
                           # Monitor data
                           logs=logs,
                           
                           # Reports data
                           top_users=top_users,
                           
                           # Aspect Analysis data
                           aspect_list=aspect_list,
                           chart_aspects=chart_aspects,
                           chart_positive=chart_positive,
                           chart_negative=chart_negative,
                           chart_neutral=chart_neutral,
                           aspect_reviews=aspect_reviews,
                           total_aspects_found=total_aspects_found,
                           total_reviews_processed=len(all_reviews))
# -------- Admin Extra Routes -------- #
@app.route("/admin/users")
@login_required
def admin_users():
    if not current_user.is_admin: abort(403)
    users = User.query.all()
     # Pass positive, negative, neutral for template
    positive = Review.query.filter_by(sentiment="positive").count()
    negative = Review.query.filter_by(sentiment="negative").count()
    neutral = Review.query.filter_by(sentiment="neutral").count()
    avg_confidence = round(db.session.query(func.avg(Review.confidence)).scalar() or 0, 2)

    return render_template("admin_users.html",
                           users=users,
                           positive=positive,
                           negative=negative,
                           neutral=neutral,
                           avg_confidence=avg_confidence)
   # return render_template("admin_users.html", users=users)
@app.route("/admin/aspects", methods=["GET","POST"])
@login_required
def admin_aspects():
    if not current_user.is_admin: 
        abort(403)
    
    if request.method == "POST":
        name = request.form.get("name","").strip()
        industry = request.form.get("industry","").strip()
        
        if name:
            # Check if aspect already exists
            existing_aspect = AspectCategory.query.filter_by(name=name).first()
            if existing_aspect:
                flash("Aspect already exists", "danger")
            else:
                new_aspect = AspectCategory(name=name, industry=industry if industry else None)
                db.session.add(new_aspect)
                db.session.commit()
                flash("Aspect added successfully", "success")
        
        return redirect(url_for("admin_aspects"))
    
    aspects = AspectCategory.query.all()
    
    # Required for admin_base.html
    positive = Review.query.filter_by(sentiment="positive").count()
    negative = Review.query.filter_by(sentiment="negative").count()
    neutral = Review.query.filter_by(sentiment="neutral").count()
    avg_confidence = round(db.session.query(func.avg(Review.confidence)).scalar() or 0, 2)

    return render_template("admin_aspects.html",
                           aspects=aspects,
                           positive=positive,
                           negative=negative,
                           neutral=neutral,
                           avg_confidence=avg_confidence)




# -------- Admin Reports -------- #
@app.route("/admin/reports")
@login_required
def admin_reports():
    if not current_user.is_admin: abort(403)

    total_reviews = Review.query.count()
    total_users = User.query.count()
    total_datasets = Dataset.query.count()

    top_users = db.session.query(User.username, func.count(Review.id).label("cnt")) \
                 .join(Review, Review.user_id==User.id) \
                 .group_by(User.id).order_by(func.count(Review.id).desc()).limit(5).all()

    positive = Review.query.filter_by(sentiment="positive").count()
    negative = Review.query.filter_by(sentiment="negative").count()
    neutral = Review.query.filter_by(sentiment="neutral").count()
    avg_confidence = round(db.session.query(func.avg(Review.confidence)).scalar() or 0, 2)

    return render_template("admin_reports.html",
                           total_reviews=total_reviews,
                           total_users=total_users,
                           total_datasets=total_datasets,
                           top_users=top_users,
                           positive=positive,
                           negative=negative,
                           neutral=neutral,
                           avg_confidence=avg_confidence)

# -------- Admin Reports CSV Export -------- #
@app.route("/admin/reports/export_csv")
@login_required
def admin_reports_export_csv():
    if not current_user.is_admin:
        abort(403)

    total_reviews = Review.query.count()
    total_users = User.query.count()
    total_datasets = Dataset.query.count()

    top_users = db.session.query(User.username, func.count(Review.id).label("cnt")) \
                 .join(Review, Review.user_id==User.id) \
                 .group_by(User.id).order_by(func.count(Review.id).desc()).limit(5).all()

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["Metric", "Value"])
    cw.writerow(["Total Reviews", total_reviews])
    cw.writerow(["Total Users", total_users])
    cw.writerow(["Total Datasets", total_datasets])
    cw.writerow([])
    cw.writerow(["Top Users", "Review Count"])
    for username, count in top_users:
        cw.writerow([username, count])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=admin_reports.csv"
    output.headers["Content-type"] = "text/csv"
    return output
# -------- Admin Reports PDF Export -------- #
from functools import wraps

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/reports/export_pdf')
@login_required
@admin_required
def admin_reports_export_pdf():
    try:
        # Gather report data
        total_reviews = Review.query.count()
        total_users = User.query.count()
        total_datasets = Dataset.query.count()
        top_users = db.session.query(User.username, func.count(Review.id).label("cnt")) \
                     .join(Review, Review.user_id==User.id) \
                     .group_by(User.id).order_by(func.count(Review.id).desc()).limit(5).all()

        # Create PDF buffer
        pdf_buffer = BytesIO()
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        
        # Set up PDF
        pdf.setTitle("Admin Reports")
        y_position = height - 50
        line_height = 14
        
        # Add title
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, y_position, "Admin Reports")
        y_position -= line_height * 2
        
        # Add summary statistics
        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, y_position, f"Total Reviews: {total_reviews}")
        y_position -= line_height
        pdf.drawString(50, y_position, f"Total Users: {total_users}")
        y_position -= line_height
        pdf.drawString(50, y_position, f"Total Datasets: {total_datasets}")
        y_position -= line_height * 2
        
        # Add most active users section
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_position, "Most Active Users:")
        y_position -= line_height
        
        pdf.setFont("Helvetica", 10)
        if top_users:
            for username, count in top_users:
                if y_position < 50:
                    pdf.showPage()
                    y_position = height - 50
                    pdf.setFont("Helvetica", 10)
                pdf.drawString(50, y_position, f"{username}: {count} reviews")
                y_position -= line_height
        else:
            pdf.drawString(50, y_position, "No user data available")
            y_position -= line_height
        
        # Save PDF
        pdf.save()
        pdf_buffer.seek(0)
        
        # Return PDF as download
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"admin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        flash(f"Error generating PDF: {str(e)}", "error")
        return redirect(url_for('admin_reports'))
    
"""export"""
# -------- Admin Reports Review Export -------- #
@app.route("/admin/reports/export_reviews_csv")
@login_required
def admin_export_reviews_csv():
    if not current_user.is_admin:
        abort(403)
    
    try:
        # Get all reviews with user information
        reviews = db.session.query(Review, User).join(User, Review.user_id == User.id).all()
        
        si = io.StringIO()
        cw = csv.writer(si)
        
        # Write header
        cw.writerow(["User ID", "Username", "Email", "Review Content", "Sentiment", 
                    "Confidence", "Aspects", "Created At"])
        
        # Write review data
        for review, user in reviews:
            # Extract aspects as string
            aspects_str = ""
            if review.aspect_sentiments and isinstance(review.aspect_sentiments, dict):
                aspects = review.aspect_sentiments.get("aspects", {})
                aspects_str = "; ".join([f"{k}:{v}" for k, v in aspects.items()])
            
            cw.writerow([
                user.id,
                user.username,
                user.email,
                review.content,
                review.sentiment,
                f"{review.confidence}%" if review.confidence else "N/A",
                aspects_str,
                review.created_at.strftime("%Y-%m-%d %H:%M:%S") if review.created_at else "N/A"
            ])
        
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=user_reviews_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
        
    except Exception as e:
        flash(f"Error exporting reviews: {str(e)}", "danger")
        return redirect(url_for("admin_reports"))

@app.route("/admin/reports/export_reviews_pdf")
@login_required
def admin_export_reviews_pdf():
    if not current_user.is_admin:
        abort(403)
    
    try:
        # Get all reviews with user information
        reviews = db.session.query(Review, User).join(User, Review.user_id == User.id).all()
        
        # Create PDF buffer
        pdf_buffer = BytesIO()
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        
        # Set up PDF
        pdf.setTitle("User Reviews Export")
        y_position = height - 50
        line_height = 14
        
        # Add title
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, y_position, "User Reviews Export")
        y_position -= line_height * 2
        
        # Add summary
        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, y_position, f"Total Reviews: {len(reviews)}")
        y_position -= line_height
        pdf.drawString(50, y_position, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y_position -= line_height * 2
        
        # Add reviews
        pdf.setFont("Helvetica", 10)
        for i, (review, user) in enumerate(reviews, 1):
            # Check if we need a new page
            if y_position < 100:
                pdf.showPage()
                y_position = height - 50
                pdf.setFont("Helvetica", 10)
            
            # Review header
            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawString(50, y_position, f"Review #{i}")
            y_position -= line_height
            
            pdf.setFont("Helvetica", 10)
            pdf.drawString(50, y_position, f"User: {user.username} ({user.email})")
            y_position -= line_height
            pdf.drawString(50, y_position, f"Sentiment: {review.sentiment} (Confidence: {review.confidence}%)")
            y_position -= line_height
            
            # Review content with word wrap
            content = review.content
            if len(content) > 150:
                content = content[:147] + "..."
            
            content_lines = simpleSplit(f"Content: {content}", "Helvetica", 10, 500)
            for line in content_lines:
                if y_position < 50:
                    pdf.showPage()
                    y_position = height - 50
                    pdf.setFont("Helvetica", 10)
                pdf.drawString(50, y_position, line)
                y_position -= line_height
            
            # Aspects
            if review.aspect_sentiments and isinstance(review.aspect_sentiments, dict):
                aspects = review.aspect_sentiments.get("aspects", {})
                if aspects:
                    aspects_text = "Aspects: " + ", ".join([f"{k}({v})" for k, v in aspects.items()])
                    if y_position < 50:
                        pdf.showPage()
                        y_position = height - 50
                        pdf.setFont("Helvetica", 10)
                    pdf.drawString(50, y_position, aspects_text)
                    y_position -= line_height
            
            # Date
            date_str = review.created_at.strftime('%Y-%m-%d %H:%M') if review.created_at else 'N/A'
            if y_position < 50:
                pdf.showPage()
                y_position = height - 50
                pdf.setFont("Helvetica", 10)
            pdf.drawString(50, y_position, f"Date: {date_str}")
            y_position -= line_height * 2
        
        # Save PDF
        pdf.save()
        pdf_buffer.seek(0)
        
        # Return PDF as download
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"user_reviews_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        flash(f"Error generating PDF: {str(e)}", "error")
        return redirect(url_for('admin_reports'))
# -------- Export Reviews by Sentiment -------- #
@app.route("/admin/reports/export_reviews_by_sentiment/<sentiment>")
@login_required
def admin_export_reviews_by_sentiment(sentiment):
    if not current_user.is_admin:
        abort(403)
    
    if sentiment not in ['positive', 'negative', 'neutral']:
        flash("Invalid sentiment type", "danger")
        return redirect(url_for("admin_reports"))
    
    try:
        # Get reviews by sentiment with user information
        reviews = db.session.query(Review, User).join(User, Review.user_id == User.id)\
                    .filter(Review.sentiment == sentiment).all()
        
        si = io.StringIO()
        cw = csv.writer(si)
        
        # Write header
        cw.writerow(["User ID", "Username", "Email", "Review Content", "Sentiment", 
                    "Confidence", "Aspects", "Created At"])
        
        # Write review data
        for review, user in reviews:
            # Extract aspects as string
            aspects_str = ""
            if review.aspect_sentiments and isinstance(review.aspect_sentiments, dict):
                aspects = review.aspect_sentiments.get("aspects", {})
                aspects_str = "; ".join([f"{k}:{v}" for k, v in aspects.items()])
            
            cw.writerow([
                user.id,
                user.username,
                user.email,
                review.content,
                review.sentiment,
                f"{review.confidence}%" if review.confidence else "N/A",
                aspects_str,
                review.created_at.strftime("%Y-%m-%d %H:%M:%S") if review.created_at else "N/A"
            ])
        
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = f"attachment; filename={sentiment}_reviews_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
        
    except Exception as e:
        flash(f"Error exporting {sentiment} reviews: {str(e)}", "danger")
        return redirect(url_for("admin_reports"))
    

@app.route("/admin/users/<int:user_id>")
@login_required
def admin_view_user(user_id):
    user = User.query.get_or_404(user_id)
    return render_template("admin_view_user.html", user=user)

@app.route("/admin/users/delete/<int:user_id>", methods=["POST"])
@login_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)

    try:
        db.session.delete(user)
        db.session.commit()
        flash("User deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting user: {str(e)}", "danger")

    return redirect(url_for("admin_users"))

@app.route("/admin/update_sentiments")
@login_required
def admin_update_sentiments():
    # Your logic here
    flash("Sentiments updated successfully!", "success")
    return redirect(url_for("admin"))


# -------- Aspect Management Routes -------- #
@app.route("/admin/aspects/edit", methods=["POST"])
@login_required
def admin_aspect_edit():
    if not current_user.is_admin:
        abort(403)
    
    aspect_id = request.form.get("aspect_id")
    name = request.form.get("name", "").strip()
    industry = request.form.get("industry", "").strip()
    
    if not name:
        flash("Aspect name is required", "danger")
        return redirect(url_for("admin_aspects"))
    
    aspect = AspectCategory.query.get_or_404(aspect_id)
    aspect.name = name
    aspect.industry = industry if industry else None
    
    db.session.commit()
    flash("Aspect updated successfully", "success")
    return redirect(url_for("admin_aspects"))

@app.route("/admin/aspects/delete/<int:aspect_id>", methods=["POST"])
@login_required
def admin_aspect_delete(aspect_id):
    if not current_user.is_admin:
        abort(403)
    
    aspect = AspectCategory.query.get_or_404(aspect_id)
    
    try:
        db.session.delete(aspect)
        db.session.commit()
        flash("Aspect deleted successfully", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting aspect: {str(e)}", "danger")
    
    return redirect(url_for("admin_aspects"))

# -------- Model Feedback Route -------- #
@app.route("/admin/monitor/feedback", methods=["POST"])
@login_required
def admin_model_feedback():
    if not current_user.is_admin:
        abort(403)
    
    data = request.get_json()
    feedback = data.get('feedback')
    
    # Here you would typically save the feedback to improve your model
    # For now, we'll just log it
    print(f"Model feedback received: {feedback}")
    
    return jsonify({"status": "success", "message": "Feedback recorded"})

# -------- Enhanced Monitor Route -------- #
@app.route("/admin/monitor")
@login_required
def admin_monitor():
    if not current_user.is_admin:
        abort(403)
    
    logs = ProcessingLog.query.order_by(ProcessingLog.created_at.desc()).all()
    
    # Calculate performance metrics
    successful_logs = [log for log in logs if log.status == 'success']
    avg_processing_time = round(
        sum(log.processing_time or 0 for log in successful_logs) / len(successful_logs), 2
    ) if successful_logs else 0.0
    
    success_rate = round(
        (len(successful_logs) / len(logs)) * 100, 2
    ) if logs else 100.0
    
    # Estimate storage used (in MB)
    total_reviews = Review.query.count()
    storage_used = round(total_reviews * 0.1, 2)  # Rough estimate: 0.1MB per review
    
    # Required for admin_base.html
    positive = Review.query.filter_by(sentiment="positive").count()
    negative = Review.query.filter_by(sentiment="negative").count()
    neutral = Review.query.filter_by(sentiment="neutral").count()
    avg_confidence = round(db.session.query(func.avg(Review.confidence)).scalar() or 0, 2)

    return render_template("admin_monitor.html", 
                           logs=logs,
                           avg_processing_time=avg_processing_time,
                           success_rate=success_rate,
                           storage_used=storage_used,
                           active_models=2,  # VADER + HuggingFace
                           positive=positive,
                           negative=negative,
                           neutral=neutral,
                           avg_confidence=avg_confidence)

# -------- Aspect Analysis Route -------- #
@app.route("/admin/aspect_analysis")
@login_required
def admin_aspect_analysis():
    if not current_user.is_admin:
        abort(403)
    
    # Get all reviews
    reviews = Review.query.all()
    
    # Aggregate aspect data from all reviews
    aspect_data = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
    total_aspects_found = 0
    
    for review in reviews:
        # Always extract fresh aspects to ensure consistency
        aspects = extract_aspects_smart(review.content)
        
        # Update the review with extracted aspects
        if not review.aspect_sentiments or not isinstance(review.aspect_sentiments, dict):
            review.aspect_sentiments = {}
        
        review.aspect_sentiments["aspects"] = aspects
        review.aspect_sentiments["extracted_at"] = datetime.utcnow().isoformat()
        
        # Aggregate the aspect data
        for aspect, sentiment in aspects.items():
            if sentiment in ("positive", "negative", "neutral"):
                aspect_data[aspect][sentiment] += 1
                aspect_data[aspect]["total"] += 1
                total_aspects_found += 1
    
    # Commit aspect data to reviews
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error committing aspect data: {e}")
    
    # Convert to list and sort by total mentions
    aspect_list = []
    for aspect, counts in aspect_data.items():
        if counts["total"] >= 1:  # Only include aspects mentioned at least once
            total_mentions = counts["total"]
            aspect_list.append({
                "name": aspect.title(),  # Proper capitalization for display
                "positive": counts["positive"],
                "negative": counts["negative"], 
                "neutral": counts["neutral"],
                "total": total_mentions,
                "positive_percent": round((counts["positive"] / total_mentions) * 100, 2) if total_mentions > 0 else 0,
                "negative_percent": round((counts["negative"] / total_mentions) * 100, 2) if total_mentions > 0 else 0,
                "neutral_percent": round((counts["neutral"] / total_mentions) * 100, 2) if total_mentions > 0 else 0,
                "net_sentiment": round(((counts["positive"] - counts["negative"]) / total_mentions) * 100, 2) if total_mentions > 0 else 0
            })
    
    # Sort by total mentions (descending)
    aspect_list.sort(key=lambda x: x["total"], reverse=True)
    
    # Prepare data for charts - limit to top 15 for better visualization
    top_aspects = aspect_list[:15]
    chart_aspects = [aspect["name"] for aspect in top_aspects]
    chart_positive = [aspect["positive"] for aspect in top_aspects]
    chart_negative = [aspect["negative"] for aspect in top_aspects]
    chart_neutral = [aspect["neutral"] for aspect in top_aspects]
    
    # Get sample reviews for each top aspect
    aspect_reviews = {}
    for aspect_data in top_aspects[:10]:  # Top 10 aspects for examples
        aspect_name = aspect_data["name"].lower()
        sample_reviews = []
        
        for review in reviews:
            # Extract aspects from this review
            if review.aspect_sentiments and isinstance(review.aspect_sentiments, dict):
                aspects_in_review = review.aspect_sentiments.get("aspects", {})
            else:
                aspects_in_review = extract_aspects_smart(review.content)
                
            # Check if this aspect exists in the review (case-insensitive)
            review_aspects_lower = {k.lower(): v for k, v in aspects_in_review.items()}
            if aspect_name in review_aspects_lower:
                sample_reviews.append({
                    'content': review.content[:150] + '...' if len(review.content) > 150 else review.content,
                    'sentiment': review_aspects_lower[aspect_name],
                    'review_sentiment': review.sentiment,
                    'confidence': review.confidence
                })
                if len(sample_reviews) >= 3:  # Limit to 3 sample reviews per aspect
                    break
        aspect_reviews[aspect_data["name"]] = sample_reviews
    
    # Calculate summary statistics
    positive_reviews = sum(1 for r in reviews if r.sentiment == "positive")
    negative_reviews = sum(1 for r in reviews if r.sentiment == "negative")
    neutral_reviews = sum(1 for r in reviews if r.sentiment == "neutral")
    avg_confidence = round(db.session.query(func.avg(Review.confidence)).scalar() or 0, 2)

    return render_template("admin_aspect_analysis.html",
                           aspects=aspect_list,
                           chart_aspects=chart_aspects,
                           chart_positive=chart_positive,
                           chart_negative=chart_negative,
                           chart_neutral=chart_neutral,
                           aspect_reviews=aspect_reviews,
                           total_aspects_found=total_aspects_found,
                           total_reviews_processed=len(reviews),
                           positive=positive_reviews,
                           negative=negative_reviews,
                           neutral=neutral_reviews,
                           avg_confidence=avg_confidence)
# -------- Debug Aspect Extraction -------- #
@app.route("/admin/debug_aspect_extraction")
@login_required
def debug_aspect_extraction():
    if not current_user.is_admin:
        abort(403)
    
    # Test with problematic examples
    test_texts = [
        "The battery life is amazing but the price is too high",
        "You are great and your service is excellent", 
        "This product has good quality but poor packaging",
        "I love the display but hate the battery",
        "Your work is amazing but the delivery time is terrible"
    ]
    
    results = []
    for text in test_texts:
        aspects = extract_aspects_smart(text)
        doc = nlp(text)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        results.append({
            'text': text,
            'noun_chunks': noun_chunks,
            'aspects_extracted': aspects
        })
    
    return jsonify(results)

# -------- Force Complete Reprocessing -------- #
@app.route("/admin/force_reprocess_all_aspects")
@login_required
def force_reprocess_all_aspects():
    if not current_user.is_admin:
        abort(403)
    
    try:
        # Get all reviews
        reviews = Review.query.all()
        updated_count = 0
        total_aspects_found = 0
        
        for review in reviews:
            # Completely reprocess with new logic
            aspects = extract_aspects_smart(review.content)
            
            # Create new aspect_sentiments structure
            review.aspect_sentiments = {
                "aspects": aspects,
                "reprocessed_at": datetime.utcnow().isoformat(),
                "version": "2.0"  # Mark with version to track changes
            }
            
            total_aspects_found += len(aspects)
            updated_count += 1
            
            # Commit in batches to avoid memory issues
            if updated_count % 100 == 0:
                db.session.commit()
                print(f"Processed {updated_count} reviews...")
        
        db.session.commit()
        flash(f"✅ Successfully reprocessed {updated_count} reviews! Found {total_aspects_found} total aspects.", "success")
        
    except Exception as e:
        db.session.rollback()
        flash(f"❌ Error reprocessing aspects: {str(e)}", "danger")
    
    return redirect(url_for("admin_aspect_analysis"))



# -------- Logout -------- #
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("✅ You have been logged out.", "info")
    return redirect(url_for("login"))


# -------- Main -------- #
if __name__ == "__main__":
    app.run(debug=True)