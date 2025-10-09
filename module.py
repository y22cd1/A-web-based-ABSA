from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # ✅ join date

    reviews = db.relationship("Review", back_populates="user", lazy=True)

    def __repr__(self):
        return f"<User {self.username} ({self.email})>"


class Review(db.Model):
    __tablename__ = "review"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20))  # sentiment analysis
    confidence  = db.Column(db.Float)     # confidence score
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # ✅ review timestamp
    aspect_sentiments = db.Column(db.JSON, nullable=True)

    user = db.relationship("User", back_populates="reviews")




    
class AspectCategory(db.Model):
    __tablename__ = "aspect_category"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    industry = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ProcessingLog(db.Model):
    __tablename__ = "processing_log"
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(200), nullable=True)
    reviews_count = db.Column(db.Integer, nullable=True)
    processing_time = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(50))   # success / error
    message = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Dataset(db.Model):
    __tablename__ = "dataset"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    reviews_count = db.Column(db.Integer, default=0)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="datasets")

    def __repr__(self):
        return f"<Review by {self.user.username}: {self.content[:20]}...>"
