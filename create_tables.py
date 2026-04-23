from app import app, db

with app.app_context():
    print('Creating tables using db.create_all()')
    db.create_all()
    print('Done')
