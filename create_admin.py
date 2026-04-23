import getpass
from app import app, db
from module import User
from werkzeug.security import generate_password_hash
def create_admin(username, email, password, is_admin=True):
    with app.app_context():
        user = User.query.filter((User.username == username) | (User.email == email)).first()
        if user:
            print('User exists, updating to admin and setting password...')
            user.username = username
            user.email = email
            user.password = generate_password_hash(password, method='scrypt')
            user.is_admin = is_admin
        else:
            user = User(username=username, email=email, password=generate_password_hash(password, method='scrypt'), is_admin=is_admin)
            db.session.add(user)
        db.session.commit()
        print(f'Admin user "{username}" is ready.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create or update an admin user')
    parser.add_argument('--username', default='admin', help='Admin username')
    parser.add_argument('--email', default='admin@example.com', help='Admin email')
    parser.add_argument('--password', help='Admin password (will prompt if not provided)')
    args = parser.parse_args()

    pwd = args.password
    if not pwd:
        pwd = getpass.getpass('Enter password for admin user: ')
        pwd2 = getpass.getpass('Confirm password: ')
        if pwd != pwd2:
            print('Passwords do not match. Exiting.')
            raise SystemExit(1)

    create_admin(args.username, args.email, pwd)
