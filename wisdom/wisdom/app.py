import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Database setup
conn = sqlite3.connect('database.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        password TEXT NOT NULL
    )
''')
conn.commit()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        session_data TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')
conn.commit()



@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('chat'))
    else:
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('chat'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user(username)

        if user and check_password_hash(user[2], password):
            session['username'] = username  # Set username in session
            flash('Login successful!', 'success')
            return redirect(url_for('chat'))

        else:
            flash('Login failed. Please check your username and password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    # Clear the session or perform any other necessary logout logic
    session.clear()
    return redirect(url_for('login'))  # Redirect to login page or another suitable endpoint


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_input = request.form.get('user_input')
        # Process user input and generate a response (you can replace this with your own logic)
        response = generate_response(user_input)
        save_user_data(session['username'], response)
        return jsonify({'response': response})

    username = session['username']
    user_data = get_user_data_with_timestamp(username)
    return render_template('chat.html', username=username, user_data=user_data)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'username' in session:
        return redirect(url_for('chat'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            flash('Please enter both username and password.', 'danger')
        elif get_user(username):
            flash('Username already exists. Please choose another one.', 'danger')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            user_id = create_user(username, hashed_password)
            flash('Signup successful! You can now log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')


# Helper functions
def get_user(username):
    cursor.execute('SELECT * FROM users WHERE username=?', (username,))
    return cursor.fetchone()


def get_user_data_with_timestamp(username):
    user = get_user(username)
    if user:
        cursor.execute('SELECT timestamp, session_data FROM user_data WHERE user_id=?', (user[0],))
        user_data = cursor.fetchall()
        return user_data
    return None


def create_user(username, password):
    cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    return cursor.lastrowid


def save_user_data(username, session_data):
    user = get_user(username)
    if user:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('INSERT INTO user_data (user_id, session_data, timestamp) VALUES (?, ?, ?)', (user[0], session_data, timestamp))
        conn.commit()


# Define the generate_response function
def generate_response(user_input):
    # This is a simple example, you should replace it with your own logic
    if user_input.lower() == 'hello':
        return 'Hi there! How can I help you?'
    else:
        return "I'm sorry, I didn't understand that."


@app.route('/fetch_data')
def fetch_data():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    username = session['username']
    user_data = get_user_data_with_timestamp(username)
    return render_template('fetch_data.html', username=username, user_data=user_data)


def get_user_data_with_timestamp(username):
    user = get_user(username)
    if user:
        cursor.execute('SELECT timestamp, session_data FROM user_data WHERE user_id=?', (user[0],))
        user_data = cursor.fetchall()
        return user_data
    return None


def save_user_data(username, data):
    user = get_user(username)
    if user:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('INSERT INTO user_data (user_id, timestamp, data) VALUES (?, ?, ?)', (user[0], timestamp, data))
        conn.commit()

if __name__ == '__main__':
    app.run(debug=True)
