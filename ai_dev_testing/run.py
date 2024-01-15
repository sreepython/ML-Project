from app.run import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

""""bash
export FLASK_APP=app.run
flask run

Or on Windows:

bash
set FLASK_APP=app.run
flask run"""
