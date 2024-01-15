# ML-Project

- **Root directory**: This is where your project resides. It contains all other directories and files for your application. In this case, it's named `ai_dev_testing`.
  
- **App Directory**: This is where most of your Python code will live. It has several subdirectories including `blueprints`, `models`, `views`, `forms`, `static`, and `templates`. These directories are for organizing your application's components into logical categories.
  
- **Static Directory**: This contains static files such as CSS, JS, images that don't change often. It has subdirectories for each type of file (css, js, images). 

The `__init__.py` files are empty but they enable Python to recognize this directory as a package so you can import from it. Without them, you would get an error when trying to use the corresponding directories as modules.

- **Blueprints**: In Flask, blueprints are used to organise related views and other code. You can think of them as modular components for your application that can be reused across different parts of the app. This is useful when you have large applications with many routes or views. 

- **Models**: These are Python classes that define how our data should be stored in the database. They represent the tables in a relational database and include fields corresponding to the columns in those tables.

- **Views**: In Flask, these correspond to the different routes of your app, i.e., URLs of your application. Each view function (also known as route) returns some response depending on the request type (GET or POST).

- **Forms**: Forms are used for creating forms in HTML that user can fill out and submit. Flask-WTF is a library that provides integration between WTForms, a form validation toolkit for Python, and Jinja2 templates. 

- **Static**: This directory contains static files such as CSS, JS, images which do not change often (like stylesheets, scripts, fonts). The browser does not need to request these files from the server every time it loads a page. They are served by the web server directly.

- **Templates**: These contain HTML pages where we can put Jinja2 template language syntax which will be replaced with actual data when the templates get rendered in Flask. 
