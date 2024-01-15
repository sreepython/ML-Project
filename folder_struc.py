
import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Root project directory
root = 'ai_dev_testing'
create_dir(root)

# Application directory
app_dir = f'{root}/app'
create_dir(app_dir)

folders = ['blueprints',  'models', 'views', 'forms',  'static', 'templates']
for folder in folders:
    create_dir(f'{app_dir}/{folder}')

# Static subdirectories
subfolders = ['css', 'js', 'images']
for subfolder in subfolders:
    create_dir(f'{app_dir}/static/{subfolder}')

for folder in folders + subfolders:
    print(folder)
    if folder != "static":
        init_file = f'{app_dir}/{folder}/__init__.py'
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass  # This creates the empty file