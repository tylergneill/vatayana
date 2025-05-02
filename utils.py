def get_app_version():
    app_version_filepath = './VERSION'
    with open(app_version_filepath, 'r', encoding='utf8') as file:
        # Assuming the __app_version__ line is the first line
        return file.readline().strip().split('=')[1].strip().replace("'", "").replace('"', '')

