import os


def create_file_folder(filename):
    path = os.path.dirname(__file__) + f"/{filename}_params_state_dict"
    if os.path.exists(path):
        (path, foldername) = os.path.split(path)
        print(f"folder: \'{foldername}\' already exist")
    else:
        os.makedirs(path)


create_file_folder("authent")
