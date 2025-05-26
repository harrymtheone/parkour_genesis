import subprocess


def is_ninja_available():
    try:
        subprocess.check_output(['ninja', '--version'])
        return True
    except Exception as e:
        raise e
        return False


print(is_ninja_available())
