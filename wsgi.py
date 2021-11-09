from color_image import app as application
from waitress import serve

if __name__ == "__main__":
    serve(application, host='0.0.0.0', port=5000)
