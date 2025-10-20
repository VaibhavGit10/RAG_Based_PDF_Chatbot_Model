import os
from dotenv import load_dotenv
from app import create_app

# Load environment variables from .env file
load_dotenv()

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

