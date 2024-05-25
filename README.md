# AI-Drug-Chcek


## Prerequisites

1. **Node.js**: Required for npm (Node Package Manager) and Tailwind CSS.

2. **npm**: Comes with Node.js and is used to manage JavaScript dependencies.
   - Install with Node.js: `npm install -g npm`

3. **Python 3.8 or higher**: Required for running the Flask web framework.
  


## Installation

### Python Packages

1.  **Install Flask and related packages**:
    ```bash
    pip install Flask Flask-SQLAlchemy Flask-Migrate
    ```

### Node.js Packages

1. **Initialize npm** (if not already initialized in your project directory):
    ```bash
    npm init -y
    ```

2. **Install Tailwind CSS and related packages**:
    ```bash
    npm install -D tailwindcss autoprefixer postcss
    ```


## Usage

### Running the Application

1. **Create a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Set Environment Variables**:
    ```bash
    export FLASK_APP=app.py
    export FLASK_ENV=development
    ```

3. **Run the Flask Application**:
    ```bash
    flask run
    ```

### Building CSS with Tailwind

1. **Run the build script**:
    ```bash
    npm run build:css
    ```


