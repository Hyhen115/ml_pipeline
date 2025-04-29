from flask import Flask, request, jsonify
import os
from ml_pipeline import run_ml_pipeline

app = Flask(__name__)

# upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save file
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(csv_path)
        
        # Get parameters from form or JSON
        feature_cols = request.form.get('feature_cols', '').split(',')
        label_col = request.form.get('label_col', '')
        data_types = request.form.get('data_types', '')
        if not feature_cols or not label_col or not data_types:
            return jsonify({"error": "Missing feature_cols, label_col, or data_types"}), 400
        
        # Parse data_types (assuming JSON string, e.g., '{"AT": "numerical", "region": "categorical"}')
        import json
        try:
            data_types = json.loads(data_types)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid data_types format"}), 400
        
        # Run ML pipeline
        result = run_ml_pipeline(csv_path, feature_cols, label_col, data_types)
        
        # Remove model object (not JSON-serializable)
        result.pop("best_model", None)
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)