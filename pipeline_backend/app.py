from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import uuid
from ml_pipeline import run_ml_pipeline, parse_s3_path
import io
import zipfile

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Upload folder configuration
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# !!! this is bucket name
S3_BUCKET = os.getenv("S3_BUCKET", "ml-pipeline-data-test-hyhen")
S3_INPUT_PREFIX = "input"
S3_MODEL_PREFIX = "models"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

emr = boto3.client('emr', region_name='us-east-1')
s3 = boto3.client(
    's3',
    region_name=AWS_REGION,
    config=boto3.session.Config(signature_version='s3v4')
)

# change upload to using generate_upload_url and then using
@app.route('/generate_upload_url', methods=['POST'])
def generate_upload_url():
    """Generate presigned URL for direct client upload"""
    try:
        filename = request.json.get('filename', '')
        if not filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files allowed"}), 400

        key = f"uploads/{uuid.uuid4()}/{filename}"

        presigned_url = s3.generate_presigned_post(
            Bucket=S3_BUCKET,
            Key=key,
            ExpiresIn=3600
        )
        return jsonify(presigned_url), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/start_pipeline', methods=['POST'])
def start_pipeline():
    """Start ML pipeline with S3 data"""
    try:
        data = request.json
        s3_path = f"s3a://{S3_BUCKET}/{data['key']}"

        # Get parameters from JSON
        feature_cols = data.get('feature_cols', [])
        label_col = data.get('label_col', '')
        data_types = data.get('data_types', {})
        train_test_split = data.get('train_test_split', 0.8)

        # Validate required parameters
        if not feature_cols:
            return jsonify({"error": "Feature columns must be specified"}), 400

        if not label_col:
            return jsonify({"error": "Label column must be specified"}), 400

        if not data_types:
            return jsonify({"error": "Data types must be specified"}), 400

        # Call ML pipeline
        result = run_ml_pipeline(s3_path, feature_cols, label_col, data_types, train_test_split)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_download_url', methods=['GET'])
def generate_download_url():
    """Generate presigned URL for model download"""
    try:
        model_path = request.args.get('model_path')
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': model_path
            },
            ExpiresIn=3600
        )
        return jsonify({"download_url": presigned_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files supported"}), 400
            
        # # Save file
        # csv_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        # file.save(csv_path)

        # Upload to S3
        file_id = str(uuid.uuid4())
        s3_key = f"{S3_INPUT_PREFIX}/{file_id}/{file.filename}"
        s3.upload_fileobj(file, S3_BUCKET, s3_key)
        s3_input_path = f"s3a://{S3_BUCKET}/{s3_key}"

        # Get parameters from form
        feature_cols = request.form.get('feature_cols', '').split(',')
        feature_cols = [col.strip() for col in feature_cols if col.strip()]
        
        label_col = request.form.get('label_col', '').strip()
        data_types_str = request.form.get('data_types', '')
        
        # Validate required parameters
        if not feature_cols:
            return jsonify({"error": "Feature columns must be specified"}), 400
            
        if not label_col:
            return jsonify({"error": "Label column must be specified"}), 400
            
        if not data_types_str:
            return jsonify({"error": "Data types must be specified"}), 400
        
        # Parse data_types JSON
        try:
            data_types = json.loads(data_types_str)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid data types format, use JSON format"}), 400
        
        # Run ML pipeline
        result = run_ml_pipeline(s3_input_path, feature_cols, label_col, data_types)

        # # Decode and save the model
        # model_base64 = result.pop("model_base64")  # Extract the Base64 string
        # model_bytes = base64.b64decode(model_base64)
        # model_dir = "/app/saved_models"
        # os.makedirs(model_dir, exist_ok=True)
        # model_path = os.path.join(model_dir, f"best_model_{result['best_model_name']}.zip")
        # with open(model_path, "wb") as f:
        #     f.write(model_bytes)
        #
        # return jsonify({
        #     **result,
        #     "saved_model_name": f"best_model_{result['best_model_name']}.zip"  # Add this
        # }), 200

        return jsonify({
            **result,
            "s3_input_path": s3_input_path
        }), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid data_types JSON"}), 400      
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# add get to download
@app.route('/download_model', methods=['GET'])
def download_model():
    try:
        model_path = request.args.get('model_path')
        bucket, key = parse_s3_path(model_path)

        # Generate presigned URL
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=3600
        )
        return jsonify({"download_url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/download_model_zip', methods=['GET'])
def download_model_zip():
    """Download model as a zip file"""
    try:
        model_path = request.args.get('model_path')
        if not model_path:
            return jsonify({"error": "Model path is required"}), 400
        
        # List all files in the model directory
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=model_path)
        
        if 'Contents' not in response or len(response['Contents']) == 0:
            return jsonify({"error": f"No files found at {model_path}"}), 404
        
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for obj in response['Contents']:
                file_key = obj['Key']
                try:
                    file_obj = s3.get_object(Bucket=S3_BUCKET, Key=file_key)
                    file_data = file_obj['Body'].read()
                    
                    # Add file to zip (remove the model_path prefix from the filename)
                    arcname = file_key.replace(model_path + '/', '')
                    if not arcname:  # Handle case where file_key is exactly model_path
                        arcname = os.path.basename(file_key)
                    zipf.writestr(arcname, file_data)
                except Exception as e:
                    print(f"Error adding {file_key} to zip: {str(e)}")
        
        zip_buffer.seek(0)
        
        # Extract model name for the download filename
        model_name = os.path.basename(model_path.rstrip('/'))
        
        # Return the zip file
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{model_name}_model.zip"
        )
        
    except Exception as e:
        print(f"Error in download_model_zip: {str(e)}")
        return jsonify({"error": str(e)}), 500

# @app.route('/download_model', methods=['GET'])
# def download_model():
#     model_name = request.args.get('model_name')
#     s3 = boto3.client('s3')
#     url = s3.generate_presigned_url(
#         'get_object',
#         Params={'Bucket': 'ml-pipeline-data-test', 'Key': f'models/{model_name}'},
#         ExpiresIn=3600
#     )
#     return jsonify({"download_url": url})

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)