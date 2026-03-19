from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import traceback
from geospatial_analysis import start_automation
from flasgger import Swagger
from dotenv import load_dotenv
import requests
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables")

@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
        return jsonify({'message': 'Failed to load dashboard'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        print("Received POST to /api/analyze")
        print(f"Form data: {dict(request.form)}")
        if 'file' in request.files:
            print(f"File: {request.files['file'].filename}")
        else:
            print("No file in request")

        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'message': 'No file uploaded or invalid file', 'status': 400}), 400
        
        file = request.files['file']
        start_date = request.form.get('startDate')
        end_date = request.form.get('endDate')
        satellite = request.form.get('satellite')
        cloud_percentage = request.form.get('cloudPercentage')
        indices = request.form.get('indices', [])

        if not all([start_date, end_date, satellite, cloud_percentage, indices]):
            print("Missing form fields")
            return jsonify({'message': 'Missing required form fields', 'status': 400}), 400

        try:
            cloud_percentage = int(cloud_percentage)
            indices = json.loads(indices)
        except ValueError as ve:
            print(f"Invalid form data: {str(ve)}")
            return jsonify({'message': f'Invalid form data: {str(ve)}', 'status': 400}), 400

        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        print(f"Saving file to: {file_path}")
        file.save(file_path)

        if not os.path.exists(file_path):
            print(f"File not saved: {file_path}")
            return jsonify({'message': 'Failed to save uploaded file', 'status': 500}), 500

        print("Calling start_automation")
        result = start_automation(
            file_path=file_path,
            start_date=start_date,
            end_date=end_date,
            satellite=satellite,
            cloud_percentage=cloud_percentage,
            indices=indices
        )

        result_json = json.loads(result)

        # Clean up uploaded file (uncomment if you want to delete after processing)
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        #     print(f"Removed file: {file_path}")

        return jsonify({"message": "Analysis complete", "data": result_json, 'status': 200}), 200
    
    except Exception as e:
        print(f"Error in /api/analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'message': f'Internal server message: {str(e)}'}), 500
    
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chatbot requests using Google's Gemini API.
    The chatbot is trained to only answer questions about geospatial analysis,
    satellite imagery, vegetation indices, and time series data.
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'message': 'No message provided', 'status': 400}), 400
        
        # Check if Gemini API key is configured
        if not GEMINI_API_KEY:
            return jsonify({
                'message': 'Gemini API key not configured. Please set GEMINI_API_KEY environment variable.',
                'status': 500
            }), 500
        
        # System prompt to constrain the AI to domain-specific topics
        system_prompt = """You are an AI assistant specialized in geospatial analytics and satellite imagery analysis. 
You can ONLY answer questions about:
- Satellite imagery (Sentinel-2, Landsat-8)
- Vegetation indices (NDVI, EVI, NDWI, SAVI)
- Cloud coverage and cloud percentage
- Time series analysis of satellite data
- Geospatial data processing
- GeoJSON files and geographic data formats
- Earth observation and remote sensing
- Image visualization and map layers
- Statistical analysis of vegetation and land cover

You are an AI assistant specialized in geospatial analysis using Google Earth Engine (GEE) and Python. 
Your task is to process AOI files (.geojson or .json), fetch Sentinel-2 or Landsat-8 imagery, apply scaling factors, calculate vegetation indices (NDVI, NDWI, SAVI, EVI), 
compute time-series statistics (mean, min, max), retrieve the latest cloud-free image clipped to AOI, and generate 
summary stats (vegetation cover %, healthy area km², total AOI area km²). You must also provide visualization URLs for RGB and each index. 
Always return results in structured JSON with fields: time_series, stats, and visualization.
Handle errors gracefully with clear messages (e.g., "Unsupported file format", "No images found for given criteria", "Failed to initialize Earth Engine"). 
Do not hallucinate—base all outputs strictly on the defined workflow.

If a user asks about something clearly outside the dashboard (e.g., personal advice, legal/medical diagnosis, unrelated external services, or requests for other websites/tools), 
politely decline to answer outside your scope. When declining, keep it short and helpful: apologize briefly, 
state your limitation, offer a small suggestion or search phrase, 
and optionally give ways to get more help (contact email, documentation link, or ask to rephrase within the dashboard). 
Always keep tone friendly and professional.

You must respond ONLY in English. Keep your answers concise, technical, and helpful."""

        try:
            # Use Gemini 1.5 Flash (free tier)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Generate response
            response = model.generate_content(
                f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": 150
                }
            )
            
            ai_response = response.text.strip()
            
            return jsonify({
                'message': 'Success',
                'response': ai_response,
                'status': 200
            }), 200
            
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return jsonify({
                'message': f'Error calling Gemini API: {str(e)}',
                'status': 500
            }), 500
            
    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'message': f'Internal server error: {str(e)}',
            'status': 500
        }), 500

# Health check endpoint for Render
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 8000))
    # Run the app - set debug=False in production
    app.run(host='0.0.0.0', port=port, debug=False)