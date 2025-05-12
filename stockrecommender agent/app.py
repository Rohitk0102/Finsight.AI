import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import json
import re # Import regular expressions for parsing

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Directly assigning your Gemini API key ---
# WARNING: Hardcoding API keys is a security risk.
GEMINI_API_KEY = "AIzaSyCdXSnAiUrfA1Eas4DPMppGvFbFbL1PSvY"
# ---------------------------------------------f

# Configure Gemini API
model = None # Initialize model variable
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # --- USE 'gemini-1.5-flash-latest' ---
    # This is the standard name for the latest Flash model.
    model_name_to_use = 'gemini-1.5-flash-latest'
    model = genai.GenerativeModel(model_name_to_use)
    # ------------------------------------
    logging.info(f"Gemini API configured and model '{model.model_name}' initialized successfully.") # Log which model is used
except ValueError as ve:
    logging.critical(f"Invalid Gemini API Key format or value: {ve}")
    model = None # Ensure model is None on error
except Exception as e:
    logging.critical(f"Failed to configure Gemini API or initialize model '{model_name_to_use}': {e}")
    model = None # Ensure model is None on error

# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML form page."""
    template_path = os.path.join(app.template_folder, 'index.html')
    if not os.path.exists(template_path):
        logging.error(f"Template file not found at: {template_path}")
        return "Error: index.html template not found in 'templates' folder.", 404
    return render_template('index.html')

# Modified /recommend endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    """Receives preferences, gets recommendations from LLM, returns structured data."""
    if not model: # Check if model was initialized successfully earlier
        return jsonify({"error": "LLM Model not initialized. Check API Key and configuration."}), 500

    try:
        # Get data from the submitted form
        investment_goal = request.form.get('investment_goal', 'Any')
        risk_tolerance = request.form.get('risk_tolerance', '5') # Default to moderate
        preferred_sector = request.form.get('preferred_sector', 'Any')
        time_horizon = request.form.get('time_horizon', 'Any')
        esg_importance = request.form.get('esg_importance', '5') # Default to moderate

        logging.info(f"--- Recommendation Request Received ---")
        logging.info(f"  Goal: {investment_goal}, Risk: {risk_tolerance}, Sector: {preferred_sector}, Horizon: {time_horizon}, ESG: {esg_importance}")
        logging.info(f"------------------------------------")

        # --- Construct Prompt for LLM ---
        # (Prompt remains the same as before)
        prompt = f"""
        Act as a financial analyst. Based on the following user preferences:
        - Investment Goal: {investment_goal}
        - Risk Tolerance: {risk_tolerance}/10 (1=low, 10=high)
        - Preferred Sector: {preferred_sector}
        - Investment Time Horizon: {time_horizon}
        - ESG Importance: {esg_importance}/10 (1=low, 10=high)

        Please recommend exactly 3 stocks listed on major US exchanges (like NYSE or NASDAQ).
        For each stock, provide:
        1. Ticker Symbol (e.g., AAPL)
        2. Company Name (e.g., Apple Inc.)
        3. A brief description (2-3 sentences) explaining why it fits the preferences.

        Format the output clearly for parsing, like this example:
        Ticker: AAPL
        Name: Apple Inc.
        Description: [Brief description here]
        ---
        Ticker: MSFT
        Name: Microsoft Corporation
        Description: [Brief description here]
        ---
        Ticker: JNJ
        Name: Johnson & Johnson
        Description: [Brief description here]
        """

        # --- Call the LLM ---
        logging.info(f"Sending request to Gemini model ({model.model_name})...")
        response = model.generate_content(prompt)
        llm_text_response = response.text
        logging.info(f"Received response from Gemini:\n{llm_text_response}")

        # --- Parse the LLM Response ---
        recommendations = []
        stock_blocks = re.findall(r"Ticker:\s*(.*?)\s*Name:\s*(.*?)\s*Description:\s*(.*?)(?:\n---|\Z)", llm_text_response, re.DOTALL | re.IGNORECASE)

        if not stock_blocks:
            logging.warning("Could not parse stock recommendations from LLM response.")
            return jsonify({"error": "Could not parse recommendations from AI.", "raw_response": llm_text_response}), 500

        for block in stock_blocks:
            ticker, name, description = block
            recommendations.append({
                "ticker": ticker.strip(),
                "name": name.strip(),
                "description": description.strip(),
                # --- Placeholder Financial Data ---
                "match_percent": f"{100 + int(risk_tolerance) * 3}%",
                "price": f"${float(hash(ticker) % 500 + 50):.2f}",
                "pe_ratio": f"{float(hash(ticker) % 40 + 15):.1f}",
                "dividend_yield": f"{float(hash(ticker) % 30 / 10):.2f}%"
                # --- End Placeholder Data ---
            })
            if len(recommendations) >= 3: # Limit to 3 recommendations
                break

        if not recommendations:
            return jsonify({"error": "AI did not provide recommendations in the expected format.", "raw_response": llm_text_response}), 500

        # Return the structured data
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        logging.error(f"Error processing /recommend request: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred processing your request."}), 500

# ==============================================================
# == END OF /recommend ENDPOINT ==
# ==============================================================

if __name__ == '__main__':
    print("\nStarting Flask app for LLM-Based Recommendations...")
    print(f"Gemini Key Configured: {'Yes (Hardcoded)' if 'GEMINI_API_KEY' in globals() and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY' else 'No / Placeholder'}")
    print(f"Gemini Model Initialized: {'Yes (' + model.model_name + ')' if model else 'No! Check API Key/Config!'}") # Show initialized model name
    print("\nFlask server running. Open http://127.0.0.1:5000 in your browser.")
    print("Press CTRL+C to quit.\n")
    # Change host to '0.0.0.0'
    app.run(debug=True, host='0.0.0.0', port=5000)