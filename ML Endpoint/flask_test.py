from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Add this import
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for mobile app

# Environment variables for flexibility
MODEL_PATH = os.getenv('MODEL_PATH', 'best_model2.tflite')
PORT = int(os.getenv('PORT', 8080))  # Cloud Run sets PORT environment variable

# Load the TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("Model loaded successfully")
    logger.info(f"Input details: {input_details}")
    logger.info(f"Output details: {output_details}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

CLASSES = ['Blackheads', 'Cyst', 'Papules', 'Pustules', 'Whiteheads']

RECOMMENDATIONS = {
    'Blackheads': (
        "Cleanse with a salicylic acid cleanser, such as 'CeraVe Renewing SA Cleanser' or 'Acnes Natural Care Oil Control Cleanser', to help clear clogged pores. "
        "Exfoliate 2–3 times weekly with a BHA product like 'Some By Mi AHA BHA PHA 30 Days Miracle Toner' to remove excess sebum and dead skin cells. "
        "Incorporate a retinoid at night, such as 'Avoskin Miraculous Retinol Ampoule', to prevent future blackheads. "
        "Finish with a non-comedogenic moisturizer, such as 'Wardah Acnederm Day Moisturizer', to maintain skin hydration. "
        "Avoid over-exfoliating, as it can irritate the skin and worsen blackheads."
    ),
    'Cyst': (
        "Use a gentle hydrating cleanser, like 'Hada Labo Gokujyun Ultimate Moisturizing Face Wash', to prevent irritation. "
        "Spot-treat with benzoyl peroxide, such as 'Benzolac 2.5%', or sulfur-based products like 'JF Sulfur Acne Care' to reduce inflammation and bacteria. "
        "For persistent or severe cystic acne, consult a dermatologist for possible oral medications like isotretinoin or prescription-strength treatments. "
        "Avoid picking or squeezing cysts, as this can lead to scarring."
    ),
    'Papules': (
        "Wash your face with a salicylic acid-based foaming cleanser, such as 'The Tea Tree Skin Clearing Facial Wash' or 'Acnes Foaming Wash', to reduce inflammation and unclog pores. "
        "Strengthen the skin barrier using niacinamide products like 'Somethinc Niacinamide + Moisture Beet Serum'. "
        "Apply a spot treatment with benzoyl peroxide, such as 'Benzolac 2.5%', to target active papules. "
        "Introduce active ingredients slowly to prevent skin irritation."
    ),
    'Pustules': (
        "Cleanse with a tea tree oil cleanser, like 'The Tea Tree Skin Clearing Facial Wash', or a salicylic acid cleanser, such as 'Wardah Acnederm Pure Foaming Cleanser', to reduce bacteria. "
        "Use a benzoyl peroxide product, such as 'Benzolac 2.5%', as a spot treatment to combat pustules. "
        "Avoid squeezing or popping pustules to minimize scarring and further inflammation. "
        "Pair treatments with sunscreen during the day to protect healing skin."
    ),
    'Whiteheads': (
        "Start with a glycolic or salicylic acid cleanser, such as 'Safi White Expert Oil Control & Acne Cleanser', to remove excess oil and debris. "
        "Exfoliate gently with an AHA toner, like 'Avoskin Miraculous Refining Toner', to prevent the buildup of dead skin cells. "
        "Incorporate a retinoid, such as 'Somethinc Level 1% Retinol Serum', to accelerate cell turnover and prevent whiteheads. "
        "Finish with a lightweight moisturizer, such as 'Emina Ms. Pimple Acne Solution Moisturizing Gel', to keep the skin hydrated without clogging pores. "
        "Always apply sunscreen when using retinoids or exfoliating acids."
    )
}

PRODUCT_IMAGES = {
    'Blackheads': {
        'CeraVe Renewing SA Cleanser': 'https://storage.googleapis.com/acne-scan-storage/Benzolac.webp',
        'Some By Mi AHA BHA PHA 30 Days Miracle Toner': 'https://storage.googleapis.com/acne-scan-storage/Some%20By%20Mi%20AHA%20BHA%20Toner.png',
        'Avoskin Miraculous Retinol Ampoule': 'https://storage.googleapis.com/acne-scan-storage/Avoskin%20Miraculous%20Retinol%20Ampoule.jpg',
        'Wardah Acnederm Day Moisturizer': 'https://storage.googleapis.com/acne-scan-storage/Wardah%20Acnederm%20Moist.png'
    },
    'Cyst': {
        'Hada Labo Gokujyun Ultimate Moisturizing Face Wash': 'https://storage.googleapis.com/acne-scan-storage/Hada%20Labo%20Cleanser.png',
        'Benzolac 2.5%': 'https://storage.googleapis.com/acne-scan-storage/Benzolac.webp',
        'JF Sulfur Acne Care': 'https://storage.googleapis.com/acne-scan-storage/JF%20Sulvur.webp'
    },
    'Papules': {
        'The Tea Tree Skin Clearing Facial Wash': 'https://storage.googleapis.com/acne-scan-storage/Tea%20Tree%20Facial%20Wash.webp',
        'Somethinc Niacinamide + Moisture Beet Serum': 'https://storage.googleapis.com/acne-scan-storage/Somethinc%20Niacinamide%20%2B%20Moisture%20Beet%20Serum.jpg',
        'Benzolac 2.5%': 'https://storage.googleapis.com/acne-scan-storage/Benzolac.webp'
    },
    'Pustules': {
        'The Tea Tree Skin Clearing Facial Wash': 'https://storage.googleapis.com/acne-scan-storage/Tea%20Tree%20Facial%20Wash.webp',
        'Wardah Acnederm Pure Foaming Cleanser': 'https://storage.googleapis.com/acne-scan-storage/Wardah%20Acnederm%20Cleanser.webp',
        'Benzolac 2.5%': 'https://storage.googleapis.com/acne-scan-storage/Benzolac.webp'
    },
    'Whiteheads': {
        'Safi White Expert Oil Control & Acne Cleanser': 'https://storage.googleapis.com/acne-scan-storage/Safi%20Cleanser.jpg',
        'Avoskin Miraculous Refining Toner': 'https://storage.googleapis.com/acne-scan-storage/Avoskin%20AHA%20BHA%20Toner.png',
        'Somethinc Level 1% Retinol Serum': 'https://storage.googleapis.com/acne-scan-storage/Somethinc%20Retinol%20Serum.jpg',
        'Emina Ms. Pimple Acne Solution Moisturizing Gel': 'https://storage.googleapis.com/acne-scan-storage/Emina%20mspimple.jpg'
    }
}

PRODUCT_LINKS = {
    'Blackheads': {
        'CeraVe Renewing SA Cleanser': 'https://tokopedia.link/MjUiFMO64Ob',
        'Some By Mi AHA BHA PHA 30 Days Miracle Toner': 'https://tokopedia.link/oTjYmfM64Ob',
        'Avoskin Miraculous Retinol Ampoule': 'https://tokopedia.link/VWqJklS64Ob',
        'Wardah Acnederm Day Moisturizer': 'https://tokopedia.link/E4BeCWV64Ob'
    },
    'Cyst': {
        'Hada Labo Gokujyun Ultimate Moisturizing Face Wash': 'https://tokopedia.link/30trXvY64Ob',
        'Benzolac 2.5%': 'https://tokopedia.link/QSHSEQ064Ob',
        'JF Sulfur Acne Care': 'https://tokopedia.link/lbGjE5764Ob'
    },
    'Papules': {
        'The Tea Tree Skin Clearing Facial Wash': 'https://tokopedia.link/zJSvjWc74Ob',
        'Somethinc Niacinamide + Moisture Beet Serum': 'https://tokopedia.link/akbRkYi74Ob',
        'Benzolac 2.5%': 'https://tokopedia.link/QSHSEQ064Ob'
    },
    'Pustules': {
        'The Tea Tree Skin Clearing Facial Wash': 'https://tokopedia.link/zJSvjWc74Ob',
        'Wardah Acnederm Pure Foaming Cleanser': 'https://tokopedia.link/l1fXnVo74Ob',
        'Benzolac 2.5%': 'https://tokopedia.link/QSHSEQ064Ob'
    },
    'Whiteheads': {
        'Safi White Expert Oil Control & Acne Cleanser': 'https://tokopedia.link/XkwFZjt74Ob',
        'Avoskin Miraculous Refining Toner': 'https://tokopedia.link/VWqJklS64Ob',
        'Somethinc Level 1% Retinol Serum': 'https://tokopedia.link/EeNh0pD74Ob',
        'Emina Ms. Pimple Acne Solution Moisturizing Gel': 'https://tokopedia.link/jTMEMKI74Ob'
    }
}



def preprocess_image(image):
    """Preprocess the image to match model input requirements"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def predict_image(image_array):
    """Make prediction using TFLite model"""
    try:
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data[0]
    except Exception as e:
        logger.error(f"Error in predict_image: {str(e)}")
        raise


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug("Received prediction request")

        # Validate request
        if not request.files and not request.json:
            return jsonify({
                'success': False, 
                'error': 'No image provided'
            }), 400

        # Load image from request
        if 'file' in request.files:
            logger.debug("Processing uploaded file")
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'success': False, 
                    'error': 'No selected file'
                }), 400
            
            image = Image.open(file)
        elif request.json and 'image' in request.json:
            logger.debug("Processing base64 image")
            try:
                image_data = request.json['image']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            except Exception as e:
                return jsonify({
                    'success': False, 
                    'error': f'Invalid image data: {str(e)}'
                }), 400
        else:
            return jsonify({
                'success': False, 
                'error': 'No image provided'
            }), 400

        processed_image = preprocess_image(image)
        predictions = predict_image(processed_image)

        top_prediction_idx = np.argmax(predictions)
        top_prediction = CLASSES[top_prediction_idx]
        confidence = float(predictions[top_prediction_idx])

        recommendation = RECOMMENDATIONS.get(top_prediction, "No recommendation available")
        product_images = PRODUCT_IMAGES.get(top_prediction, {})
        product_links = PRODUCT_LINKS.get(top_prediction, {})

        all_predictions = [
            {'class': CLASSES[idx], 'confidence': float(pred)}
            for idx, pred in enumerate(predictions)
        ]
        all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)

        response = {
            'success': True,
            'prediction': top_prediction,
            'confidence': confidence,
            'recommendation': recommendation,
            'product_images': product_images,
            'product_links': product_links,
            'all_predictions': all_predictions
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
