from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch  # Add this import for torch

app = Flask(__name__)

# Load GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()


@app.route('/')
def home():
    return render_template('index.html')  # Your HTML file


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    user_data = {
        'id': request.form['id'],
        'gender': request.form['gender'],
        'age': request.form['age'],
        'hypertension': request.form['hypertension'],
        'heart_disease': request.form['heart_disease'],
        'ever_married': request.form['ever_married'],
        'work_type': request.form['work_type'],
        'Residence_type': request.form['Residence_type'],
        'avg_glucose_level': request.form['avg_glucose_level'],
        'bmi': request.form['bmi'],
        'smoking_status': request.form['smoking_status'],
    }

    # Here you should load your model and make predictions based on user_data
    # For now, we'll just simulate a prediction result.
    result = "Predicted result: Low risk of stroke"

    return render_template('index.html', prediction_result=result)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json['message']

    # Tokenize the input message
    input_ids = tokenizer.encode(user_message, return_tensors='pt')

    # Generate a response from GPT-2
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1,
                                no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode the generated response
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the response as JSON
    return jsonify({'response': bot_response})


if __name__ == '__main__':
    app.run(debug=True)
