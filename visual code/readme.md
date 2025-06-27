app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = tf.keras.models.load_model('model/poultry_model.h5')
class_names = ['Healthy', 'Avian Influenza', 'Newcastle Disease', 'Infectious Bronchitis']  # Example classes

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result, confidence = predict_disease(filepath)
    return render_template('result.html', result=result, confidence=round(confidence*100, 2))

if __name__ == '__main__':
    app.run(debug=True)


---

### 3. *templates/index.html*

html
<!DOCTYPE html>
<html>
<head>
    <title>Poultry Disease Classifier</title>
</head>
<body>
    <h1>Upload Poultry Image</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Classify">
    </form>
</body>
</html>


---

### 4. *templates/result.html*

html
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h2>Prediction: {{ result }}</h2>
    <p>Confidence: {{ confidence }}%</p>
    <a href="/">Try another</a>
</body>
</html>
