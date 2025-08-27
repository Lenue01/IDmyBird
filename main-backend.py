from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from torchvision import transforms
from PIL import Image
import traceback

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})  #ensure is running on same port as the react front end


UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'static/models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload and model directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

#load the model from the static folder
model_path = os.path.join(MODEL_FOLDER, 'bird_model_v2.pt')
try:
    from torchvision.models import resnet
    torch.serialization.add_safe_globals([resnet.ResNet])
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#class names
class_names = [ "Black footed Albatross", "Laysan Albatross", "Sooty Albatross", "Groove billed Ani", "Crested Auklet", "Least Auklet", "Parakeet Auklet", "Rhinoceros Auklet", "Brewer Blackbird", "Red winged Blackbird", "Rusty Blackbird", "Yellow headed Blackbird", "Bobolink", "Indigo Bunting", "Lazuli Bunting", "Painted Bunting", "Cardinal", "Spotted Catbird", "Gray Catbird", "Yellow breasted Chat", "Eastern Towhee", "Chuck will Widow", "Brandt Cormorant", "Red faced Cormorant", "Pelagic Cormorant", "Bronzed Cowbird", "Shiny Cowbird", "Brown Creeper", "American Crow", "Fish Crow", "Black billed Cuckoo", "Mangrove Cuckoo", "Yellow billed Cuckoo", "Gray crowned Rosy Finch", "Purple Finch", "Northern Flicker", "Acadian Flycatcher", "Great Crested Flycatcher", "Least Flycatcher", "Olive sided Flycatcher", "Scissor tailed Flycatcher", "Vermilion Flycatcher", "Yellow bellied Flycatcher", "Frigatebird", "Northern Fulmar", "Gadwall", "American Goldfinch", "European Goldfinch", "Boat tailed Grackle", "Eared Grebe", "Horned Grebe", "Pied billed Grebe", "Western Grebe", "Blue Grosbeak", "Evening Grosbeak", "Pine Grosbeak", "Rose breasted Grosbeak", "Pigeon Guillemot", "California Gull", "Glaucous winged Gull", "Heermann Gull", "Herring Gull", "Ivory Gull", "Ring billed Gull", "Slaty backed Gull", "Western Gull", "Anna Hummingbird", "Ruby throated Hummingbird", "Rufous Hummingbird", "Green Violetear", "Long tailed Jaeger", "Pomarine Jaeger", "Blue Jay", "Florida Jay", "Green Jay", "Dark eyed Junco", "Tropical Kingbird", "Gray Kingbird", "Belted Kingfisher", "Green Kingfisher", "Pied Kingfisher", "Ringed Kingfisher", "White breasted Kingfisher", "Red legged Kittiwake", "Horned Lark", "Pacific Loon", "Mallard", "Western Meadowlark", "Hooded Merganser", "Red breasted Merganser", "Mockingbird", "Nighthawk", "Clark Nutcracker", "White breasted Nuthatch", "Baltimore Oriole", "Hooded Oriole", "Orchard Oriole", "Scott Oriole", "Ovenbird", "Brown Pelican", "White Pelican", "Western Wood Pewee", "Sayornis", "American Pipit", "Whip poor Will", "Horned Puffin", "Common Raven", "White necked Raven", "American Redstart", "Geococcyx", "Loggerhead Shrike", "Great Grey Shrike", "Baird Sparrow", "Black throated Sparrow", "Brewer Sparrow", "Chipping Sparrow", "Clay colored Sparrow", "House Sparrow", "Field Sparrow", "Fox Sparrow", "Grasshopper Sparrow", "Harris Sparrow", "Henslow Sparrow", "Le Conte Sparrow", "Lincoln Sparrow", "Nelson Sharp tailed Sparrow", "Savannah Sparrow", "Seaside Sparrow", "Song Sparrow", "Tree Sparrow", "Vesper Sparrow", "White crowned Sparrow", "White throated Sparrow", "Cape Glossy Starling", "Bank Swallow", "Barn Swallow", "Cliff Swallow", "Tree Swallow", "Scarlet Tanager", "Summer Tanager", "Artic Tern", "Black Tern", "Caspian Tern", "Common Tern", "Elegant Tern", "Forsters Tern", "Least Tern", "Green tailed Towhee", "Brown Thrasher", "Sage Thrasher", "Black capped Vireo", "Blue headed Vireo", "Philadelphia Vireo", "Red eyed Vireo", "Warbling Vireo", "White eyed Vireo", "Yellow throated Vireo", "Bay breasted Warbler", "Black and white Warbler", "Black throated Blue Warbler", "Blue winged Warbler", "Canada Warbler", "Cape May Warbler", "Cerulean Warbler", "Chestnut sided Warbler", "Golden winged Warbler", "Hooded Warbler", "Kentucky Warbler", "Magnolia Warbler", "Mourning Warbler", "Myrtle Warbler", "Nashville Warbler", "Orange crowned Warbler", "Palm Warbler", "Pine Warbler", "Prairie Warbler", "Prothonotary Warbler", "Swainson Warbler", "Tennessee Warbler", "Wilson Warbler", "Worm eating Warbler", "Yellow Warbler", "Northern Waterthrush", "Louisiana Waterthrush", "Bohemian Waxwing", "Cedar Waxwing", "American Three toed Woodpecker", "Pileated Woodpecker", "Red bellied Woodpecker", "Red cockaded Woodpecker", "Red headed Woodpecker", "Downy Woodpecker", "Bewick Wren", "Cactus Wren", "Carolina Wren", "House Wren", "Marsh Wren", "Rock Wren", "Winter Wren", "Common Yellowthroat" ]



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Bird Classification API'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

    try:
        # Save the uploaded file
        filename = file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            confidence = probabilities[0][predicted].item()

        # Get prediction results
        predicted_class = class_names[predicted.item()]
        confidence_percentage = f"{confidence * 100:.2f}%"

        # Clean up uploaded file
        os.remove(image_path)

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence_percentage,
            'image_path': filename
        })



    except Exception as e:
        traceback.print_exc()
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)