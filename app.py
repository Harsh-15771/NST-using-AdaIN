import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content_image = FileField('Content Image')
    style_image = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)
model_path = os.path.join("experiment", "final_model", "decoder_final.pth")
decoder.load_state_dict(torch.load(model_path, map_location=device))
encoder.eval()
decoder.eval()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        c_feats = encoder(content_tensor, is_test=True)
        s_feats = encoder(style_tensor, is_test=True)

        t = adaptive_instance_normalization(c_feats, s_feats)
        t = alpha * t + (1 - alpha) * c_feats

        g = decoder(t)

    return g


def save_image(tensor, path):
    tensor = tensor.cpu().clamp(0, 1)
    image = transforms.ToPILImage()(tensor.squeeze(0))
    image.save(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content_image.data and form.content_image.data.filename:
            if allowed_file(form.content_image.data.filename):
                content_filename = secure_filename(form.content_image.data.filename)
                form.content_image.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename

        else:
            content_filename = form.content_path.data

        if form.style_image.data and form.style_image.data.filename:
            if allowed_file(form.style_image.data.filename):
                style_filename = secure_filename(form.style_image.data.filename)
                form.style_image.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data      

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            try:
                content_image = Image.open(content_path).convert("RGB")
                style_image = Image.open(style_path).convert("RGB")
                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filname = f"stylized_{content_filename}"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filname)
                save_image(stylized_image.cpu(), result_path)

                result_image = result_filname

            except Exception as e:
                error = f"Error processing images: {str(e)}"

    if request.method == 'POST':
        if not content_filename:
            error = "Please upload a content image."
        if not style_filename:
            error = "Please upload a style image."

    return render_template('index.html', form=form, result_image=result_image, content_image = content_filename, style_image = style_filename, error=error)

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)