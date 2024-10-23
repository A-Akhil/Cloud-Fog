import streamlit as st
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image
import io

# Import your model and utility functions
from models.gen.SPANet import Generator
from utils import heatmap

def check_cuda():
    """Check if CUDA is available and return device."""
    if torch.cuda.is_available():
        st.sidebar.success("CUDA is available! Using GPU")
        return True, torch.device('cuda')
    else:
        st.sidebar.warning("CUDA is not available. Using CPU")
        return False, torch.device('cpu')

def load_model(device):
    """Load the pretrained model."""
    try:
        gen = Generator(gpu_ids=[0] if device.type == 'cuda' else [])
        model_path = "pretrained_models/RICE1/gen_model_epoch_200.pth"
        
        if device.type == 'cuda':
            param = torch.load(model_path)
        else:
            param = torch.load(model_path, map_location=torch.device('cpu'))
            
        gen.load_state_dict(param)
        gen = gen.to(device)
        gen.eval()
        return gen
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, device, model):
    """Process the input image and return the cloud-removed result."""
    # Convert PIL Image to numpy array
    img = np.array(image).astype(np.float32)
    
    # Normalize and transpose
    img = img / 255
    img = img.transpose(2, 0, 1)
    img = img[None]
    
    with torch.no_grad():
        x = torch.from_numpy(img)
        x = x.to(device)
        
        start_time = time.time()
        att, out = model(x)
        processing_time = time.time() - start_time
        
        # Process input image
        x_ = x.cpu().numpy()[0]
        x_rgb = x_ * 255
        x_rgb = x_rgb.transpose(1, 2, 0).astype('uint8')
        
        # Process output image
        out_ = out.cpu().numpy()[0]
        out_rgb = np.clip(out_[:3], 0, 1) * 255
        out_rgb = out_rgb.transpose(1, 2, 0).astype('uint8')
        
        # Process attention map
        att_ = att.cpu().numpy()[0] * 255
        att_heatmap = heatmap(att_.astype('uint8'))[0]
        att_heatmap = att_heatmap.transpose(1, 2, 0)
        
        return x_rgb, out_rgb, att_heatmap, processing_time

def convert_to_pil(image_array):
    """Convert numpy array to PIL Image."""
    return Image.fromarray(image_array)

def main():
    st.title("Cloud Removal Application")
    st.write("Upload an image to remove clouds using deep learning")
    
    # Check CUDA availability
    cuda_available, device = check_cuda()
    
    # Load model
    model = load_model(device)
    
    if model is None:
        st.error("Failed to load the model. Please check the model path and try again.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
            # Process button
            if st.button("Remove Clouds"):
                with st.spinner("Processing..."):
                    # Process image
                    input_rgb, output_rgb, attention_map, proc_time = process_image(image, device, model)
                    
                    # Create columns for results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Input Image")
                        st.image(convert_to_pil(input_rgb), use_column_width=True)
                    
                    with col2:
                        st.subheader("Cloud-free Result")
                        st.image(convert_to_pil(output_rgb), use_column_width=True)
                    
                    with col3:
                        st.subheader("Attention Map")
                        st.image(convert_to_pil(attention_map), use_column_width=True)
                    
                    st.success(f"Processing completed in {proc_time:.2f} seconds!")
                    
                    # Add download buttons
                    output_pil = convert_to_pil(output_rgb)
                    output_bytes = io.BytesIO()
                    output_pil.save(output_bytes, format='PNG')
                    output_bytes = output_bytes.getvalue()
                    
                    st.download_button(
                        label="Download Cloud-free Image",
                        data=output_bytes,
                        file_name="cloud_free_image.png",
                        mime="image/png"
                    )
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            
    # Add sidebar information
    with st.sidebar:
        st.subheader("About")
        st.write("This application uses a deep learning model to remove clouds from satellite imagery.")
        st.write("Model: SPANet")
        st.write(f"Device being used: {device}")
        
        st.subheader("Instructions")
        st.write("1. Upload an image using the file uploader")
        st.write("2. Click 'Remove Clouds' to process the image")
        st.write("3. Download the cloud-free result if desired")

if __name__ == "__main__":
    main()