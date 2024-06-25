import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from transformers import AutoProcessor, BlipForConditionalGeneration


st.title("Camera Image and Video Analysis Demo")

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Function to process image
def process_image(image):
    # Prepare inputs for the model
    inputs = processor(image, "a picture of", return_tensors="pt")
    
    # Generate the caption
    outputs = model.generate(**inputs)
    
    # Decode and return the generated caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function to process video
def process_video(video):
    frames_value = []
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video.read())
    cap = cv2.VideoCapture(tfile.name)

     # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)

    frame_count = 0
    success = True
    while success:
        # Set the position of the next frame to be captured
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)

        # Read the frame
        success, frame = cap.read()

        if success:
            # Save the frame as an image file
            temp_frame_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_frame_file.name, frame)
            image = Image.open(temp_frame_file.name)
            value = process_image(image)
            frames_value.append(value)
        frame_count += 1

    # Release the video capture object
    cap.release()
    print("Video processing complete.")
    st.text(".\n".join(frames_value))


# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image
    processed_image = process_image(image)
    
    # Display the processed image
    st.text(processed_image)

# Upload video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.video(uploaded_video)
    
    # Process the video
    processed_frame = process_video(uploaded_video)
    
    if processed_frame is not None:
        st.image(processed_frame, caption="First Frame of Video (Grayscale)", use_column_width=True, channels="GRAY")
    else:
        st.error("Error processing video")
