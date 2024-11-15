import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
import os
import requests

# URL for the YOLO model (not used in this code since we're loading locally)
model_url = 'https://github.com/Lahya-Priya/tennis_detection/blob/main/best.pt'

# Ensure compatibility with Windows paths
pathlib.PosixPath = pathlib.WindowsPath

# Define local paths for model and repository
repo_path = 'C:/Users/lahya/OneDrive/Desktop/hello world app'
model_path = 'C:/Users/lahya/OneDrive/Desktop/hello world app/best.pt'  # Replace with your actual .pt file path

# Load the custom YOLOv5 model
model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')

# Streamlit app UI
st.title('🎾 Tennis Tracking App')
st.write('Upload a tennis video to detect players in real-time.')

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path = str(temp_video.name)
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Create a temporary file for the output video
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = output_temp.name
    output_temp.close()  # Close the temp file so it can be used by VideoWriter

    # Prepare VideoWriter to save the processed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        frame = np.squeeze(results.render())  # Draw detection boxes on the frame

        # Write frame to output video file
        out.write(frame)

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_container_width=True)

    # Release video resources
    cap.release()
    out.release()  # Ensure VideoWriter is released before accessing output video file

    st.success('Video processing complete!')

    # Provide a download button for the processed video
    with open(output_video_path, 'rb') as f:
        st.download_button(
            label="⬇ Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)

st.write("Ensure 'best.pt' is in the same directory or provide the correct path in model_path.")