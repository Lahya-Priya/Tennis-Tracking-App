import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
import os
import time

# Ensure compatibility with Windows paths
pathlib.PosixPath = pathlib.WindowsPath

# Define local paths for model and repository
repo_path = 'C:/Users/lahya/OneDrive/Desktop/hello world app'
model_path = 'C:/Users/lahya/OneDrive/Desktop/hello world app/best.pt'  # Replace with your actual .pt file path

# Load the custom YOLOv5 model
model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')

# Streamlit Sidebar for fun and detailed user instructions
st.sidebar.title("🎾 Tennis Tracking Instructions 📝")
st.sidebar.info(
    """
    **Welcome to the  Tennis Player & Ball Tracker!** 😎

    *🚀 How to Get Started:*
    1. **Hit Upload!**: Choose a tennis video (MP4, AVI, MOV).
    2. **Sit Back & Relax**: Watch as the magic happens 🧙‍♂️ — players and balls detected in real-time.
    3. **Download Your Highlights**: Once the video is processed, hit **Download** and keep the action forever! 🎬
    
    *Tip*: The app runs smoothly with a solid GPU for fast processing ⚡💨. 

    Let's get started and have some tennis fun! 🎉
    """
)

# Main App UI
st.title('🎾 Tennis Tracking App')
st.write("Upload a tennis video and see the players and ball tracked in real-time. 🏃‍♀️🎾⚡️")

# File uploader for video input
uploaded_video = st.file_uploader("Choose a tennis video to upload...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Set up the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    st.write("⏳ Processing your video... Please wait. 🎥✨")

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection model (mixed precision if CUDA is available)
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                results = model(frame)
        else:
            results = model(frame)

        frame = np.squeeze(results.render())  # Draw detection boxes on the frame

        # Write frame to output video file
        out.write(frame)

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_container_width=True)

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Ensure consistent frame rate in display
        time.sleep(1 / fps)

    # Release video resources
    cap.release()
    out.release()

    st.success("🎉 Video processing complete! 🎬")

    # Provide download button for the processed video
    st.write("📥 Download your tennis video here! 🏆")
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
