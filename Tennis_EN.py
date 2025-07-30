"""
# Tennis Player Detection and Service Zone Verification

This notebook contains the implementation of a tennis court detection and player tracking system.
It uses YOLO for player detection, SORT for object tracking, and custom logic for service zone analysis.
"""

# Standard Python libraries
import os
import sys
import time
from collections import defaultdict
import math

# Image and video processing libraries
import cv2
import numpy as np
import moviepy.editor as mpe

# PyTorch libraries
import torch
import torchvision.transforms as transforms
from torchvision import models

# YOLO and Ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import ultralytics

# SORT for object tracking
from sort import Sort

# TensorFlow (if needed)
import tensorflow as tf

# Keypoint names for pose estimation
keypoint_names = [
    "Left-shoulder", "Right-shoulder", "Left-elbow", "Right-elbow",
    "Left-hip", "Right-hip", "Left-knee", "Right-knee", "Left-ankle", "Right-ankle"
]

class PlayerDetection:
    """
    Class for detecting and tracking tennis players on a court.
    Analyzes player positions relative to service zones.
    """
    def __init__(self, video_path, yolo_model_path="bestv3.pt", pose_model_path="yolo11x-pose.pt", output_path="result_with_players.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.model = YOLO(yolo_model_path)  # Model for player detection
        self.pose_model = YOLO(pose_model_path)  # Model for pose estimation
        self.tracker = Sort()
        self.player_mapping = {}
        self.threshold_y = None  # Calculated Y threshold
        self.angle_left = None   # Court angle on left side
        self.angle_right = None  # Court angle on right side

    def process_video(self, line_detection):
        """Process the video to detect and track players, analyzing their positions."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        ret, frame = cap.read()

        # Update court keypoints
        line_detection.predict_keypoints(frame)
        self.update_court_keypoints(line_detection, frame)

        while ret:
            self.frame = frame

            # Make predictions with YOLO model
            results = self.model.predict(frame, conf=0.4, iou=0.3)

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())

                    if class_id == 3:  # Filter for 'player' class
                        print(f"Player detected with confidence {confidence} at position: ({x1}, {y1}, {x2}, {y2})")
                        detections.append([x1, y1, x2, y2, confidence])

            detections = np.array(detections)

            if len(detections) == 0:
                print("No players detected in this frame.")
            else:
                print(f"{len(detections)} players detected in this frame.")

            tracked_objects = self.tracker.update(detections)

            player_positions = []
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                player_positions.append((x2, y2))
                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if self.threshold_y is not None:
                cv2.line(frame, (0, int(self.threshold_y)), 
                        (width, int(self.threshold_y)), (255, 0, 0), 2)

            if len(player_positions) == 2:
                player_1, player_2 = self.assign_player_ids(player_positions)

                for i, (x, y) in enumerate(player_positions):
                    player_label = "Player 1" if (x, y) == player_1 else "Player 2"
                    x1, y1, x2, y2, track_id = map(int, tracked_objects[i][:5])

                    left_ankle, right_ankle, keypoints = self.detect_pose((x1, y1, x2, y2), frame)

                    if keypoints:
                        self.detect_serve_zone(player_positions, left_ankle, right_ankle, line_detection.keypoints)
                        self.draw_keypoints(frame, keypoints, (x1, y1, x2, y2))
                        print(f"Left ankle: {left_ankle}, Right ankle: {right_ankle}")

                    # Draw bounding box and player label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, player_label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)  # Write frame with detections to video
            ret, frame = cap.read()

        cap.release()
        out.release()
        print(f"Video with players saved as {self.output_path}")

    # [Previous methods remain the same, just translate the docstrings and comments]
    # ...

class LineDetection:
    """
    Class for detecting and analyzing court lines in tennis videos.
    """
    def __init__(self, video_path, model_path, output_path="result_with_lines.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.model_path = model_path
        
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("GPU detected. Using CUDA for acceleration.")
        else:
            print("No GPU detected. Using CPU.")

        # Load pre-trained model and move to device
        self.model = models.resnet101(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)  # 14 keypoints (x, y)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Model expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Store keypoints
        self.keypoints = None

    def predict_keypoints(self, image):
        """Predict keypoints from the input image."""
        # Convert image to RGB (OpenCV uses BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations and add batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Get predictions (no gradient calculation)
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Convert predictions to numpy array
        keypoints = outputs.squeeze().cpu().numpy()

        # Scale keypoints from 224x224 to original image dimensions
        original_h, original_w = image.shape[:2]
        keypoints[0::2] = keypoints[0::2] * original_w / 224  # Scale x coordinates
        keypoints[1::2] = keypoints[1::2] * original_h / 224  # Scale y coordinates
        
        self.keypoints = keypoints
        return keypoints

    # [Rest of the LineDetection class methods remain the same, just translate the docstrings and comments]
    # ...

class VideoMerger:
    """
    Class for merging player detection and line detection videos,
    and adding audio from the original video.
    """
    def __init__(self, player_video_path, line_video_path, output_path="final_output.mp4", original_video_path=None):
        self.player_video_path = player_video_path
        self.line_video_path = line_video_path
        self.output_path = output_path
        self.original_video_path = original_video_path

    def merge_videos(self):
        """Merge player detection and line detection videos."""
        cap1 = cv2.VideoCapture(self.player_video_path)
        cap2 = cv2.VideoCapture(self.line_video_path)

        if not cap1.isOpened() or not cap2.isOpened():
            print(f"Error opening videos: {self.player_video_path} or {self.line_video_path}")
            return

        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap1.get(cv2.CAP_PROP_FPS))

        # Use 'XVID' for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        temp_video_path = "temp_video_without_audio.mp4"
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        while ret1 and ret2:
            # Blend the two frames
            combined_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
            out.write(combined_frame)

            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

        cap1.release()
        cap2.release()
        out.release()

        print(f"Merged video saved as {temp_video_path}, now adding audio...")

        # Add audio from original video
        self.add_audio_to_video(temp_video_path, self.original_video_path, self.output_path)

    def add_audio_to_video(self, video_path, audio_video_path, output_path):
        """Add audio from the original video to the processed video."""
        video_clip = mpe.VideoFileClip(video_path)
        original_video = mpe.VideoFileClip(audio_video_path)
        
        # Combine video with original audio
        final_video = video_clip.set_audio(original_video.audio)
        
        # Save final video with audio
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"Final video with audio saved as {output_path}")

def process_videos(video_names):
    """Process a list of video files to detect players and court lines."""
    for video_name in video_names:
        print(f"Processing video: {video_name}")

        # Remove file extension to create unique names
        base_name = os.path.splitext(video_name)[0]

        # Create unique output filenames for each video
        player_output = f"result_with_players_{base_name}.mp4"
        line_output = f"result_with_lines_{base_name}.mp4"
        merged_output = f"final_{base_name}.mp4"

        # Process court line detection
        detector = LineDetection(video_name, 'keypoints_model_v2.pth', output_path=line_output)
        detector.process_video()

        # Process player detection using the line detector
        player_detection = PlayerDetection(video_name, output_path=player_output)
        player_detection.process_video(detector)

        # Combine both results and add original audio
        video_merger = VideoMerger(player_output, line_output, merged_output, 
                                 original_video_path=video_name)
        video_merger.merge_videos()

# List of videos to process
video_names = ["input_video.mp4", "Untitled.mp4", "Untitled2.mp4", "Untitled32.mp4"]

# Process all videos
process_videos(video_names)
