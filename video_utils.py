import cv2
import os
from moviepy.editor import ImageSequenceClip

def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        i += 1
    cap.release()
    return frames, fps


def combine_frames_to_video(frames_dir, output_path, fps):
    frames = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    )
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False)
