Driver Drowsiness Detection System
AI-powered real-time driver drowsiness detection using MediaPipe and Computer Vision
📋 Description
A real-time drowsiness detection system that monitors a driver's eye activity through a webcam using MediaPipe's 468-point facial landmark detection. The system computes the Eye Aspect Ratio (EAR) to detect signs of drowsiness and triggers an audible alarm before an accident can occur — providing a non-intrusive, computer vision-powered safety solution.
🚨 Why It Matters

20% of fatal crashes involve drowsy driving
1 in 25 adults fall asleep at the wheel monthly
1,550+ lives are lost annually due to drowsy driving

⚙️ How It Works
The system uses the EAR algorithm:
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
Six facial landmarks are tracked per eye. When EAR drops below 0.25 for 20 consecutive frames, a drowsiness alert is triggered.
🔄 Pipeline
Webcam Input → Frame Processing (BGR→RGB) → Face Landmarker (468 pts) → EAR Calculation → ALARM
✨ Features

Live green eye contour overlay using OpenCV polylines
Real-time EAR score displayed on screen
Red alert banner after 20 consecutive low-EAR frames
Threaded audio alarm at 2500Hz
Auto-reset when EAR normalizes
Counter resets when no face is detected

🛠️ Tech Stack
CategoryToolsLanguagePython 3.xCV LibraryOpenCVFace DetectionMediaPipeMath/DistanceNumPy, SciPyAudiowinsound (Windows)
💻 Hardware Requirements

Processor: Intel i5+
Camera: HD Webcam
RAM: 4GB minimum

📊 Performance

Processing Speed: ~30 FPS
Alert Latency: 0.67s
MediaPipe Landmark Accuracy: 99%+
