Face Recognition Attendance System using InsightFace with ArcFace
An AI-based attendance management system that automatically marks attendance using real-time face recognition. The system uses InsightFace with ArcFace for accurate facial recognition and stores attendance records securely in a database.

Project Overview
Traditional attendance systems are time-consuming and prone to proxy attendance. This project solves that problem by using computer vision and deep learning to recognize faces from live camera input and mark attendance automatically. The system works in real time and is suitable for classrooms and institutional environments.

Features
Real-time face detection and recognition
Face registration with multiple samples for higher accuracy
Face quality check for low-light and blurred faces
Multi-frame embedding aggregation
Automatic attendance marking with timestamp
SQLite database for secure storage
Supports webcam and mobile camera (DroidCam)

Technologies Used
Programming Language: Python
Face Recognition Framework: InsightFace
Recognition Model: ArcFace
Computer Vision: OpenCV
Numerical Computing: NumPy, SciPy
Database: SQLite

System Workflow
Capture live video from camera
Detect faces using InsightFace
Enhance and check face quality
Extract facial features (embeddings)
Match with registered users using cosine similarity
Mark attendance and store it in database
Display results in real time

How to Run the Project
Clone the repository: git clone https://github.com/ShreyasR001/Face-Recognition-attendance-system.git
Navigate to the project folder: cd Face-Recognition-attendance-system
Install required libraries: pip install insightface opencv-python numpy scipy
Run the application: python attendance_insightface_enhanced.py

Project Structure
├── attendance_insightface_enhanced.py
├── test_insightface.py
├── show_db.py
├── .gitignore
├── README.md

Experimental Results
Accurate recognition up to ~15 feet
Average FPS: 15–25 (CPU mode)
Works under varying lighting conditions
Prevents duplicate attendance entries

Future Enhancements
GPU acceleration for faster performance
Face anti-spoofing integration
Web-based dashboard for attendance reports
Cloud database support

👤 Author
Shreyas R
https://github.com/ShreyasR001
AI & Machine Learning Engineering Student

Acknowledgements
InsightFace open-source framework
OpenCV community
InsightFace open-source framework

OpenCV community
