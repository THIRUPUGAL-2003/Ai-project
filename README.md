Smart Interview System 📋💻
Welcome to the Smart Interview System, an AI-powered platform designed to streamline the interview process for both candidates and administrators. This system supports scheduling, conducting, and evaluating interviews with advanced features like AI-based response analysis, object detection, and automated result tracking. 🚀

Table of Contents
Features
Tech Stack
Installation
Usage
API Endpoints
File Structure
Contributing
Contact
License
Features ✨
Candidate Management 👤: Candidates can log in, participate in interviews, and submit responses.
Admin Panel 🖥️: Admins can schedule interviews, manage candidate data, and view dashboards with results.
Interview Types 📝:
Level 1 (Aptitude): 50 MCQs with a 50-minute time limit.
Level 2 (Q&A): 5 AI-evaluated questions answered via speech.
Level 3 (HR): Google Meet with HR or automated Q&A.
AI Evaluation 🧠: Uses LLaMA model to score responses based on relevance, accuracy, and depth.
Proctoring 📸: Real-time detection of mobile phones and multiple persons using Faster R-CNN and YOLOv5 models.
Email Notifications 📧: Automated emails for interview schedules, passwords, and results.
Analytics Dashboard 📊: Visualize candidate performance with filters for date, score, and job role.
Secure Authentication 🔒: Admin and candidate login with email and password validation.
Tech Stack 🛠️
Backend: FastAPI, Python 🐍
Frontend: Jinja2 templates, HTML, CSS
AI Models:
LLaMA (via Ollama) for answer evaluation
Faster R-CNN for mobile detection
YOLOv5 for person detection
Data Processing: Pandas, NumPy, Plotly
Computer Vision: OpenCV, PyTorch, Torchvision
Email: SMTP (Gmail)
Other Libraries: SentenceTransformers, scikit-learn, uvicorn
Installation ⚙️
Clone the Repository:
bash

Copy
git clone https://github.com/your-username/smart-interview-system.git
cd smart-interview-system
Set Up a Virtual Environment:
bash

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
bash

Copy
pip install -r requirements.txt
Ensure you have a requirements.txt file with all dependencies (e.g., fastapi, uvicorn, torch, opencv-python, etc.).
Set Up Pre-trained Models:
Download the pre-trained mobile_detection_model.pth and place it in the project root.
Ensure YOLOv5 is accessible via torch.hub.
Configure Email Settings:
Update the sender_email and sender_password in the send_email function or use environment variables for security.
Run the Application:
bash

Copy
python z8.py
The server will start at http://127.0.0.1:8004.
Usage 📖
Admin Access:
Navigate to /admin/login to log in or register.
Use the admin panel (/admin/panel) to schedule interviews and view dashboards.
Candidate Access:
Candidates receive login credentials via email.
Log in at /candidate/login within the 5-minute pre-interview window.
Complete the interview (Aptitude, Q&A, or HR) and submit results.
Proctoring:
Ensure a working camera and microphone.
The system detects mobile phones or multiple persons, issuing warnings (3 warnings lead to termination).
Dashboard:
Admins can filter results by level, date, score, or job role at /admin/dashboard.
Download results or send selection/rejection emails.
API Endpoints 🔗
GET /: Home page with role selection.
GET/POST /admin/login: Admin login.
GET/POST /admin/register: Admin registration (requires @vdartinc.com email).
POST /admin/schedule: Schedule interviews for candidates.
GET /admin/dashboard: View filtered interview results.
POST /interview/detect_frame: Detect mobile phones or multiple persons in video frames.
POST /interview/submit: Submit interview responses for evaluation.
POST /ap1/evaluate: Evaluate aptitude test answers.
POST /candidate/submit_name: Submit candidate name post-interview.
File Structure 📂
text

Copy
smart-interview-system/
├── templates/                # Jinja2 HTML templates
│   ├── index.html
│   ├── admin_login.html
│   ├── admin_register.html
│   ├── admin_panel.html
│   ├── candidate_login.html
│   ├── interview.html
│   ├── aptitude.html
│   ├── admin_dashboard.html
│   ├── visual_dashboard.html
│   └── candidate_submit_name.html
├── static/                   # Static files (CSS, JS, etc.)
├── generated_questions/      # Cached Q&A pairs for Level 2
├── interview_video/          # Stored candidate video recordings
├── mobile_detection_model.pth # Pre-trained mobile detection model
├── z8.py                     # Main application code
├── register.json             # Admin credentials
├── interview_results.xlsx    # Level 2 interview results
├── aptitude_results.xlsx     # Level 1 aptitude results
├── hr_result.xlsx            # Level 3 HR results
└── README.md                 # This file
Contributing 🤝
Contributions are welcome! 🙌

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.
Please ensure code follows PEP 8 standards and includes appropriate tests.

Contact 📧
For questions or support, reach out to:

Email: 125150054@sastra.ac.in
License 📜
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to adjust the GitHub repository URL (https://github.com/your-username/smart-interview-system.git) to match your actual repository. Let me know if you need help creating the requirements.txt or any other modifications! 🚀
