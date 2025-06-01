##
# Smart Interview System 📋💻
#

Welcome to the **Smart Interview System**, an AI-powered platform designed to streamline the interview process for both candidates and administrators.  
This system supports scheduling, conducting, and evaluating interviews with advanced features like AI-based response analysis, object detection, and automated result tracking. 🚀

##
# Table of Contents
#

- Features
- Tech Stack
- Installation
- Usage
- API Endpoints
- File Structure
- Contributing
- Contact
- License

##
# Features ✨
#

- **Candidate Management 👤**: Candidates can log in, participate in interviews, and submit responses.  
- **Admin Panel 🖥️**: Admins can schedule interviews, manage candidate data, and view dashboards with results.  
- **Interview Types 📝**:  
  - **Level 1 (Aptitude)**: 50 MCQs with a 50-minute time limit.  
  - **Level 2 (Q&A)**: 5 AI-evaluated questions answered via speech.  
  - **Level 3 (HR)**: Google Meet with HR or automated Q&A.  
- **AI Evaluation 🧠**: Uses LLaMA model to score responses based on relevance, accuracy, and depth.  
- **Proctoring 📸**: Real-time detection of mobile phones and multiple persons using Faster R-CNN and YOLOv5 models.  
- **Email Notifications 📧**: Automated emails for interview schedules, passwords, and results.  
- **Analytics Dashboard 📊**: Visualize candidate performance with filters for date, score, and job role.  
- **Secure Authentication 🔒**: Admin and candidate login with email and password validation.  

##
# Tech Stack 🛠️
#

- **Backend**: FastAPI, Python 🐍  
- **Frontend**: Jinja2 templates, HTML, CSS  
- **AI Models**:  
  - LLaMA (via Ollama) for answer evaluation  
  - Faster R-CNN for mobile detection  
  - YOLOv5 for person detection  
- **Data Processing**: Pandas, NumPy, Plotly  
- **Computer Vision**: OpenCV, PyTorch, Torchvision  
- **Email**: SMTP (Gmail)  
- **Other Libraries**: SentenceTransformers, scikit-learn, uvicorn  

##
# Installation ⚙️
#

### Clone the Repository:
```bash
git clone https://github.com/your-username/smart-interview-system.git  
cd smart-interview-system

##
# Installation ⚙️
#

### Clone the Repository:
```bash
git clone https://github.com/your-username/smart-interview-system.git  
cd smart-interview-system
```

### Set Up a Virtual Environment:
```bash
python -m venv venv  
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Set Up Pre-trained Models:
- Download `mobile_detection_model.pth` and place it in the project root.
- Ensure YOLOv5 is accessible via `torch.hub`.

### Configure Email Settings:
- Update `sender_email` and `sender_password` in the `send_email()` function in `z8.py`, or use environment variables for better security.

### Run the Application:
```bash
python smart interview.py
```

Server will run at: `http://127.0.0.1:8004`

##
# Usage 📖
#

### Admin Access:
- Visit `/admin/login` to log in or `/admin/register` to create an account (restricted to `@vdartinc.com`).
- After logging in, go to `/admin/panel` to:
  - Schedule interviews
  - View candidates
  - Monitor real-time results
  - Access dashboards and visualizations

### Candidate Access:
- Candidates receive login credentials via email.
- They must log in at `/candidate/login` and will only be allowed to access the interview during the valid time window.
- Interview Process:
  - Level 1: Aptitude MCQs
  - Level 2: AI-graded Q&A
  - Level 3: Google Meet or simulated HR questions

### Proctoring:
- **Real-time camera access** is required for:
  - Mobile phone detection using Faster R-CNN  
  - Multiple persons detection using YOLOv5  
- **3 strikes policy**: 3 alerts = auto-termination

### Dashboard:
- Accessible to admins via `/admin/dashboard`
- Features:
  - Filters for Level, Date, Job Role
  - Visual analytics
  - Export results
  - Send selection/rejection emails

##
# API Endpoints 🔗
#

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET/POST | `/admin/login` | Admin login |
| GET/POST | `/admin/register` | Admin registration (email restricted) |
| POST | `/admin/schedule` | Schedule interviews |
| GET | `/admin/dashboard` | View analytics and results |
| POST | `/interview/detect_frame` | Mobile/multiple-person detection |
| POST | `/interview/submit` | Candidate interview submission |
| POST | `/ap1/evaluate` | Evaluate aptitude answers |
| POST | `/candidate/submit_name` | Submit candidate name |

##
# File Structure 📂
#

```bash
smart-interview-system/
├── templates/
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
├── static/
├── mobile_detection_model.pth
├── smart interview.py
├── register.json
└── README.md
```

##
# Contributing 🤝
#

1. Fork this repository  
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push the changes:
   ```bash
   git push origin feature/your-feature
   ```
5. Create a pull request  

Please follow PEP8 and document your code where necessary.

##
# Contact 📧
#

- 📬 Email: thirupugals2003@gmail.com

##
# License 📜
#

This project is licensed under the **MIT License**.  
See the `LICENSE` file for more details.

