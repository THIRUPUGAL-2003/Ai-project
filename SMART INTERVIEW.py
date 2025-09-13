import os
import uvicorn
import json
import random
import re
import time
import base64
import asyncio
import io
import torch
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import string
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File, Response, Query, APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse, FileResponse  
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from email.mime.text import MIMEText
import plotly
import smtplib
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from datetime import datetime as dt
import hashlib
import ollama

# Helper Function: Email Sending
def send_email(to_address, subject, message):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "125150054@sastra.ac.in"
    sender_password = "ksdd eqpj rtqv ytfz"  # Use environment variables in production!
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_address
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [to_address], msg.as_string())
        server.quit()
        print(f"Email successfully sent to {to_address}")
    except Exception as e:
        print(f"Failed to send email to {to_address}: {e}")

# Helper Function: Generate Timetable
def generate_timetable(candidate_emails: List[str], from_time: str, to_time: str, interview_date: str) -> tuple[str, List[dict]]:
    try:
        start_datetime = datetime.strptime(f"{interview_date} {from_time}", "%Y-%m-%d %H:%M")
        end_datetime = datetime.strptime(f"{interview_date} {to_time}", "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date or time format")
    
    total_duration = (end_datetime - start_datetime).total_seconds() / 60
    if total_duration <= 0:
        raise HTTPException(status_code=400, detail="To time must be after from time")
    
    num_candidates = len(candidate_emails)
    slot_duration = total_duration // num_candidates
    
    timetable = []
    current_time = start_datetime
    for i, email in enumerate(candidate_emails):
        slot_start = current_time
        slot_end = current_time + timedelta(minutes=slot_duration)
        timetable.append({
            "candidate_email": email,
            "start_time": slot_start.strftime("%H:%M"),
            "end_time": slot_end.strftime("%H:%M"),
            "date": interview_date
        })
        current_time = slot_end
    
    table_text = "Candidate Email       | Interview Date | Start Time | End Time\n"
    table_text += "---------------------|----------------|------------|---------\n"
    for slot in timetable:
        email = slot['candidate_email']
        date = slot['date']
        start = slot['start_time']
        end = slot['end_time']
        table_text += f"{email:<20} | {date:<14} | {start:<10} | {end}\n"
    
    return table_text, timetable

# Helper Function: Load HR Questions
def load_hr_questions():
    with open("HR.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    context = data["data"][0]["paragraphs"][0]["context"]
    questions = [q.strip() for q in context.split("\n") if q.strip()]
    return questions
# Helper Function: Evaluate Answer with Llama Model
import ollama

def evaluate_answer_with_llama(question: str, candidate_answer: str) -> float:
    """
    Evaluates a candidate's answer to a question using the Llama model and returns a score between 0 and 5.
    
    Args:
        question (str): The question asked to the candidate.
        candidate_answer (str): The candidate's response to the question.
    
    Returns:
        float: A score between 0.0 and 5.0 reflecting the quality and relevance of the answer.
    """
    # Construct the prompt for the Llama model with additional conditions
    prompt = f"""
Evaluate the following answer to the question and provide a score out of 5 based on the following criteria:
1. **Relevance**: If the candidate repeats the question or provides an answer unrelated to the question, score 0.
2. **Accuracy**: If the answer contains factually incorrect information, reduce the score significantly (e.g., 0 to 1).
3. **Completeness**: If the answer partially addresses the question, score between 1 and 3 based on how much is covered.
4. **Clarity**: If the answer is vague, poorly structured, or confusing, reduce the score by up to 1 point.
5. **Depth**: If the answer provides insightful details, examples, or reasoning relevant to the question, increase the score up to 5.
6. **Conciseness**: If the answer is overly verbose or includes irrelevant details, reduce the score by up to 0.5 points.
7. **Originality**: If the answer demonstrates unique perspectives or creative approaches (without deviating from relevance), increase the score by up to 0.5 points.
8. **Engagement**: If the answer is compelling, well-phrased, or maintains the reader's interest, increase the score by up to 0.5 points.
9. **Contextual Appropriateness**: If the answer uses language or tone inappropriate for the question's context (e.g., overly casual for a formal question), reduce the score by up to 0.5 points.
10. **Specificity**: If the answer is generic or lacks precise details when the question demands them, reduce the score by up to 1 point.
11. **Alignment with Intent**: If the answer misinterprets the question's intent (e.g., answering a 'why' question with a 'what'), reduce the score by up to 1 point.
12. **Medium-level answer**: A clear, partially complete answer with minor issues should score around 3 to 4.
13. **Excellent answer**: A fully relevant, accurate, clear, concise, insightful, and engaging answer that aligns perfectly with the question's intent should score 5.

Respond with only the numerical score (e.g., 3.7).

Question: {question}
Answer: {candidate_answer}

Score:
"""
    
    # Send the prompt to the Llama model and get the response
    response = ollama.generate(model="llama3.2:1b", prompt=prompt)
    
    # Extract and process the score from the response
    try:
        score_text = response["response"].strip()
        score = float(score_text)
        # Clamp the score to ensure it’s between 0 and 5
        return min(max(score, 0.0), 5.0)
    except (KeyError, ValueError):
        # Return 0.0 if the response is invalid or cannot be converted to a float
        return 0.0

# Helper Functions for Admin Registration
def load_registered_admins():
    if os.path.exists("register.json"):
        with open("register.json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        default_admins = []
        with open("register.json", "w", encoding="utf-8") as f:
            json.dump(default_admins, f, ensure_ascii=False, indent=4)
        return default_admins

def save_registered_admins(admins):
    with open("register.json", "w", encoding="utf-8") as f:
        json.dump(admins, f, ensure_ascii=False, indent=4)

# Configuration and Template Setup
TEMPLATE_DIR = "templates"
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)

templates_data = {
    "index.html": """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>💻 Smart Interview System</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(135deg, #e3f2fd, #bbdefb);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      margin: 0;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .hero {
      text-align: center;
      padding: 40px;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 15px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
      max-width: 1200px;
      width: 100%;
      animation: fadeIn 1s ease-in;
    }
    .hero h1 {
      font-size: 2.5rem;
      color: #0d47a1;
      margin-bottom: 20px;
    }
    .hero p {
      font-size: 1.2rem;
      color: #1565c0;
      margin-bottom: 30px;
    }
    .btn-custom {
      padding: 12px 30px;
      font-size: 1.1rem;
      border-radius: 25px;
      transition: transform 0.3s, box-shadow 0.3s;
    }
    .btn-custom:hover {
      transform: translateY(-3px);
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    .modal-content {
      border-radius: 15px;
    }
    .modal-header {
      background: #1976d2;
      color: white;
      border-top-left-radius: 15px;
      border-top-right-radius: 15px;
    }
    .modal-body {
      max-height: 60vh;
      overflow-y: auto;
    }
    .modal-footer {
      justify-content: center;
    }
    .btn-submit {
      background: #1976d2;
      border: none;
      padding: 10px 20px;
      border-radius: 10px;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-submit:hover {
      background: #1565c0;
      transform: translateY(-2px);
    }
    .btn-submit:disabled {
      background: #6c757d;
      cursor: not-allowed;
    }
    .how-it-works {
      margin-top: 30px;
      background: rgba(255, 255, 255, 0.95);
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    .how-it-works h3 {
      font-size: 1.5rem;
      color: #37474f;
      margin-bottom: 20px;
      text-align: center;
    }
    .steps-container {
      display: flex;
      justify-content: space-between;
      flex-wrap: nowrap;
      gap: 10px;
    }
    .step {
      flex: 1;
      text-align: center;
      padding: 10px;
      min-width: 150px;
    }
    .step-icon {
      font-size: 2rem;
      color: #1976d2;
      margin-bottom: 10px;
    }
    .step-text h5 {
      font-size: 1rem;
      color: #37474f;
      margin: 0 0 5px;
    }
    .step-text p {
      font-size: 0.85rem;
      color: #455a64;
      margin: 0;
    }
    @media (max-width: 768px) {
      .steps-container {
        flex-wrap: wrap;
      }
      .step {
        min-width: 45%;
      }
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="hero">
    <h1>💻 Smart Interview System</h1>
    <p>Choose your role to begin</p>
    <button class="btn btn-primary btn-custom mx-2" data-bs-toggle="modal" data-bs-target="#termsModal">Candidate Login</button>
    <a href="/admin/login" class="btn btn-secondary btn-custom mx-2">Admin Login</a>
    <div class="how-it-works">
      <h3>How It Works</h3>
      <div class="steps-container">
        <div class="step">
          <i class="bi bi-calendar3 step-icon"></i>
          <div class="step-text">
            <h5>📅 Scheduling</h5>
            <p>Admins schedule interviews and notify candidates via email.</p>
          </div>
        </div>
        <div class="step">
          <i class="bi bi-camera-video step-icon"></i>
          <div class="step-text">
            <h5>🎥 Interview Levels</h5>
            <p>Candidates complete aptitude, Q&A, or HR interviews with AI monitoring.</p>
          </div>
        </div>
        <div class="step">
          <i class="bi bi-cpu step-icon"></i>
          <div class="step-text">
            <h5>🧠 Evaluation</h5>
            <p>Responses are analyzed using advanced AI models.</p>
          </div>
        </div>
        <div class="step">
          <i class="bi bi-bar-chart step-icon"></i>
          <div class="step-text">
            <h5>📊 Analytics</h5>
            <p>Admins access dashboards to review performance and results.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Terms and Conditions Modal -->
  <div class="modal fade" id="termsModal" tabindex="-1" aria-labelledby="termsModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="termsModalLabel">Candidate Terms and Conditions</h5>
          <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <h6>Rules, Regulations, and Process</h6>
          <p>Welcome to the Smart Interview System. Please adhere to the following guidelines:</p>
          <ul>
            <li><strong>Login Window:</strong> You can log in from 5 minutes before to 15 minutes after your scheduled interview time. Check your email for details.</li>
            <li><strong>Credentials:</strong> Use the email and password sent to you. Contact support if you face issues.</li>
            <li><strong>Interview Levels:</strong>
              <ul>
                <li><strong>Level 1 (Aptitude):</strong> 50 multiple-choice questions, 50 minutes. Camera and microphone required for monitoring.</li>
                <li><strong>Level 2 (Q&A):</strong> 5 questions answered via speech, 30 seconds each. Answers are evaluated using AI.</li>
                <li><strong>Level 3 (HR):</strong> Either a Google Meet with HR or 5 automated Q&A questions. Follow the provided link for Google Meet.</li>
              </ul>
            </li>
            <li><strong>Prohibited Actions:</strong> Use of mobile phones or multiple persons in the frame will result in warnings. Three warnings lead to termination.</li>
            <li><strong>Technical Requirements:</strong> Ensure a stable internet connection, working camera, and microphone. Allow browser permissions for both.</li>
            <li><strong>Submission:</strong> After the interview, submit your name to save your results. Results are processed and stored securely.</li>
          </ul>
          <p>By proceeding, you agree to follow these rules, participate honestly, and accept that violations may lead to interview termination.</p>
          <div class="form-check">
            <input class="form-check-input" type="checkbox" id="agreeCheckbox">
            <label class="form-check-label" for="agreeCheckbox">
              I agree to the terms and conditions
            </label>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-submit" id="submitTerms" disabled onclick="redirectToLogin()">Submit</button>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    // Enable/disable submit button based on checkbox
    document.getElementById('agreeCheckbox').addEventListener('change', function() {
      document.getElementById('submitTerms').disabled = !this.checked;
    });

    // Redirect to candidate login page
    function redirectToLogin() {
      window.location.href = '/candidate/login';
    }
  </script>
</body>
</html>
""",
    "admin_login.html": """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin Login</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(to right, #eceff1, #cfd8dc);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .login-container {
      background: #fff;
      padding: 40px;
      border-radius: 15px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
      max-width: 450px;
      width: 100%;
      animation: slideUp 0.8s ease-out;
    }
    h2 {
      color: #37474f;
      margin-bottom: 30px;
      font-weight: 600;
    }
    .form-label {
      color: #455a64;
      font-weight: 500;
    }
    .form-control {
      border-radius: 10px;
      padding: 12px;
      transition: border-color 0.3s;
    }
    .form-control:focus {
      border-color: #1976d2;
      box-shadow: 0 0 5px rgba(25, 118, 210, 0.5);
    }
    .btn-login {
      background: #1976d2;
      border: none;
      padding: 12px;
      border-radius: 10px;
      font-size: 1.1rem;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-login:hover {
      background: #1565c0;
      transform: translateY(-2px);
    }
    .forgot-link, .register-link {
      display: block;
      text-align: center;
      margin-top: 15px;
      color: #1976d2;
      text-decoration: none;
    }
    .forgot-link:hover, .register-link:hover {
      text-decoration: underline;
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(50px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="login-container">
    <h2 class="text-center">Admin Login</h2>
    <form action="/admin/login" method="post">
      <div class="mb-4">
        <label for="email" class="form-label">Email Address</label>
        <input type="email" class="form-control" id="email" name="email" placeholder="Enter your email" required value="{{ email if email else '' }}">
      </div>
      <div class="mb-4">
        <label for="password" class="form-label">Password</label>
        <input type="password" class="form-control" id="password" name="password" placeholder="Enter your password" required>
      </div>
      <button type="submit" class="btn btn-login w-100">Login</button>
      <a href="/admin/forgot_password" class="forgot-link">Forgot Password?</a>
    </form>
    <p class="register-link">Don't have an account? <a href="/admin/register">Register here</a></p>
    {% if message %}
      <div class="alert alert-danger mt-3">
        {{ message }}
      </div>
    {% endif %}
  </div>
</body>
</html>
""",
    "admin_register.html": """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin Registration</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(to right, #eceff1, #cfd8dc);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .login-container {
      background: #fff;
      padding: 40px;
      border-radius: 15px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
      max-width: 450px;
      width: 100%;
      animation: slideUp 0.8s ease-out;
    }
    h2 {
      color: #37474f;
      margin-bottom: 30px;
      font-weight: 600;
    }
    .form-label {
      color: #455a64;
      font-weight: 500;
    }
    .form-control {
      border-radius: 10px;
      padding: 12px;
      transition: border-color 0.3s;
    }
    .form-control:focus {
      border-color: #1976d2;
      box-shadow: 0 0 5px rgba(25, 118, 210, 0.5);
    }
    .btn-login {
      background: #1976d2;
      border: none;
      padding: 12px;
      border-radius: 10px;
      font-size: 1.1rem;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-login:hover {
      background: #1565c0;
      transform: translateY(-2px);
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(50px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="login-container">
    <h2 class="text-center">Admin Registration</h2>
    <form action="/admin/register" method="post">
      <div class="mb-4">
        <label for="name" class="form-label">Name</label>
        <input type="text" class="form-control" id="name" name="name" placeholder="Enter your name" required value="{{ name if name else '' }}">
      </div>
      <div class="mb-4">
        <label for="email" class="form-label">Email Address</label>
        <input type="email" class="form-control" id="email" name="email" placeholder="Enter your email" required value="{{ email if email else '' }}">
      </div>
      <div class="mb-4">
        <label for="password" class="form-label">Password</label>
        <input type="password" class="form-control" id="password" name="password" placeholder="Enter your password" required>
      </div>
      <button type="submit" class="btn btn-login w-100">Register</button>
    </form>
    <p class="login-link mt-3 text-center">Already have an account? <a href="/admin/login">Login here</a></p>
    {% if message %}
      <div class="alert {% if message_type == 'success' %}alert-success{% else %}alert-danger{% endif %} mt-3">
        {{ message }}
        {% if message_type == 'success' %}
          <br><a href="/admin/login" class="btn btn-primary mt-2">Go to Login</a>
        {% endif %}
      </div>
    {% endif %}
  </div>
</body>
</html>
""",
    "admin_forgot_password.html": """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Forgot Password</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(to right, #eceff1, #cfd8dc);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .login-container {
      background: #fff;
      padding: 40px;
      border-radius: 15px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
      max-width: 450px;
      width: 100%;
      animation: slideUp 0.8s ease-out;
    }
    h2 {
      color: #37474f;
      margin-bottom: 30px;
      font-weight: 600;
    }
    .form-label {
      color: #455a64;
      font-weight: 500;
    }
    .form-control {
      border-radius: 10px;
      padding: 12px;
      transition: border-color 0.3s;
    }
    .form-control:focus {
      border-color: #1976d2;
      box-shadow: 0 0 5px rgba(25, 118, 210, 0.5);
    }
    .btn-login {
      background: #1976d2;
      border: none;
      padding: 12px;
      border-radius: 10px;
      font-size: 1.1rem;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-login:hover {
      background: #1565c0;
      transform: translateY(-2px);
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(50px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="login-container">
    <h2 class="text-center">Forgot Password</h2>
    <form action="/admin/forgot_password" method="post">
      <div class="mb-4">
        <label for="email" class="form-label">Email Address</label>
        <input type="email" class="form-control" id="email" name="email" placeholder="Enter your email" required value="{{ email if email else '' }}">
      </div>
      <button type="submit" class="btn btn-login w-100">Send Password</button>
    </form>
    <p class="login-link mt-3 text-center"><a href="/admin/login">Back to Login</a></p>
    {% if message %}
      <div class="alert {% if message_type == 'success' %}alert-success{% else %}alert-danger{% endif %} mt-3">
        {{ message }}
      </div>
    {% endif %}
  </div>
</body>
</html>
""",
    "admin_panel.html": """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin Panel</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: #f4f6f9;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      padding: 20px;
    }
    .panel-container {
      background: #fff;
      padding: 30px;
      border-radius: 15px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
      max-width: 800px;
      margin: 0 auto;
      animation: fadeIn 0.8s ease-in;
    }
    h2 {
      color: #263238;
      text-align: center;
      margin-bottom: 30px;
      font-weight: 600;
    }
    .form-label {
      color: #455a64;
      font-weight: 500;
    }
    .form-control, .form-select {
      border-radius: 10px;
      padding: 12px;
      transition: border-color 0.3s;
    }
    .form-control:focus, .form-select:focus {
      border-color: #0288d1;
      box-shadow: 0 0 5px rgba(2, 136, 209, 0.5);
    }
    .btn-primary {
      background: #0288d1;
      border: none;
      padding: 12px;
      border-radius: 10px;
      font-size: 1.1rem;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-primary:hover {
      background: #0277bd;
      transform: translateY(-2px);
    }
    .btn-info {
      background: #4fc3f7;
      border: none;
      padding: 10px;
      border-radius: 10px;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-info:hover {
      background: #29b6f6;
      transform: translateY(-2px);
    }
    .btn-secondary {
      background: #6c757d;
      border: none;
      padding: 10px;
      border-radius: 10px;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-secondary:hover {
      background: #5a6268;
      transform: translateY(-2px);
    }
    .form-check-label {
      margin-left: 10px;
    }
    .invalid-feedback {
      color: #d32f2f;
      font-size: 0.9rem;
      margin-top: 5px;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  </style>
</head>
<body>
 <div class="panel-container">
  <h2>Schedule Interview</h2>
  <form action="/admin/schedule" method="post" id="schedule-form">
    <div class="mb-3">
      <label for="level" class="form-label">Interview Level</label>
      <select class="form-select" id="level" name="level" required onchange="updateFields()">
        <option value="">-- Select Level --</option>
        <option value="1" {% if level == "1" %}selected{% endif %}>Level 1 (Aptitude Test)</option>
        <option value="2" {% if level == "2" %}selected{% endif %}>Level 2 (Q&A Interview)</option>
        <option value="3" {% if level == "3" %}selected{% endif %}>Level 3 (Google Meet)</option>
      </select>
    </div>
    <div class="mb-3">
      <label for="job_title" class="form-label">Job Title</label>
      <input type="text" class="form-control" id="job_title" name="job_title" placeholder="Enter job title" required value="{{ job_title | default('') }}" />
    </div>
    <div class="mb-3" id="job_desc_group" style="display:none;">
      <label for="job_description" class="form-label">Job Description</label>
      <textarea class="form-control" id="job_description" name="job_description" rows="3" placeholder="Enter job description"></textarea>
    </div>
    <div class="mb-3" id="cand_email_group" style="display:none;">
      <label for="candidate_emails" class="form-label">Candidate Emails (comma separated)</label>
      <input type="text" class="form-control" id="candidate_emails" name="candidate_emails" placeholder="e.g., email1@example.com, email2@example.com" required value="{{ candidate_emails | default('') }}" />
      <div id="candidate-emails-error" class="invalid-feedback" style="display:none;">
        All emails must contain '@' and '.' (e.g., user@domain.com).
      </div>
    </div>
    <div class="mb-3" id="intv_date_group" style="display:none;">
      <label for="interview_datetime" class="form-label">Interview Date and Time</label>
      <input type="datetime-local" class="form-control" id="interview_datetime" name="interview_datetime" />
    </div>
    <div class="mb-3" id="hr_available_group" style="display:none;">
      <label class="form-label">Is HR Available?</label>
      <div class="form-check form-check-inline">
        <input class="form-check-input" type="radio" name="hr_available" id="hr_yes" value="yes" onclick="toggleHRFields(true)">
        <label class="form-check-label" for="hr_yes">Yes</label>
      </div>
      <div class="form-check form-check-inline">
        <input class="form-check-input" type="radio" name="hr_available" id="hr_no" value="no" onclick="toggleHRFields(false)">
        <label class="form-check-label" for="hr_no">No</label>
      </div>
    </div>
    <div id="hr_fields" style="display:none;">
      <div class="mb-3">
        <label for="hr_emails" class="form-label">HR Emails</label>
        <input type="text" class="form-control" id="hr_emails" name="hr_emails" placeholder="e.g., hr1@example.com, hr2@example.com">
      </div>
      <div class="mb-3">
        <label for="viewer_emails" class="form-label">Viewers' Emails (Optional)</label>
        <input type="text" class="form-control" id="viewer_emails" name="viewer_emails" placeholder="e.g., viewer1@example.com">
      </div>
      <div class="mb-3">
        <label for="from_time" class="form-label">From Time</label>
        <input type="time" class="form-control" id="from_time" name="from_time">
      </div>
      <div class="mb-3">
        <label for="to_time" class="form-label">To Time</label>
        <input type="time" class="form-control" id="to_time" name="to_time">
      </div>
      <div class="mb-3">
        <label for="hr_date" class="form-label">Interview Date</label>
        <input type="date" class="form-control" id="hr_date" name="hr_date">
      </div>
    </div>
    <button type="submit" class="btn btn-primary w-100 mb-3">Schedule Interview</button>
    <a href="/admin/dashboard" class="btn btn-info w-100">View Dashboard</a>
  </form>
  <div class="text-center mt-4">
    <a href="/" class="btn btn-secondary">Back to Home</a>
  </div>
 </div>
 <script>
    function updateFields() {
      const level = document.getElementById("level").value;
      document.getElementById("job_desc_group").style.display = level === "2" ? "block" : "none";
      document.getElementById("cand_email_group").style.display = level ? "block" : "none";
      document.getElementById("intv_date_group").style.display = (level === "1" || level === "2") ? "block" : "none";
      document.getElementById("hr_available_group").style.display = level === "3" ? "block" : "none";
      document.getElementById("hr_fields").style.display = "none";
    }
    function toggleHRFields(show) {
      const hrFields = document.getElementById("hr_fields");
      hrFields.style.display = show ? "block" : "none";
      document.getElementById("intv_date_group").style.display = show ? "none" : "block";
    }

    // Email validation function
    function validateEmails(emailString) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/; // Basic email regex requiring @ and .
      const emails = emailString.split(",").map(email => email.trim()).filter(email => email);
      return emails.every(email => emailRegex.test(email));
    }

    // Form submission handler
    document.getElementById("schedule-form").addEventListener("submit", function(event) {
      const candidateEmailsInput = document.getElementById("candidate_emails");
      const errorDiv = document.getElementById("candidate-emails-error");
      if (candidateEmailsInput.value) {
        if (!validateEmails(candidateEmailsInput.value)) {
          event.preventDefault();
          errorDiv.style.display = "block";
          candidateEmailsInput.classList.add("is-invalid");
        } else {
          errorDiv.style.display = "none";
          candidateEmailsInput.classList.remove("is-invalid");
        }
      }
    });

    // Run updateFields on page load to reflect pre-selected level
    window.onload = function() {
      updateFields();
    };
 </script>
</body>
</html>
""", 
    "candidate_login.html ":"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Candidate Login</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(to right, #eceff1, #cfd8dc);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .login-container {
      background: #fff;
      padding: 40px;
      border-radius: 15px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
      max-width: 450px;
      width: 100%;
      animation: slideUp 0.8s ease-out;
    }
    h2 {
      color: #37474f;
      margin-bottom: 30px;
      font-weight: 600;
    }
    .form-label {
      color: #455a64;
      font-weight: 500;
    }
    .form-control {
      border-radius: 10px;
      padding: 12px;
      transition: border-color 0.3s;
    }
    .form-control:focus {
      border-color: #1976d2;
      box-shadow: 0 0 5px rgba(25, 118, 210, 0.5);
    }
    .btn-login {
      background: #1976d2;
      border: none;
      padding: 12px;
      border-radius: 10px;
      font-size: 1.1rem;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-login:hover {
      background: #1565c0;
      transform: translateY(-2px);
    }
    .invalid-feedback {
      color: #d32f2f;
      font-size: 0.9rem;
      margin-top: 5px;
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(50px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="login-container">
    <h2 class="text-center">Candidate Login</h2>
    <form action="/candidate/login" method="post" id="login-form">
      <div class="mb-4">
        <label for="email" class="form-label">Email Address</label>
        <input type="email" class="form-control" id="email" name="email" placeholder="Enter your email" required value="{{ email if email else '' }}">
        <div id="email-error" class="invalid-feedback" style="display:none;">
          Email must contain '@' and '.' (e.g., user@domain.com).
        </div>
      </div>
      <div class="mb-4">
        <label for="password" class="form-label">Password</label>
        <input type="password" class="form-control" id="password" name="password" placeholder="Enter your password" required>
      </div>
      <button type="submit" class="btn btn-login w-100">Login</button>
    </form>
    <p class="text-center mt-3"><a href="/" class="text-decoration-none">Back to Home</a></p>
    {% if message %}
      <div class="alert alert-danger mt-3">
        {{ message }}
      </div>
    {% endif %}
  </div>
  <script>
    // Email validation function
    function validateEmail(email) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/; // Basic email regex requiring @ and .
      return emailRegex.test(email);
    }

    // Form submission handler
    document.getElementById("login-form").addEventListener("submit", function(event) {
      const emailInput = document.getElementById("email");
      const errorDiv = document.getElementById("email-error");
      if (!validateEmail(emailInput.value)) {
        event.preventDefault();
        errorDiv.style.display = "block";
        emailInput.classList.add("is-invalid");
      } else {
        errorDiv.style.display = "none";
        emailInput.classList.remove("is-invalid");
      }
    });
  </script>
</body>
</html>
""",
    "interview.html": """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Candidate Interview</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: #f5f7fa;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      padding: 20px;
    }
    .interview-container {
      background: #fff;
      padding: 30px;
      border-radius: 15px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
      max-width: 900px;
      margin: 0 auto;
      position: relative;
      animation: fadeIn 0.8s ease-in;
    }
    .illustration-container {
      text-align: center;
      margin-bottom: 20px;
    }
    .illustration-container img {
      max-width: 100%;
      height: auto;
      border-radius: 10px;
    }
    .bubble {
      padding: 15px;
      border-radius: 15px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
      max-width: 300px;
      position: absolute;
      transition: opacity 0.5s;
    }
    .question-bubble {
      background: #e3f2fd;
      top: 20px;
      right: 20px;
    }
    .answer-bubble {
      background: #fff3e0;
      top: 100px;
      left: 20px;
    }
    #hidden-video {
      width: 100%;
      max-width: 400px;
      border-radius: 10px;
      margin: 20px auto;
      display: block;
    }
    #timer {
      font-weight: bold;
      color: #d32f2f;
    }
    #warning-message {
      color: #d32f2f;
      font-weight: 500;
      margin-top: 10px;
    }
    .btn-next {
      background: #1976d2;
      border: none;
      padding: 10px 20px;
      border-radius: 10px;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-next:hover {
      background: #1565c0;
      transform: translateY(-2px);
    }
    .section {
      margin-top: 20px;
      padding: 20px;
      background: #fafafa;
      border-radius: 10px;
    }
    #question-section {
      position: relative;
    }
    #next-btn {
      position: absolute;
      bottom: 20px;
      right: 20px;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  </style>
</head>
<body>
  <div class="interview-container">
    <div id="illustration-container" class="illustration-container" style="display:none;">
      <img src="/static/image.gif" alt="Interview Illustration">
    </div>
    <div id="question-bubble" class="bubble question-bubble" style="display:none;">
      <p id="question-text" class="h5 mb-0"></p>
    </div>
    <div id="answer-bubble" class="bubble answer-bubble" style="display:none;">
      <p>Your Answer: <span id="candidate-answer"></span></p>
    </div>
    <div id="precheck" class="section text-center">
      <p class="lead">Please allow camera and microphone access for precheck.</p>
      <video id="hidden-video" autoplay muted></video>
      <canvas id="video-canvas" width="320" height="240" style="display:none;"></canvas>
      <p id="precheck-status" class="mt-3">Checking...</p>
      <button id="start-interview" class="btn btn-primary mt-3" disabled>Start Interview</button>
    </div>
    <div id="question-section" class="section" style="display:none;">
      <p>Time Remaining: <span id="timer">30</span> seconds</p>
      <p id="warning-message"></p>
      <button id="next-btn" class="btn btn-next">Next Question</button>
    </div>
    <div id="result-section" class="section text-center" style="display:none;">
      <h3>Interview Completed</h3>
      <p id="total-score"></p>
      <pre id="evaluations" class="text-start"></pre>
    </div>
  </div>
  <script>
    const email = "{{ email }}";
    let qas = {{ qas|tojson }};
    let currentQuestionIndex = 0;
    let responses = [];
    const timeLimit = 30;
    let detectionInterval;
    let warnings = 0;
    const maxWarnings = 3;
    let candidateStream;
    let mediaRecorder;
    let recordedChunks = [];
    let skipRequested = false;
    let currentRecognition = null;

    async function recognizeSpeechForPrecheck() {
      return new Promise((resolve, reject) => {
        if (!("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices)) {
          return reject("getUserMedia is not supported.");
        }
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return reject("Speech Recognition not supported");
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;
        let transcript = "";
        recognition.onresult = (event) => {
          for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
              transcript += event.results[i][0].transcript + " ";
              if (transcript.toLowerCase().includes("hello")) {
                recognition.stop();
              }
            }
          }
          document.getElementById('candidate-answer').innerText = transcript;
        };
        recognition.onerror = (event) => { console.error("Speech recognition error:", event.error); };
        recognition.onend = () => { resolve(transcript.trim()); };
        recognition.start();
      });
    }

    async function recognizeSpeechForDuration() {
      return new Promise((resolve, reject) => {
        if (!("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices)) {
          return reject("getUserMedia is not supported.");
        }
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return reject("Speech Recognition not supported");
        const recognition = new SpeechRecognition();
        currentRecognition = recognition;
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;
        let finalTranscript = "";
        let lastResultTime = Date.now();
        let minDurationPassed = false;

        setTimeout(() => {
          minDurationPassed = true;
        }, 30000); // Minimum 30 seconds

        const checkSilence = setInterval(() => {
          if (minDurationPassed && (Date.now() - lastResultTime > 2000)) {
            recognition.stop();
          }
        }, 1000);

        recognition.onresult = (event) => {
          lastResultTime = Date.now();
          let interimTranscript = "";
          for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
              finalTranscript += event.results[i][0].transcript + " ";
            } else {
              interimTranscript += event.results[i][0].transcript + " ";
            }
          }
          document.getElementById('candidate-answer').innerText = finalTranscript + interimTranscript;
        };
        recognition.onerror = (event) => {
          clearInterval(checkSilence);
          console.error("Speech recognition error:", event.error);
          reject(event.error);
        };
        recognition.onend = () => {
          clearInterval(checkSilence);
          currentRecognition = null;
          resolve(finalTranscript.trim());
        };
        recognition.start();
      });
    }

    async function updateCanvasWithDetection() {
      try {
        const video = document.getElementById('hidden-video');
        if (!video.videoWidth || !video.videoHeight) return;
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));
        const formData = new FormData();
        formData.append("email", "precheck");
        formData.append("file", blob, "frame.jpg");
        const res = await fetch("/interview/detect_frame", { method: "POST", body: formData });
        const data = await res.json();
        document.getElementById('warning-message').innerText = data.message;
        const match = data.message.match(/Warning\((\d)\/3\)/);
        if(match) {
          let newWarningCount = parseInt(match[1]);
          if(newWarningCount > warnings) {
            warnings = newWarningCount;
            if(warnings >= maxWarnings) {
              alert("Interview terminated due to repeated violations.");
              clearInterval(detectionInterval);
              window.location.href = "/candidate/login";
              return;
            }
          }
        }
      } catch (err) {
        console.error("Detection error:", err);
      }
    }

    async function startPrecheck() {
      try {
        if (!("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices)) {
          document.getElementById('precheck-status').innerText = "getUserMedia not supported.";
          return;
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        candidateStream = stream;
        if (!stream || stream.getVideoTracks().length === 0) {
          document.getElementById('precheck-status').innerText = "Enable your camera.";
          return;
        }
        const video = document.getElementById('hidden-video');
        video.srcObject = stream;
        video.onloadedmetadata = async () => {
          video.play();
          await new Promise(resolve => setTimeout(resolve, 2000));
          const precheckDetectionInterval = setInterval(updateCanvasWithDetection, 2000);
          const transcript = await recognizeSpeechForPrecheck();
          clearInterval(precheckDetectionInterval);
          if (transcript.toLowerCase().includes("hello")) {
            document.getElementById('precheck-status').innerText = "Precheck successful.";
            document.getElementById('start-interview').disabled = false;
          } else {
            document.getElementById('precheck-status').innerText = "Voice detection failed. Refreshing...";
            setTimeout(() => { window.location.reload(); }, 2000);
          }
        };
      } catch (err) {
        document.getElementById('precheck-status').innerText = "Precheck failed: " + err;
      }
    }

    async function askQuestion() {
      skipRequested = false;
      document.getElementById('question-bubble').style.display = "block";
      document.getElementById('answer-bubble').style.display = "block";
      document.getElementById('illustration-container').style.display = "block";
      if (currentQuestionIndex >= qas.length) {
        submitInterview();
        return;
      }
      document.getElementById('candidate-answer').innerText = "";
      const currentQA = qas[currentQuestionIndex];
      document.getElementById('question-text').innerText = currentQA.question;
      let synth = window.speechSynthesis;
      synth.speak(new SpeechSynthesisUtterance(currentQA.question));
      let remainingTime = timeLimit;
      document.getElementById('timer').innerText = remainingTime;
      document.getElementById('question-section').style.display = "block";
      let timerIntervalLocal = setInterval(() => { 
         remainingTime--; 
         document.getElementById('timer').innerText = remainingTime; 
         if (remainingTime <= 0) clearInterval(timerIntervalLocal);
      }, 1000);
      const answer = await recognizeSpeechForDuration();
      clearInterval(timerIntervalLocal);
      const candidateAnswer = document.getElementById('candidate-answer').innerText;
      responses.push({ question: currentQA.question, candidate_answer: candidateAnswer });
      document.getElementById('candidate-answer').innerText = "";
      currentQuestionIndex++;
      if (currentQuestionIndex < qas.length) {
        askQuestion();
      } else {
        clearInterval(detectionInterval);
        submitInterview();
      }
    }

    async function uploadRecording(blob) {
      const formData = new FormData();
      formData.append("file", blob, "recording.webm");
      formData.append("email", email);
      try {
        const res = await fetch("/interview/upload_recording", { method: "POST", body: formData });
        const data = await res.json();
        console.log("Upload response:", data.message);
      } catch (err) {
        console.error("Error uploading recording:", err);
      }
    }

    async function submitInterview() {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        mediaRecorder.onstop = async () => {
          const blob = new Blob(recordedChunks, { type: "video/webm" });
          await uploadRecording(blob);
          finalizeSubmission();
        };
      } else {
        finalizeSubmission();
      }
    }

    function finalizeSubmission() {
      const payload = { email: email, responses: responses };
      fetch("/interview/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).then(() => {
        window.location.href = "/candidate/submit_name?email=" + encodeURIComponent(email);
      });
    }

    document.getElementById('next-btn').addEventListener('click', () => {
      if (currentRecognition) {
        currentRecognition.stop();
      }
      skipRequested = true;
    });

    document.getElementById('start-interview').addEventListener('click', () => {
      if (candidateStream) {
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(candidateStream);
        mediaRecorder.ondataavailable = event => { if (event.data.size > 0) { recordedChunks.push(event.data); } };
        mediaRecorder.start();
      }
      document.getElementById('precheck').style.display = "none";
      detectionInterval = setInterval(detectFrame, 2000);
      askQuestion();
    });

    async function detectFrame() {
      const video = document.getElementById('hidden-video');
      if (!video.videoWidth || !video.videoHeight) return;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));
      const formData = new FormData();
      formData.append("email", email);
      formData.append("file", blob, "frame.jpg");
      try {
        const res = await fetch("/interview/detect_frame", { method: "POST", body: formData });
        const data = await res.json();
        document.getElementById('warning-message').innerText = data.message;
        const match = data.message.match(/Warning\((\d)\/3\)/);
        if(match) {
          let newWarningCount = parseInt(match[1]);
          if(newWarningCount > warnings) {
            warnings = newWarningCount;
            if(warnings >= maxWarnings) {
              alert("Interview terminated due to repeated violations.");
              clearInterval(detectionInterval);
              window.location.href = "/candidate/login";
              return;
            }
            clearInterval(detectionInterval);
            setTimeout(() => {
              detectionInterval = setInterval(detectFrame, 2000);
            }, 5000);
          }
        }
      } catch(err) {
        console.error("Detection error:", err);
      }
    }

    window.onload = function() {
      startPrecheck();
    };
  </script>
</body>
</html>
""",
    "aptitude.html": """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Aptitude Test</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: #f5f7fa;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      padding: 20px;
    }
    .aptitude-container {
      background: #fff;
      padding: 30px;
      border-radius: 15px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      animation: fadeIn 0.8s ease-in;
    }
    .main-content {
      flex: 1;
      padding-right: 20px;
    }
    .sidebar {
      width: 200px;
      background: #fafafa;
      padding: 15px;
      border-radius: 10px;
      display: none;
      flex-direction: column;
      align-items: center;
      min-height: 300px;
    }
    .question-grid {
      display: grid;
      grid-template-columns: repeat(4, 30px);
      grid-gap: 10px;
      max-height: 200px;
      overflow-y: auto;
      width: 100%;
      justify-content: center;
    }
    .question-number {
      width: 30px;
      height: 30px;
      line-height: 30px;
      border-radius: 50%;
      text-align: center;
      font-weight: bold;
      border: 2px solid #1976d2;
      color: #1976d2;
      background: #fff;
      cursor: pointer;
      transition: all 0.3s;
      font-size: 0.8rem;
    }
    .question-number.active {
      background: #1976d2;
      color: #fff;
    }
    .question-number.answered {
      background: #2e7d32;
      border-color: #2e7d32;
      color: #fff;
    }
    .question-stats {
      margin-top: 15px;
      text-align: center;
      font-size: 0.9rem;
      color: #455a64;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .question-stats button {
      background: #e0e0e0;
      border: 1px solid #ccc;
      border-radius: 5px;
      padding: 5px 10px;
      margin: 5px 0;
      cursor: default;
      width: 120px;
    }
    #answered-btn {
      background: #2e7d32;
      color: #fff;
    }
    #unanswered-btn {
      background: #d32f2f;
      color: #fff;
    }
    .question-area {
      margin-top: 20px;
    }
    .question {
      font-size: 1.3rem;
      color: #263238;
      margin-bottom: 15px;
    }
    .options li {
      margin-bottom: 15px;
      padding: 10px;
      border-radius: 8px;
      transition: background 0.3s;
    }
    .options li:hover {
      background: #e3f2fd;
    }
    .timer {
      font-size: 1.2rem;
      color: #d32f2f;
      font-weight: bold;
    }
    #live-warning-message {
      color: #d32f2f;
      font-weight: 500;
    }
    .btn-nav {
      padding: 10px 20px;
      border-radius: 10px;
      transition: background 0.3s, transform 0.3s;
    }
    .btn-nav:hover {
      transform: translateY(-2px);
    }
    #hidden-video {
      width: 100%;
      max-width: 400px;
      border-radius: 10px;
      margin: 20px auto;
      display: block;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  </style>
</head>
<body>
  <div class="aptitude-container">
    <div class="main-content">
      <div id="precheck" class="text-center">
        <p class="lead">Please allow camera and microphone access for precheck.</p>
        <video id="hidden-video" autoplay muted></video>
        <canvas id="video-canvas" width="320" height="240" style="display:none;"></canvas>
        <p id="precheck-status" class="mt-3">Checking...</p>
        <button id="start-test" class="btn btn-primary mt-3" disabled>Start Test</button>
        <p id="warning-message"></p>
      </div>
      <div id="test-area" class="question-area" style="display:none;">
        <div class="timer">Time: <span id="timer">50:00</span></div>
        <p id="live-warning-message"></p>
        <div id="question-container">
          <div class="question" id="questionText"></div>
          <ul class="options list-unstyled" id="optionsContainer"></ul>
        </div>
        <div class="navigation mt-4 text-center">
          <button class="btn btn-secondary btn-nav" id="prevBtn" onclick="prevQuestion()" style="display:none;">Previous</button>
          <button class="btn btn-secondary btn-nav mx-2" id="nextBtn" onclick="nextQuestion()">Next</button>
          <button class="btn btn-primary btn-nav" id="submitBtn" onclick="submitAnswers()">Submit</button>
        </div>
      </div>
    </div>
    <div class="sidebar" id="questionList">
      <div class="question-grid" id="question-grid"></div>
      <div class="question-stats">
        <button id="answered-btn">Answered: <span id="answered-questions">0</span></button>
        <button id="unanswered-btn">Unanswered: <span id="unanswered-questions">0</span></button>
      </div>
    </div>
    <!-- Confirmation Modal -->
    <div class="modal fade" id="confirmModal" tabindex="-1" aria-labelledby="confirmModalLabel" aria-hidden="true">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="confirmModalLabel">Confirm Submission</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <p>You have answered <span id="answeredCount"></span> out of <span id="totalCount"></span> questions.</p>
            <p>Are you sure you want to submit?</p>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
            <button type="button" class="btn btn-primary" id="confirmSubmit">Confirm</button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    const email = "{{ email }}";
    let questions = {{ questions|tojson }};
    let currentIndex = 0;
    let userAnswers = {};
    let totalTime = 0;
    let timerInterval = null;
    let detectionInterval;
    let warnings = 0;
    const maxWarnings = 3;
    let candidateStream;
    let mediaRecorder;
    let recordedChunks = [];

    function updateQuestionStats() {
      const total = questions.length;
      const answered = Object.keys(userAnswers).length;
      const unanswered = total - answered;
      document.getElementById('answered-questions').innerText = answered;
      document.getElementById('unanswered-questions').innerText = unanswered;
    }

    function renderQuestion() {
      if (questions.length === 0) return;
      const currentQuestion = questions[currentIndex];
      document.getElementById('questionText').innerText = (currentIndex + 1) + ". " + currentQuestion.question;
      let optionsContainer = document.getElementById('optionsContainer');
      optionsContainer.innerHTML = "";
      currentQuestion.options.forEach(opt => {
        let li = document.createElement('li');
        let label = document.createElement('label');
        let radio = document.createElement('input');
        radio.type = 'radio';
        radio.name = 'option';
        radio.value = opt;
        if (userAnswers[currentQuestion.id] === opt) {
          radio.checked = true;
        }
        label.appendChild(radio);
        label.appendChild(document.createTextNode(" " + opt));
        li.appendChild(label);
        optionsContainer.appendChild(li);
      });
      updateQuestionCircles();
      updateQuestionStats();
      document.getElementById('prevBtn').style.display = currentIndex === 0 ? 'none' : 'inline-block';
      document.getElementById('nextBtn').style.display = currentIndex === questions.length - 1 ? 'none' : 'inline-block';
      document.getElementById('submitBtn').style.display = 'inline-block';
    }

    function saveCurrentAnswer() {
      const currentQuestion = questions[currentIndex];
      const radios = document.getElementsByName('option');
      let answered = false;
      for (let r of radios) {
        if (r.checked) {
          userAnswers[currentQuestion.id] = r.value;
          answered = true;
          break;
        }
      }
      const circles = document.getElementsByClassName('question-number');
      if (answered) {
        circles[currentIndex].classList.add('answered');
      } else {
        circles[currentIndex].classList.remove('answered');
      }
    }

    function nextQuestion() {
      saveCurrentAnswer();
      if (currentIndex < questions.length - 1) {
        currentIndex++;
        renderQuestion();
      }
    }

    function prevQuestion() {
      saveCurrentAnswer();
      if (currentIndex > 0) {
        currentIndex--;
        renderQuestion();
      }
    }

    function updateQuestionCircles() {
      const circles = document.getElementsByClassName('question-number');
      for (let i = 0; i < circles.length; i++) {
        circles[i].classList.remove('active');
        circles[i].classList.toggle('answered', questions[i].id in userAnswers);
        if (i === currentIndex) {
          circles[i].classList.add('active');
        }
      }
    }

    function buildQuestionCircles() {
      const questionGrid = document.getElementById('question-grid');
      questionGrid.innerHTML = "";
      for (let i = 0; i < questions.length; i++) {
        const circle = document.createElement('div');
        circle.className = `question-number ${i === currentIndex ? 'active' : ''} ${questions[i].id in userAnswers ? 'answered' : ''}`;
        circle.textContent = i + 1;
        circle.onclick = () => {
          saveCurrentAnswer();
          currentIndex = i;
          renderQuestion();
          updateQuestionCircles();
        };
        questionGrid.appendChild(circle);
      }
    }

    function doSubmit() {
      clearInterval(timerInterval);
      clearInterval(detectionInterval);
      fetch('/ap1/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email, answers: userAnswers })
      })
      .then(res => res.json())
      .then(result => {
        window.location.href = "/candidate/submit_name?email=" + encodeURIComponent(email);
      });
    }

    function submitAnswers() {
      saveCurrentAnswer();
      const answered = Object.keys(userAnswers).length;
      const total = questions.length;
      document.getElementById('answeredCount').innerText = answered;
      document.getElementById('totalCount').innerText = total;
      const confirmModal = new bootstrap.Modal(document.getElementById('confirmModal'));
      confirmModal.show();
    }

    function startTimer() {
      const maxTime = 50 * 60;
      let remainingTime = maxTime;
      timerInterval = setInterval(() => {
        if (remainingTime <= 0) {
          clearInterval(timerInterval);
          doSubmit();
        } else {
          const minutes = Math.floor(remainingTime / 60);
          const seconds = remainingTime % 60;
          document.getElementById('timer').textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
          remainingTime--;
        }
      }, 1000);
    }

    async function updateCanvasWithDetection() {
      try {
        const video = document.getElementById('hidden-video');
        if (!video.videoWidth || !video.videoHeight) return;
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));
        const formData = new FormData();
        formData.append("email", "precheck");
        formData.append("file", blob, "frame.jpg");
        const res = await fetch("/interview/detect_frame", { method: "POST", body: formData });
        const data = await res.json();
        document.getElementById('warning-message').innerText = data.message;
        const match = data.message.match(/Warning\((\d)\/3\)/);
        if (match) {
          let newWarningCount = parseInt(match[1]);
          if (newWarningCount > warnings) {
            warnings = newWarningCount;
            if (warnings >= maxWarnings) {
              alert("Test terminated due to repeated violations.");
              clearInterval(detectionInterval);
              window.location.href = "/candidate/login";
              return;
            }
          }
        }
      } catch (err) {
        console.error("Detection error:", err);
      }
    }

    async function startPrecheck() {
      try {
        if (!("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices)) {
          document.getElementById('precheck-status').innerText = "getUserMedia not supported.";
          return;
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        candidateStream = stream;
        if (!stream || stream.getVideoTracks().length === 0) {
          document.getElementById('precheck-status').innerText = "Enable your camera.";
          return;
        }
        const video = document.getElementById('hidden-video');
        video.srcObject = stream;
        video.onloadedmetadata = async () => {
          video.play();
          await new Promise(resolve => setTimeout(resolve, 2000));
          const precheckDetectionInterval = setInterval(updateCanvasWithDetection, 2000);
          const transcript = await recognizeSpeechForPrecheck();
          clearInterval(precheckDetectionInterval);
          if (transcript.trim().length > 0) {
            document.getElementById('precheck-status').innerText = "Precheck successful.";
            document.getElementById('start-test').disabled = false;
          } else {
            document.getElementById('precheck-status').innerText = "Voice detection failed. Refreshing...";
            setTimeout(() => { window.location.reload(); }, 2000);
          }
        };
      } catch (err) {
        document.getElementById('precheck-status').innerText = "Precheck failed: " + err;
      }
    }

    async function recognizeSpeechForPrecheck() {
      return new Promise((resolve, reject) => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return reject("Speech Recognition not supported");
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;
        let transcript = "";
        recognition.onresult = (event) => {
          for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
              transcript += event.results[i][0].transcript + " ";
              if (transcript.trim().length > 0) {
                recognition.stop();
              }
            }
          }
        };
        recognition.onerror = (event) => { console.error("Speech recognition error:", event.error); };
        recognition.onend = () => { resolve(transcript.trim()); };
        recognition.start();
      });
    }

    document.getElementById('start-test').addEventListener('click', () => {
