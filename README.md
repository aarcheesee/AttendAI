# AttendAI — Smart Face Recognition Attendance System

> A real-world attendance system built with Flask, OpenCV, and SQLite — designed for classroom use.

---

## Features

- Face Recognition** — Live LBPH-based face detection and recognition via webcam
- QR Code Backup** — Students can mark attendance by scanning a personal QR code
- Lecture Timing** — Set lecture start time and grace period; attendance auto-marked as **Present** or **Late**
- Student Management** — Register students with photo, class, division, roll no, email, and phone
- Attendance Dashboard** — View all records with Present / Late / Absent status badges
- Absent Tracking** — Automatically shows who hasn't marked attendance for the active lecture
- Per-Student Stats** — Attendance analytics per student (last 30 days)
- CSV Export** — Download full attendance records as a spreadsheet
- Admin Panel** — Manage system users with role-based access
- Secure Login** — Hashed passwords with Werkzeug, session-based auth

---

 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask |
| Face Recognition | OpenCV (LBPH), Haar Cascades |
| Database | SQLite (via Python sqlite3) |
| QR Code | qrcode, pyzbar |
| Frontend | Jinja2, HTML/CSS, Chart.js |
| Auth | Werkzeug password hashing |

---

 Project Structure

```
attendance-system/
├── app.py                  # Main Flask application
├── users.db                # SQLite database (auto-created, not tracked)
├── requirements.txt        # Python dependencies
├── dataset/                # Face image samples (not tracked)
├── static/
│   ├── style.css
│   ├── previews/           # Student photo thumbnails
│   └── qrcodes/            # Generated QR codes
└── templates/
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── students.html
    ├── add_student.html
    ├── capture.html
    ├── confirm.html
    ├── index.html
    ├── qr.html
    ├── admin.html
    └── attendance.html
```

---

 Setup & Installation

 Prerequisites
- macOS / Linux
- Python 3.11+
- Homebrew (macOS)
- Webcam

 1. Clone the repository
```bash
git clone https://github.com/aarcheesee/AttendAI.git
cd AttendAI
```

 2. Create a virtual environment
```bash
conda create -n attendai python=3.11 -y
conda activate attendai
```

 3. Install dependencies
```bash
brew install cmake        # macOS only — required for dlib
pip install -r requirements.txt
```

 4. Run the application
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5004**

 Default login
| Username | Password |
|---|---|
| admin | admin123 |

> Change the password after first login via the Admin Panel.

---

## How to Use

 Adding a Student
1. Go to **Students → Add Student**
2. Fill in name, class, division, roll no, email, phone
3. Click **Save & Start Face Capture** — camera opens automatically
4. Student looks at camera — 60 face samples are captured
5. Model retrains automatically

 Taking Attendance
1. Go to **Dashboard → Start Lecture**
2. Enter lecture name, start time, and late-after minutes (default: 10)
3. Click **Start Camera** — face recognition begins
4. Recognised faces are marked **Present** or **Late** automatically

 Viewing Records
- Dashboard shows all attendance with status badges
- Absent students are listed live during an active lecture
- Click **Download CSV** to export records

---

Database Schema

```
users        — login accounts (id, username, password)
students     — student profiles (name, class, division, roll_no, email, phone)
attendance   — records (name, lecture, date, time, status)
settings     — system settings (current_lecture, start_time, late_after_mins)
```

---

Screenshots

> Dashboard with attendance records and lecture timing

> Student registry with photo thumbnails and details

> Live face capture with progress bar

---

Requirements

```
flask
werkzeug
opencv-python
opencv-contrib-python
numpy
pandas
cmake
dlib
face_recognition
qrcode[pil]
pyzbar
Pillow
```

Install all with:
```bash
pip install -r requirements.txt
```

---

 Known Limitations

- LBPH recognition accuracy drops in poor lighting or at extreme angles
- Designed for single-camera setups
- Best performance with 20–30 students per session

---

Author

Aarchi bhanushali
Second Year Project — Smart Attendance System  
Built with Flask + OpenCV + SQLite

---

## License

This project is for educational purposes.