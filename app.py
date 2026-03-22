import sqlite3
from flask import Flask, render_template, Response, request, redirect, session, send_file, flash
import cv2
import pandas as pd
import os
import datetime
import numpy as np
import time
last_marked = {}

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT
                )''')

    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", "1234"))
    except:
        pass

    conn.commit()
    conn.close()

init_db()

# ================= APP =================
app = Flask(__name__)
app.secret_key = "attendance_secret"

current_lecture = "None"
dataset_path = "dataset"
attendance_file = "attendance.csv"

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= INIT =================
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name","Lecture","Date","Time"]).to_csv(attendance_file,index=False)

# ================= TRAIN =================
def train_model():
    faces, labels = [], []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        label_map[label_id] = person_name

        for image_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img,(200,200))
            faces.append(img)
            labels.append(label_id)

        label_id += 1

    if faces:
        recognizer.train(faces, np.array(labels))

    return label_map

label_map = train_model()

# ================= ATTENDANCE =================
def mark_attendance(name, lecture):
    df = pd.read_csv(attendance_file)

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    already = df[
        (df["Name"] == name) &
        (df["Lecture"] == lecture) &
        (df["Date"] == date)
    ]

    if already.empty:
        new_row = pd.DataFrame([[name, lecture, date, time_now]],
                            columns=["Name", "Lecture", "Date", "Time"])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)

        flash(f"{name} marked present", "success")

# ================= CAMERA =================
def generate_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 1.2, 6, minSize=(100,100)
        )

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(200,200))

            name = "Unknown"
            color = (0,0,255)

            if len(label_map) > 0:
                try:
                    label, confidence = recognizer.predict(face)

                    if confidence < 75:
                        name = label_map[label]
                        color = (0,255,0)
                        mark_attendance(name, current_lecture)
                except:
                    pass

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
            cv2.putText(frame,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# ================= CAPTURE =================
def capture_faces(name):
    camera = cv2.VideoCapture(0)
    count = 0
    path = os.path.join(dataset_path, name)

    if not os.path.exists(path):
        os.makedirs(path)

    while count < 50:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(200,200))

            file_path = os.path.join(path, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

    # retrain after capture
    global label_map
    label_map = train_model()

# ================= ROUTES =================
@app.route('/')
def login():
    return render_template("login.html")

@app.route('/login', methods=["POST"])
def login_check():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",(username,password))
    user = c.fetchone()
    conn.close()

    if user:
        session["user"] = username
        flash("Login successful!", "success")
        return redirect("/dashboard")

    flash("Invalid credentials", "error")
    return redirect("/")

@app.route('/register', methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                      (username, password))
            conn.commit()
            flash("Account created!", "success")
            return redirect("/")
        except:
            flash("Username exists", "error")

        conn.close()

    return render_template("register.html")

@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        return redirect("/")

    df = pd.read_csv(attendance_file)
    chart_data = df["Name"].value_counts()

    return render_template("dashboard.html",
                           records=df.to_dict('records'),
                           total=len(df),
                           students=df["Name"].nunique(),
                           chart_labels=list(chart_data.index),
                           chart_values=list(chart_data.values))

@app.route('/set_lecture', methods=["POST"])
def set_lecture():
    global current_lecture
    current_lecture = request.form["lecture"]
    flash(f"Lecture set to {current_lecture}", "info")
    return redirect("/dashboard")

@app.route('/add_student', methods=["POST"])
def add_student():
    name = request.form["name"]
    return render_template("capture.html", name=name)

@app.route('/video_feed_capture/<name>')
def video_feed_capture(name):
    return Response(capture_faces(name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    if "user" not in session:
        return redirect("/")
    return render_template("index.html")

@app.route('/video')
def video():
    if "user" not in session:
        return redirect("/")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download')
def download():
    if "user" not in session:
        return redirect("/")
    return send_file(attendance_file, as_attachment=True)

@app.route('/reset')
def reset():
    if "user" not in session:
        return redirect("/")
    pd.DataFrame(columns=["Name","Lecture","Date","Time"]).to_csv(attendance_file,index=False)
    flash("Attendance reset", "error")
    return redirect("/dashboard")

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect("/")

@app.route('/admin')
def admin_panel():
    if "user" not in session:
        return redirect("/")

    # only admin allowed
    if session["user"] != "admin":
        flash("Access denied", "error")
        return redirect("/dashboard")

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id, username FROM users")
    users = c.fetchall()
    conn.close()

    return render_template("admin.html", users=users)

@app.route('/delete_user/<int:user_id>', methods=["POST"])
def delete_user(user_id):
    if "user" not in session:
        return redirect("/")

    if session["user"] != "admin":
        flash("Access denied", "error")
        return redirect("/dashboard")

    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # prevent deleting admin
    c.execute("SELECT username FROM users WHERE id=?", (user_id,))
    user = c.fetchone()

    if user and user[0] == "admin":
        flash("Cannot delete admin", "error")
        conn.close()
        return redirect("/admin")

    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    flash("User deleted successfully", "success")
    return redirect("/admin")
# ================= RUN =================
if __name__=="__main__":
    app.run(debug=True, port=5001)