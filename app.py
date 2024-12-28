import os
import subprocess
from flask import Flask, request, render_template, send_from_directory, url_for, jsonify
import numpy as np
from Hfinder import Hfinder

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extrinsic', methods=['GET'])
def extrinsic_index():
    return render_template('extrinsic.html')

@app.route('/api/extrinsic', methods=['POST'])
def calculate_extrinsic():
    # Load from request body json
    camera_ks = np.array(request.json['camera_ks'], np.float32)
    dist = np.array(request.json['dist'], np.float32)
    # nmtx = np.array(request.json['nmtx'], np.float32)
    court3D = np.array(request.json['court3D'], np.float32)
    court2D = np.array(request.json['court2D'], np.float32)

    hfinder = Hfinder(camera_ks, dist, camera_ks, court3D, court2D)
    return jsonify(hfinder.getExtrinsic_mat().tolist())

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return 'No files part', 400

    uploaded_files = request.files.getlist('files')
    saved_files = []

    for file in uploaded_files:
        if file.filename:
            filename = os.path.basename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            saved_files.append(save_path)

    # 執行 main.py
    try:
        main_result = subprocess.run(
            ['python3.8', 'main.py', '--data_folder', UPLOAD_FOLDER, '--output_dir', PROCESSED_FOLDER],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if main_result.returncode != 0:
            return f'Error in main.py:\n{main_result.stderr}', 500
        print(main_result.stdout)

    except Exception as e:
        print(e.with_traceback())
        return f'Failed to execute main.py: {e}', 500

    # 執行 render3D.py
    try:
        csv_path = os.path.join(PROCESSED_FOLDER, 'Model3D.csv')
        second_result = subprocess.run(
            ['python3.8', 'render3D.py', '--csv', csv_path, '--output_dir', PROCESSED_FOLDER],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if second_result.returncode != 0:
            return f'Error in render3D.py:\n{second_result.stderr}', 500
        print(second_result.stdout)

    except Exception as e:
        print(e.with_traceback())
        return f'Failed to execute render3D.py: {e}', 500

    # 假設 render3D.py 生成處理後的影片 processed_video.mp4
    processed_video_path = os.path.join(PROCESSED_FOLDER, '3D_trajectory.mp4')
    if not os.path.exists(processed_video_path):
        return 'Processed video not found', 500

    # 返回處理後影片的路徑給前端
    return url_for('uploaded_file', filename=processed_video_path)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    directory = os.path.dirname(filename)
    filename = os.path.basename(filename)
    return send_from_directory(os.path.join('.', directory), filename)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
