# test_upload.py
import requests

url = 'http://192.168.18.97:5000/upload-audio'
file_path = r'C:\Users\lenovo\Downloads\TestVoiceSorahFatiha (online-audio-converter.com).mp3'   # ← use a real file path
user_id = '00000000-0000-0000-0000-000000000001'    # ← paste the ID you got earlier

with open(file_path, 'rb') as f:
    files = {'file': f}
    data = {'user_id': user_id}
    resp = requests.post(url, files=files, data=data)
    print(resp.status_code, resp.json())
