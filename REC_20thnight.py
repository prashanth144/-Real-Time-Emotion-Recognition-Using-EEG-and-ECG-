import joblib
import numpy as np
from pythonosc import dispatcher, osc_server

# --- SETTINGS ---
RECEIVER_IP = "100.108.22.92"
RECEIVER_PORT = 5006
OSC_ADDRESS = "/eeg/features"

# --- 1. Load Your 4-Class Trained Model ---
try:
    model = joblib.load('emotion_model.pkl')
    print("✅ 4-Class emotion model (happy, sad, calm, angry) loaded successfully.")
except FileNotFoundError:
    print("🛑 Error: 'emotion_model.pkl' not found. Please place the model file here.")
    exit()

# --- 2. Define Function to Handle Incoming Data ---
def eeg_handler(address, delta, theta, alpha, beta, gamma):
    try:
        features = np.array([[delta, theta, alpha, beta, gamma]])
        prediction = model.predict(features)
        emotion = prediction[0]
        print(f"Received 5 features  ==>  Predicted Emotion: {str(emotion).upper()} 🧠")
    except Exception as e:
        print(f"Error during classification: {e}")

# --- 3. Start the Server ---
if __name__ == "__main__":
    disp = dispatcher.Dispatcher()
    disp.map(OSC_ADDRESS, eeg_handler)

    server = osc_server.ThreadingOSCUDPServer((RECEIVER_IP, RECEIVER_PORT), disp)
    print(f"✅ Server is listening on {server.server_address}")
    print("Waiting for data from the lab PC... (Press Ctrl+C to stop)")
    server.serve_forever()
