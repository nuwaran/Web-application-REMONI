import eventlet

eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from nlp_engine import nlp_engine
import pandas as pd
from datetime import datetime, timedelta
import os
from request_to_openai import gpt
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import time
import re
import boto3
import json
from io import StringIO
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------
# Flask & SocketIO setup
# -------------------------------
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# ==================== AWS S3 Configuration ====================
S3_KEY_ID = os.getenv("S3_KEY_ID", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "remonitest")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PATIENT_ID = os.getenv("PATIENT_ID", "00001")

# Polling interval for S3 updates (seconds)
S3_POLL_INTERVAL = 5

# Initialize S3 client
try:
    s3_client = boto3.client(
        service_name='s3',
        region_name=AWS_REGION,
        aws_access_key_id=S3_KEY_ID,
        aws_secret_access_key=S3_SECRET_KEY
    )
    logger.info("✅ AWS S3 client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize S3 client: {e}")
    s3_client = None

# Storage for vitals and alerts
latest_vitals = {
    'heart_rate': 0,
    'spo2': 0,
    'blood_pressure': {'systolic': 0, 'diastolic': 0},
    'skin_temperature': 0,
    'timestamp': 0,
    'datetime': 'Never',
    'patient_id': PATIENT_ID
}

fall_alerts = []
vitals_df = pd.DataFrame()

# Track last signal file timestamp to detect new updates
last_signal_timestamp = 0
last_wifi_timestamp = 0

# Store latest WiFi connection info
latest_wifi_connection = None

# Track which alert IDs have been emitted to prevent duplicates
emitted_alert_ids = set()  # Store IDs of alerts already sent to chatbox

# -------------------------------
# Local storage paths
# -------------------------------
PLOT_FOLDER = './static/local_data/show_data/'
os.makedirs(PLOT_FOLDER, exist_ok=True)


# ==================== S3 Data Fetching Functions ====================
def fetch_vitals_from_s3():
    """Fetch latest vitals data from S3 (current month's CSV)"""
    global vitals_df, latest_vitals

    if not s3_client:
        logger.warning("S3 client not available")
        return False

    try:
        # Get current year-month
        year_month = datetime.now().strftime("%Y-%m")
        s3_key = f"{PATIENT_ID}/time_series/{year_month}.csv"

        # Download CSV from S3
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        csv_data = obj['Body'].read().decode('utf-8')

        # Load into DataFrame
        df = pd.read_csv(StringIO(csv_data))

        if not df.empty:
            vitals_df = df

            # Update latest vitals from most recent row
            latest_row = df.iloc[-1]
            latest_vitals = {
                'heart_rate': int(latest_row.get('heart_rate', 0)),
                'spo2': int(latest_row.get('spo2', 0)),
                'blood_pressure': {
                    'systolic': int(latest_row.get('blood_pressure_systolic', 0)),
                    'diastolic': int(latest_row.get('blood_pressure_diastolic', 0))
                },
                'skin_temperature': float(latest_row.get('skin_temperature', 0)),
                'timestamp': int(pd.Timestamp(latest_row.get('time_stamp')).timestamp() * 1000),
                'datetime': str(latest_row.get('time_stamp', 'Never')),
                'patient_id': str(latest_row.get('patient_id', PATIENT_ID))
            }

            logger.info(f"✅ Fetched {len(df)} vitals records from S3")
            return True

    except s3_client.exceptions.NoSuchKey:
        logger.info(f"No vitals data found in S3 for {year_month}")
        return False
    except Exception as e:
        logger.error(f"❌ Error fetching vitals from S3: {e}")
        return False


def fetch_fall_alerts_from_s3():
    """Fetch today's fall alerts from S3 and emit ONLY NEW alerts to clients"""
    global fall_alerts, emitted_alert_ids

    if not s3_client:
        return False

    try:
        # Get today's date
        date_str = datetime.now().strftime("%Y-%m-%d")
        s3_key = f"{PATIENT_ID}/fall_alerts/{date_str}.json"

        # Download fall alerts JSON
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        alerts_data = obj['Body'].read().decode('utf-8')
        alerts = json.loads(alerts_data)

        # Only add new alerts (not already in our list)
        existing_ids = {alert['id'] for alert in fall_alerts}
        new_alerts = [alert for alert in alerts if alert['id'] not in existing_ids]

        if new_alerts:
            fall_alerts.extend(new_alerts)
            logger.info(f"✅ Fetched {len(new_alerts)} new fall alerts from S3")

            # Emit ONLY alerts that haven't been emitted before
            for alert in new_alerts:
                alert_id = alert.get('id')

                # Skip if already emitted
                if alert_id in emitted_alert_ids:
                    logger.debug(f"Skipping already emitted alert ID {alert_id}")
                    continue

                # SIMPLE PAYLOAD - just the basics
                simple_payload = {
                    "patient_id": alert.get("patient_id", PATIENT_ID),
                    "confidence": alert.get("confidence", 0),
                    "datetime": alert.get("datetime", "Unknown"),
                    "type": "fall_alert"
                }

                logger.info(f"🚨 Emitting NEW fall alert to chatbox:")
                logger.info(f"   Alert ID: {alert_id}")
                logger.info(f"   Patient: {simple_payload['patient_id']}")
                logger.info(f"   Confidence: {simple_payload['confidence']}%")
                logger.info(f"   Time: {simple_payload['datetime']}")

                # Emit to all connected clients
                socketio.emit('fall_alert', simple_payload, namespace='/')

                # Mark as emitted
                emitted_alert_ids.add(alert_id)

            logger.info(f"✅ {len(new_alerts)} NEW fall alerts emitted to chatbox")
            logger.info(f"📊 Total emitted alerts tracked: {len(emitted_alert_ids)}")

        return True

    except s3_client.exceptions.NoSuchKey:
        logger.info(f"No fall alerts found for today")
        return False
    except Exception as e:
        logger.error(f"❌ Error fetching fall alerts from S3: {e}")
        return False


def fetch_wifi_connection_from_s3():
    """Fetch WiFi connection info from S3"""
    global latest_wifi_connection, last_wifi_timestamp

    if not s3_client:
        return False

    try:
        s3_key = f"{PATIENT_ID}/wifi_connection.json"

        # Download WiFi connection info
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        wifi_data = obj['Body'].read().decode('utf-8')

        wifi_info = json.loads(wifi_data)
        wifi_timestamp = wifi_info.get('timestamp', 0)

        # Check if this is a new connection
        if wifi_timestamp > last_wifi_timestamp:
            last_wifi_timestamp = wifi_timestamp
            latest_wifi_connection = wifi_info

            logger.info(f"📶 New WiFi connection detected from S3")
            logger.info(f"   SSID: {wifi_info.get('ssid')}")
            logger.info(f"   IP: {wifi_info.get('ip_address')}")

            # Send notification to chatbox
            ssid = wifi_info.get('ssid', 'Unknown')
            ip_address = wifi_info.get('ip_address', 'Unknown')
            timestamp_str = wifi_info.get('datetime', 'Unknown')

            wifi_message = (
                f"📶 RASPBERRY PI CONNECTED!\n\n"
                f"✅ WiFi Network: {ssid}\n"
                f"🌐 IP Address: {ip_address}\n"
                f"⏰ Connected at: {timestamp_str}\n"
                f"🔌 Patient ID: {wifi_info.get('patient_id', PATIENT_ID)}\n\n"
                f"🎉 System is now online and ready!\n"
                f"💡 Data is being synced via AWS S3 cloud"
            )

            # Send to all connected clients via SocketIO
            socketio.emit('chat_message', {
                'type': 'wifi_notification',
                'message': wifi_message,
                'timestamp': timestamp_str,
                'wifi_data': wifi_info
            }, namespace='/')

            logger.info("✅ WiFi notification sent to chatbox")
            return True

    except s3_client.exceptions.NoSuchKey:
        logger.debug("No WiFi connection file found in S3 yet")
        return False
    except Exception as e:
        logger.error(f"❌ Error fetching WiFi connection from S3: {e}")
        return False


def check_signal_file():
    """Check S3 signal file for new updates"""
    global last_signal_timestamp

    if not s3_client:
        return False

    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='signal_file.txt')
        signal_data = json.loads(obj['Body'].read().decode('utf-8'))

        signal_timestamp = signal_data.get('timestamp', 0)
        signal_type = signal_data.get('type', '')

        # Check if this is a new signal
        if signal_timestamp > last_signal_timestamp:
            last_signal_timestamp = signal_timestamp
            logger.info(f"📡 New signal detected: {signal_type}")

            # Fetch appropriate data based on signal type
            if signal_type == 'vitals_update':
                fetch_vitals_from_s3()
                socketio.emit('vitals_update', latest_vitals, namespace='/')

            elif signal_type == 'fall_alert':
                fetch_fall_alerts_from_s3()

            elif signal_type == 'wifi_notification':
                fetch_wifi_connection_from_s3()

            return True

    except s3_client.exceptions.NoSuchKey:
        logger.debug("No signal file found yet")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking signal file: {e}")
        return False


def s3_polling_loop():
    """Background thread to poll S3 for updates"""
    global emitted_alert_ids

    logger.info("🔄 Starting S3 polling loop...")

    # Initial data fetch (load existing alerts but DON'T emit them on startup)
    fetch_vitals_from_s3()

    # Load existing alerts but mark them as already emitted
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        s3_key = f"{PATIENT_ID}/fall_alerts/{date_str}.json"
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        alerts_data = obj['Body'].read().decode('utf-8')
        alerts = json.loads(alerts_data)

        # Load into memory but mark all as already emitted
        fall_alerts.extend(alerts)
        emitted_alert_ids = {alert['id'] for alert in alerts}
        logger.info(f"📥 Loaded {len(alerts)} existing alerts (marked as already emitted)")
        logger.info(f"   These alerts won't be shown on page reload")
    except:
        logger.info("No existing alerts to load")

    fetch_wifi_connection_from_s3()

    while True:
        try:
            # Check for new updates via signal file
            check_signal_file()

            time.sleep(S3_POLL_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Error in polling loop: {e}")
            time.sleep(S3_POLL_INTERVAL)


# ==================== Flask Routes ====================
@app.route("/")
def index():
    return render_template('doctor.html')


@app.route("/api/s3_status", methods=['GET'])
def get_s3_status():
    """Check S3 connection status"""
    return jsonify({
        'connected': s3_client is not None,
        'bucket': S3_BUCKET_NAME,
        'region': AWS_REGION,
        'patient_id': PATIENT_ID,
        'vitals_records': len(vitals_df),
        'fall_alerts': len(fall_alerts),
        'wifi_connection': latest_wifi_connection
    })


@app.route("/api/latest_vitals", methods=['GET'])
def get_latest_vitals():
    """Get latest cached vitals"""
    return jsonify(latest_vitals)


@app.route("/api/fall_alerts", methods=['GET'])
def get_fall_alerts():
    """Get fall alerts history"""
    return jsonify({
        'total': len(fall_alerts),
        'alerts': fall_alerts[-10:],
        'latest': fall_alerts[-1] if fall_alerts else None
    })


@app.route("/api/wifi_connection", methods=['GET'])
def get_wifi_connection():
    """Get latest WiFi connection info"""
    return jsonify({
        'connected': latest_wifi_connection is not None,
        'connection_info': latest_wifi_connection
    })


@app.route("/api/refresh_data", methods=['POST'])
def refresh_data():
    """Manually trigger data refresh from S3"""
    vitals_success = fetch_vitals_from_s3()
    alerts_success = fetch_fall_alerts_from_s3()
    wifi_success = fetch_wifi_connection_from_s3()

    return jsonify({
        'vitals_updated': vitals_success,
        'alerts_updated': alerts_success,
        'wifi_updated': wifi_success,
        'vitals_count': len(vitals_df),
        'alerts_count': len(fall_alerts),
        'emitted_count': len(emitted_alert_ids),
        'wifi_info': latest_wifi_connection
    })


@app.route("/api/clear_emitted_alerts", methods=['POST'])
def clear_emitted_alerts():
    """Clear emitted alerts tracker (for testing)"""
    global emitted_alert_ids

    old_count = len(emitted_alert_ids)
    emitted_alert_ids.clear()

    logger.info(f"🧹 Cleared {old_count} emitted alert IDs")

    return jsonify({
        'status': 'success',
        'message': f'Cleared {old_count} emitted alerts',
        'emitted_count': len(emitted_alert_ids)
    })


# -------------------------------
# Helper Functions
# -------------------------------
def filter_df_by_time_range(df, minutes=10):
    """Filter dataframe by time range"""
    if df.empty:
        return df
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    return df[df['time_stamp'] >= cutoff_time].copy()


def create_plot(df, vital_sign, time_range_minutes=None):
    """Create plot for vital sign"""
    if df.empty:
        return None

    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.sort_values('time_stamp')
    df_clean = df[df[vital_sign].notna()].copy()

    if df_clean.empty:
        return None

    plt.figure(figsize=(10, 6))
    plt.plot(df_clean['time_stamp'], df_clean[vital_sign],
             marker='o', linestyle='-', linewidth=2, markersize=6, color='#2196F3')

    title = f"{vital_sign.replace('_', ' ').title()}"
    if time_range_minutes:
        title += f" - Last {time_range_minutes} Minutes"

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)

    ylabel = vital_sign.replace('_', ' ').title()
    if vital_sign == 'heart_rate':
        ylabel += ' (BPM)'
    elif vital_sign == 'spo2':
        ylabel += ' (%)'
    elif vital_sign == 'skin_temperature':
        ylabel += ' (°C)'
    elif 'blood_pressure' in vital_sign:
        ylabel += ' (mmHg)'

    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_filename = f'plot_{vital_sign}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plot_path = os.path.join(PLOT_FOLDER, plot_filename)
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    return f'/static/local_data/show_data/{plot_filename}'


# -------------------------------
# Chat Endpoint
# -------------------------------
@app.route("/chat", methods=['POST'])
def chat():
    """AI-powered chat endpoint using S3 data"""
    question = request.get_json().get("message", "")
    question_lower = question.lower()

    # ========== WIFI CONNECTION QUERIES ==========
    wifi_keywords = ['wifi', 'connection', 'ip address', 'raspberry pi', 'connected', 'network']
    if any(k in question_lower for k in wifi_keywords):
        # Refresh WiFi info from S3
        fetch_wifi_connection_from_s3()

        if latest_wifi_connection:
            wifi_info = f"""
WiFi Connection Status:
• Status: Connected ✓
• SSID: {latest_wifi_connection.get('ssid', 'Unknown')}
• IP Address: {latest_wifi_connection.get('ip_address', 'Unknown')}
• Connected at: {latest_wifi_connection.get('datetime', 'Unknown')}
• Patient ID: {latest_wifi_connection.get('patient_id', PATIENT_ID)}
• Data Source: AWS S3 Cloud

The Raspberry Pi is online and syncing data via S3.
"""
            return jsonify({"answer": wifi_info})
        else:
            wifi_info = """
WiFi Connection Status:
• Status: Not connected yet
• Waiting for Raspberry Pi to connect...

To connect:
1. Join "REMONI-Setup" WiFi (password: remoni2024)
2. Access http://192.168.4.1
3. Configure your WiFi network
4. IP address will be uploaded to S3
5. Connection details will appear here automatically
"""
            return jsonify({"answer": wifi_info})

    # ========== CURRENT VITALS QUERIES ==========
    current_keywords = ['latest', 'current', 'now', 'right now']
    vitals_keywords = ['vitals', 'blood pressure', 'spo2', 'oxygen', 'heart rate', 'temperature', 'bp']

    if any(k in question_lower for k in current_keywords) and any(k in question_lower for k in vitals_keywords):
        # Refresh data from S3
        fetch_vitals_from_s3()

        if latest_vitals.get('heart_rate', 0) > 0:
            bp = latest_vitals.get('blood_pressure', {})
            vitals_text = f"""
Current Vital Signs:
• Heart Rate: {latest_vitals.get('heart_rate', 0)} BPM
• SpO2: {latest_vitals.get('spo2', 0)}%
• Blood Pressure: {bp.get('systolic', 0)}/{bp.get('diastolic', 0)} mmHg
• Skin Temperature: {latest_vitals.get('skin_temperature', 0)}°C
• Last Updated: {latest_vitals.get('datetime', 'N/A')}
• Data Source: AWS S3
"""
            system_prompt = """You are a medical assistant analyzing vital signs. Provide professional insights."""
            prompt = f"Question: {question}\n\n{vitals_text}\n\nProvide a clear response."
            gpt_reply = gpt(text=prompt, model_name="gpt-3.5-turbo", system_prompt=system_prompt)
            return jsonify({"answer": gpt_reply})

        return jsonify({"answer": "No vitals data available yet. Please ensure data is being uploaded to S3."})

    # ========== FALL ALERT QUERIES ==========
    fall_keywords = ['fall', 'alert', 'emergency', 'incident']
    if any(k in question_lower for k in fall_keywords):
        # Refresh alerts from S3
        fetch_fall_alerts_from_s3()

        if fall_alerts:
            latest = fall_alerts[-1]
            fall_info = f"""
Latest Fall Detection:
• Patient: {latest['patient_id']}
• Confidence: {latest.get('confidence', 0):.1f}%
• Time: {latest.get('datetime', 'Unknown')}

Total Falls Today: {len(fall_alerts)}
"""
            system_prompt = """You are a medical assistant. Analyze fall data and provide recommendations."""
            prompt = f"Question: {question}\n\n{fall_info}\n\nProvide professional response."
            gpt_reply = gpt(text=prompt, model_name="gpt-3.5-turbo", system_prompt=system_prompt)
            return jsonify({"answer": gpt_reply})

        return jsonify({"answer": "No fall alerts detected yet. Fall detection is active."})

    # ========== SYSTEM STATUS ==========
    status_keywords = ['status', 'connected', 'working', 'online', 'system']
    if any(k in question_lower for k in status_keywords):
        wifi_status = "Not Connected"
        wifi_ip = "N/A"
        if latest_wifi_connection:
            wifi_status = "Connected ✓"
            wifi_ip = latest_wifi_connection.get('ip_address', 'N/A')

        status_text = f"""
System Status:
• Data Source: AWS S3 (Cloud-based)
• S3 Connection: {"Connected ✓" if s3_client else "Disconnected ✗"}
• Bucket: {S3_BUCKET_NAME}
• Patient ID: {PATIENT_ID}
• WiFi Status: {wifi_status}
• Raspberry Pi IP: {wifi_ip}
• Fall Alerts: {len(fall_alerts)}
• Vitals Records: {len(vitals_df)}
• Last Update: {latest_vitals.get('datetime', 'Never')}
• Polling Interval: {S3_POLL_INTERVAL}s
• Alert Mode: SIMPLE (minimal data)
"""
        return jsonify({"answer": status_text})

    # ========== HISTORICAL DATA & PLOTS ==========
    time_range = None
    match_min = re.search(r'(\d+)\s*minute', question_lower)
    match_hr = re.search(r'(\d+)\s*hour', question_lower)
    if match_min:
        time_range = int(match_min.group(1))
    elif match_hr:
        time_range = int(match_hr.group(1)) * 60

    plot_keywords = ['plot', 'graph', 'chart', 'visualize', 'show']
    is_plot = any(k in question_lower for k in plot_keywords)

    historical_keywords = ['history', 'trend', 'over time', 'past', 'average']
    is_historical = any(k in question_lower for k in historical_keywords)

    # Detect vital signs
    vital_signs = []
    mapping = {
        'heart rate': ['heart_rate'], 'hr': ['heart_rate'], 'pulse': ['heart_rate'],
        'spo2': ['spo2'], 'oxygen': ['spo2'],
        'blood pressure': ['blood_pressure_systolic', 'blood_pressure_diastolic'],
        'bp': ['blood_pressure_systolic', 'blood_pressure_diastolic'],
        'temperature': ['skin_temperature'], 'temp': ['skin_temperature']
    }

    for keyword, cols in mapping.items():
        if keyword in question_lower:
            vital_signs.extend(cols)
            break

    # Generate plots
    if is_plot and vital_signs:
        # Refresh data from S3
        fetch_vitals_from_s3()

        if vitals_df.empty:
            return jsonify({"answer": "No historical data available yet."})

        df = vitals_df.copy()
        if time_range:
            df = filter_df_by_time_range(df, time_range)
            if df.empty:
                return jsonify({"answer": f"No data in last {time_range} minutes."})

        plot_paths = []
        for vital in vital_signs:
            if vital in df.columns:
                path = create_plot(df, vital, time_range)
                if path:
                    plot_paths.append(path)

        if plot_paths:
            time_info = f"last {time_range} minutes" if time_range else "all available data"
            return jsonify({
                "answer": f"Created plots for {', '.join([v.replace('_', ' ') for v in vital_signs])} ({time_info}).",
                "plots": plot_paths
            })
        return jsonify({"answer": "Could not generate plots from available data."})

    # Analyze historical data
    if is_historical or vital_signs:
        # Refresh data from S3
        fetch_vitals_from_s3()

        if vitals_df.empty:
            return jsonify({"answer": "No historical data available yet."})

        df = vitals_df.copy()
        if time_range:
            df = filter_df_by_time_range(df, time_range)

        analysis = f"Historical Analysis ({len(df)} records from S3):\n\n"
        for vital in ['heart_rate', 'spo2', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'skin_temperature']:
            if vital in df.columns:
                values = df[vital].dropna()
                if len(values) > 0:
                    analysis += f"• {vital.replace('_', ' ').title()}:\n"
                    analysis += f"  Latest: {values.iloc[-1]}, "
                    analysis += f"Avg: {values.mean():.1f}, "
                    analysis += f"Min: {values.min()}, Max: {values.max()}\n\n"

        system_prompt = """You are a medical assistant. Analyze historical vitals and provide insights."""
        prompt = f"Question: {question}\n\n{analysis}\n\nProvide professional analysis."
        gpt_reply = gpt(text=prompt, model_name="gpt-3.5-turbo", system_prompt=system_prompt)
        return jsonify({"answer": gpt_reply})

    # ========== GENERAL CONVERSATION ==========
    system_prompt = """You are REMONI, a medical assistant for patient monitoring. 
You help with: real-time vitals from AWS S3, historical analysis, fall detection, WiFi configuration via S3, and system status.
Fall alerts are sent in SIMPLE format with minimal data."""
    gpt_reply = gpt(text=question, model_name="gpt-3.5-turbo", system_prompt=system_prompt)
    return jsonify({"answer": gpt_reply})


# ==================== Debug Endpoint ====================
@app.route("/debug_data", methods=['GET'])
def debug_data():
    """Debug endpoint to check data"""
    latest_row = vitals_df.iloc[-1].to_dict() if not vitals_df.empty else {}

    return jsonify({
        "data_source": "AWS S3",
        "s3_bucket": S3_BUCKET_NAME,
        "patient_id": PATIENT_ID,
        "columns": list(vitals_df.columns) if not vitals_df.empty else [],
        "latest_row": latest_row,
        "total_records": len(vitals_df),
        "s3_connected": s3_client is not None,
        "cached_vitals": latest_vitals,
        "fall_alerts_count": len(fall_alerts),
        "emitted_alerts_count": len(emitted_alert_ids),
        "emitted_alert_ids": list(emitted_alert_ids),
        "last_signal_timestamp": last_signal_timestamp,
        "wifi_connection": latest_wifi_connection,
        "last_wifi_timestamp": last_wifi_timestamp,
        "alert_mode": "SIMPLE (minimal data)",
        "note": "Only NEW alerts are emitted to chatbox"
    })


# ==================== Run Server ====================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🏥 REMONI WEB APP - SIMPLE FALL ALERTS (PORT 5001)")
    print("=" * 70)
    print(f"☁️  Data Source: AWS S3")
    print(f"📦 S3 Bucket: {S3_BUCKET_NAME}")
    print(f"🌍 Region: {AWS_REGION}")
    print(f"👤 Patient ID: {PATIENT_ID}")
    print(f"🔄 Polling Interval: {S3_POLL_INTERVAL}s")
    print(f"📶 WiFi Monitoring: ENABLED (via S3)")
    print(f"🚨 Fall Alerts: SIMPLE MODE (minimal data)")
    print("=" * 70)
    print("✓ Raspberry Pi and Web App on different networks")
    print("✓ All communication via AWS S3 cloud")
    print("✓ WiFi IP notifications synced automatically")
    print("✓ Simple fall alerts - just patient, confidence, time")
    print("=" * 70 + "\n")

    # Start S3 polling in background
    threading.Thread(target=s3_polling_loop, daemon=True).start()

    print("🚀 Starting web server on http://0.0.0.0:5001\n")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
