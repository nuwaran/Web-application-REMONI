import re
from config import vital_sign_var_to_text
import matplotlib
matplotlib.use('Agg')  # For headless plotting
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# -------------------------------
# Helper Functions
# -------------------------------

def combine_data_and_time(list_date, list_time):
    """Combine lists of dates and times into full timestamps"""
    time_stamp_list = []
    for date in list_date:
        for time in list_time:
            time_stamp = f'{date} {time}'
            time_stamp_list.append(time_stamp)
    return time_stamp_list

def get_serial_path(data_folder_path):
    """Generate new sequential filename for images"""
    files = os.listdir(data_folder_path)
    image_files = [f for f in files if f.endswith('.jpg') and f[:-4].isdigit()]
    indices = [int(f.split('.')[0]) for f in image_files]
    max_index = max(indices) if indices else 0
    new_index = max_index + 1
    new_filename = f"{new_index:02d}.jpg"
    return os.path.join(data_folder_path, new_filename)

def extract_patient_id_from_text(text):
    """Extract patient ID from text"""
    pattern = r"\d{5}"
    match = re.search(pattern, text)
    return match.group(0) if match else 'unknown'

def filter_raw_df(df, intent_dict, is_current):
    """Filter dataframe based on user intent"""
    if df.empty:
        return df

    if not is_current:
        time_stamp_list = combine_data_and_time(intent_dict['list_date'], intent_dict['list_time'])
    else:
        time_stamp_list = [df['time_stamp'].iloc[0]]  # current data

    columns = ['time_stamp'] + intent_dict['vital_sign']
    filtered_df = df[columns]

    filtered_df = filtered_df[filtered_df['time_stamp'].isin(time_stamp_list)].reset_index(drop=True)
    return filtered_df

def df_to_text(df, intent_dict):
    """Convert dataframe into readable text"""
    if df.empty:
        return "No data available"

    text = ''
    title_row = 'Timestamp (Year-Month-Day Hour:Minute:Second)'

    for vital_sign in intent_dict['vital_sign']:
        title_row += f', {vital_sign_var_to_text.get(vital_sign, vital_sign)}'

    text += f'{title_row} \n'

    for i in range(len(df)):
        row_text = str(df.iloc[i]['time_stamp'])
        for vital_sign in intent_dict['vital_sign']:
            row_text += f', {df.iloc[i][vital_sign]}'
        text += f'{row_text} \n'

    return text

def plot_vital_sign(df, vital_sign):
    """Plot vital sign data"""
    if df.shape[0] <= 20:
        df_sampled = df
    else:
        sample_rate = len(df) // 20
        df_sampled = df.iloc[::sample_rate].copy()

    figure = plt.figure(figsize=(7,5))
    x = df_sampled['time_stamp']
    y = df_sampled[vital_sign]
    plt.plot(x, y)
    plt.xticks(rotation=90)
    save_path = f'./static/local_data/show_data/plot_{vital_sign}.png'
    figure.savefig(save_path, bbox_inches='tight')
    plt.close(figure)
    return save_path

def extract_unique_year_month(date_list):
    """Get unique year_month strings from date list"""
    year_month_set = {date[:7].replace('-', '_') for date in date_list}
    return sorted(list(year_month_set))

def process_key_to_retrieve_image(timestamp_list):
    """Group timestamps for S3 retrieval"""
    grouped_timestamps = defaultdict(list)
    for timestamp in timestamp_list:
        date, time = timestamp.split(' ')
        year, month, day = date.split('-')
        year_month = f'{year}_{month}'
        day_time = f'{day}_{time.replace(":", "_")}'
        day_time = day_time[:-3]  # remove seconds
        grouped_timestamps[year_month].append(day_time)
    return dict(grouped_timestamps)
