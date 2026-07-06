# 1. IMPORT LIBRARIES
import os
import random
import time
import json
import pandas as pd
from datasets import load_dataset
from google.colab import userdata
from groq import Groq

# 2. GROQ CLIENT AND MODEL CONFIGURATION
api_key = userdata.get('GROQ_API_KEY')
client = Groq(api_key=api_key)

# Now that token usage is halved, you can try using the smarter model again!
# (If it hits token limits again, switch back to "llama-3.1-8b-instant")
selected_model = "llama-3.1-8b-instant"

EXPERIMENT_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95
}

# 3. DATA LOADING AND SAMPLING
print("1/4: Loading dataset...")
ds = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona", split="data")
# Set n=100 or n=200 as desired
df = pd.DataFrame(ds).sample(n=100, random_state=42).copy()
df.reset_index(drop=True, inplace=True)

# 4. 2x2 RANDOMIZATION
print("2/4: Setting up balanced 2x2 Experimental Design...")
market_contexts = ['Bull Market (High Growth)', 'Bear Market (Severe Downturn)']
transparencies = ['Standard AI (Performance metrics only)', 'Explainable AI (Detailed rationales)']

conditions = [(m, t) for m in market_contexts for t in transparencies]
multiplier = len(df) // 4
condition_list = conditions * multiplier
random.seed(42)
random.shuffle(condition_list)

df['Market_Context'] = [c[0] for c in condition_list]
df['Transparency'] = [c[1] for c in condition_list]

new_columns = ['Financial_Literacy', 'Tech_Savviness', 'Age', 'Gender', 'Education', 'Willingness_to_Invest']
for col in new_columns:
    df[col] = None

# 5. UNIFIED FUNCTION: EXTRACTION + SIMULATION (ALL-IN-ONE)
def process_complete_twin(row, max_attempts=3):
    market_desc = (
        "BULL MARKET: The economy is thriving, with consistent growth, high consumer confidence, and strong positive market trends over the last 12 months."
        if "Bull" in row['Market_Context'] else
        "BEAR MARKET: The economy is in a severe downturn, with high volatility, widespread financial losses, and significant uncertainty over the last 12 months."
    )

    xai_desc = (
        "EXPLAINABLE AI (XAI): You have a dashboard showing exact decision rationales, feature weights, and risk assessments."
        if "Explainable" in row['Transparency'] else
        "STANDARD AI: You receive only standard performance KPIs. The internal decision logic is proprietary and completely hidden."
    )

    # Unified prompt: performs demographic extraction and investment decision simultaneously
    prompt = f"""
    --- IDENTITY PROFILE ---
    {row['persona_summary']}

    --- SCENARIO ---
    You are evaluating a new investment fund completely managed by an AI algorithm.
    Current Environment: {market_desc}
    Algorithm Transparency: {xai_desc}

    --- TASK ---
    You must perform TWO steps based STRICTLY on your profile:
    STEP 1: Extract your personal demographic and psychological data.
    STEP 2: As a real human investor subject to biases and your extracted financial/tech literacy, decide your willingness to invest in this specific fund on a scale from 1 (Absolutely not willing) to 7 (Extremely willing).

    --- OUTPUT FORMAT ---
    Return ONLY a valid JSON object with EXACTLY these 6 keys. Do not add any extra text.
    "Financial_Literacy" (integer 1-7),
    "Tech_Savviness" (integer 1-7),
    "Age" (integer),
    "Gender" (string),
    "Education" (string),
    "Willingness_to_Invest" (integer 1-7)
    """

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=EXPERIMENT_CONFIG["temperature"],
                top_p=EXPERIMENT_CONFIG["top_p"],
                response_format={"type": "json_object"} # Forces JSON output
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"   [Debug] API or JSON Error from Groq (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None

print("3/4: Starting optimized simulation (All-in-One)...")
final_columns = ['pid', 'Market_Context', 'Transparency', 'Financial_Literacy', 'Tech_Savviness', 'Age', 'Gender', 'Education', 'Willingness_to_Invest']
file_name = "Simulated_Digital_Twins_Dataset_Optimized.csv"

if os.path.exists(file_name):
    df.update(pd.read_csv(file_name))
    completed = df['Willingness_to_Invest'].notna().sum()
    print(f"Resuming experiment: {completed}/{len(df)} profiles already processed.")

print("4/4: Executing API calls...")
for index, row in df.iterrows():
    if pd.notna(df.at[index, 'Willingness_to_Invest']):
        continue

    print(f"Processing Twin {index + 1}/{len(df)} (PID: {row['pid']})...")

    json_result = process_complete_twin(row)

    if json_result:
        for key in new_columns:
            df.at[index, key] = json_result.get(key)

        df[final_columns].to_csv(file_name, index=False)

        time.sleep(2.5)
    else:
        print(f"Error for Twin {index + 1}")
        
print("\nExperiment successfully completed!")
print(f"--> REMEMBER TO DOWNLOAD THE FILE: {file_name} FROM THE LEFT SIDEBAR!")
