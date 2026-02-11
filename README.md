# Agentic-ICU-Patient-Monitoring-App


Author: Khalid Jiwani
 
**Purpose**
This project demonstrates a modern, AI-assisted workflow for ICU patient monitoring.
The app allows clinicians and students to upload patient vital sign data, visualize trends, detect critical abnormalities, and receive both procedural and medical recommendations powered by AI.

**Problem Solved: **

•	Rapid identification of clinical emergencies in ICU patient data.
•	Provides AI-driven support to guide ICU nurses and clinicians with best-practice procedural actions and general medical considerations.

**Relation to AI/AI-assisted Workflows:**

•	Integrates OpenAI's API to dynamically generate agentic recommendations and medical insights based on real patient data.
•	Features AI observability, tracking API token usage and latency for transparency and optimization.
 
**What the Code Does**

•	Data Ingestion: Accepts patient CSV files, preprocesses, and validates.
•	Abnormality Detection: Flags dangerous values (e.g., abnormal heart rate, low oxygen saturation).
•	AI Recommendations: 
        o	Sends contextual prompts to OpenAI API.
        o	Returns step-by-step procedural actions for ICU nurses.
        o	Provides general medical recommendations for educational purposes (not real clinical advice).
•	Emergency Alerts: Flashes urgent notifications for detected emergencies.
•	Visualization: Displays patient trends, summary statistics, and timeline with highlighted criticals.
•	AI Observability: Tracks and visualizes API token usage and latency for each AI call.
•	Reporting: Allows download of patient data and AI-generated recommendations.
 
**How to Run or Use**

**Prerequisites:**

•	Python 3.x
•	Required packages: streamlit, pandas, openai, plotly, python-dotenv

Setup Steps:

1.	Clone this repository
git clone https://github.com/your-username/your-repo.git
2.	Install dependencies
pip install -r requirements.txt
3.	Set up your OpenAI API key 
o	Create a .env file with:
OPENAI_API_KEY=your_openai_api_key_here
o	Do NOT commit .env or any API keys to GitHub!
4.	Run the app
streamlit run your_script.py
5.	Upload a patient CSV file and use the sidebar to explore features.

**Security & Safe Sharing**

IMPORTANT: To maintain security and protect sensitive information:
•	No API Keys: The OPENAI_API_KEY is managed via a .env file which is not included in this repository.
•	Gitignore: A .gitignore file should be used to ensure .env and other local configuration files are never uploaded to GitHub.
•	No Private Data: This script does not store or transmit personal user data beyond the session requirements for the OpenAI API.

**License**

This project is for academic demonstration only.
Not for real clinical use.
All recommendations are for educational purposes.

<img width="540" height="705" alt="image" src="https://github.com/user-attachments/assets/3a38e1c8-3046-4b1f-a3f2-bb0175c10fe6" />

