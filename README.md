# Autonomous Data Analyst

Professional AI-powered CSV data analysis platform.

## Features

- Upload CSV files
- AI-powered analysis using LangChain
- Intelligent chart generation
- Python code generation
- Simple explanations in plain English

## Models Used

- **Planner**: llama-3.3-70b-versatile
- **Code Generator**: openai/gpt-oss-120b
- **Explainer**: groq/compound-mini

## Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add Groq API key to `.env`
4. Run: `streamlit run app.py`

## Deployment on Streamlit Cloud

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app"
4. Select your repository
5. Add secrets in "Advanced settings"

## License

MIT
