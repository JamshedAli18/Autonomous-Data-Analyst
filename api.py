from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from backend import DataAnalyzer
import json

app = FastAPI(title="Autonomous Data Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    """Health check"""
    return {
        "message": "Autonomous Data Analyst API is running!",
        "models": {
            "planner": "llama-3.3-70b-versatile",
            "code_generation": "openai/gpt-oss-120b",
            "explanation": "groq/compound-mini"
        }
    }

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Upload CSV and get analysis"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        analyzer = DataAnalyzer(file_path)
        result = analyzer.analyze()
        
        response = {
            "status": "success",
            "file_name": result["file_name"],
            "data_shape": f"{result['summary']['rows']} rows × {result['summary']['shape'][1]} columns",
            "columns": result['summary']['columns'],
            "models_used": {
                "step_1_planner": "llama-3.3-70b-versatile",
                "step_2_code_generation": "openai/gpt-oss-120b",
                "step_3_explanation": "groq/compound-mini"
            },
            "analysis_plan": result['analysis_plan'],
            "generated_code": result['generated_code'],
            "insights": result['insights'],
            "charts_generated": len(result['charts']),
            "chart_titles": result['chart_titles'],
            "chart_explanations": result['chart_explanations'],
            "chart_files": result['charts']
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}