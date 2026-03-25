# GitHub Suspicious Commit Detector

ML-powered tool jo GitHub commits ko analyze karke suspicious activity detect karta hai.

## Problem Statement
GitHub repositories mein malicious commits detect karna mushkil hota hai —
credentials leak, backdoor injection, aur suspicious patterns ko manually
dhundhna time-consuming hai.

## Solution
XGBoost model jo 25+ features analyze karke real-time mein suspicious
commits detect karta hai — 99.35% ROC-AUC accuracy ke saath.

## Tech Stack
- Python, FastAPI, XGBoost
- GitHub GraphQL API
- Scikit-learn, Pandas, NumPy

## Features
- Real-time commit analysis
- GitHub GraphQL integration
- Risk scoring (LOW / MEDIUM / HIGH)
- 99.35% ROC-AUC accuracy
- 1.4M commits pe trained

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Server status |
| POST | /analyze | Repo commits analyze karo |
| POST | /predict/single | Single commit test karo |

## Installation
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

## Sample Request
POST /analyze
{
  "owner"  : "torvalds",
  "repo"   : "linux",
  "branch" : "master",
  "last_n" : 20,
  "token"  : "your_github_token"
}

## Sample Response
{
  "total_analyzed"   : 20,
  "suspicious_count" : 2,
  "suspicious_pct"   : 10.0,
  "results": [...]
}

## Backend Services By
- Ritik Yadav