# =============================================================================
# GitHub Suspicious Commit Detector — FastAPI
# =============================================================================
# Install:
#   pip install fastapi uvicorn httpx joblib numpy pandas scikit-learn xgboost
#
# Run:
#   uvicorn app:app --reload --port 8000
#
# Endpoints:
#   POST /analyze          -> ek repo ke commits analyze karo
#   POST /predict/single   -> ek commit manually predict karo
#   GET  /health           -> server check
# =============================================================================

import math
import re
import os
import httpx
import numpy as np
import pandas as pd
import joblib

from fastapi      import FastAPI, HTTPException
from pydantic     import BaseModel
from typing       import Optional
from collections  import deque
from datetime     import datetime

# =============================================================================
# 1. APP SETUP
# =============================================================================
app = FastAPI(
    title       = "GitHub Suspicious Commit Detector",
    description = "GraphQL se commits fetch karke ML model se analyze karta hai",
    version     = "1.0.0"
)

MODEL_PATH = "github_suspicious_model_final.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file nahi mila: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

# =============================================================================
# 2. REQUEST / RESPONSE SCHEMAS
# =============================================================================
class RepoRequest(BaseModel):
    owner  : str
    repo   : str
    branch : str = "main"
    last_n : int = 20
    token  : str

class SingleCommitRequest(BaseModel):
    author_timestamp        : int
    commit_hash             : str
    commit_utc_offset_hours : float
    filename                : str
    n_additions             : int
    n_deletions             : int
    subject                 : str
    author_id               : int

class CommitResult(BaseModel):
    commit_hash : str
    subject     : str
    author_id   : int
    suspicious  : bool
    probability : float
    risk_level  : str

class AnalyzeResponse(BaseModel):
    repo             : str
    branch           : str
    total_analyzed   : int
    suspicious_count : int
    suspicious_pct   : float
    results          : list[CommitResult]

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
SENSITIVE_KEYWORDS = [
    'password','passwd','secret','token','api_key','api key',
    'auth','credential','private','hack','bypass','exploit',
    'admin','root','sudo','chmod','drop table','rm -rf',
    'hardcode','hardcoded','plain text','plaintext'
]
VAGUE_KEYWORDS  = ['fix','update','misc','wip','test','temp','todo','...','change','edit']
HIGH_RISK_EXTS  = {'env','pem','key','p12','pfx','cer','crt','secret',
                   'cfg','conf','ini','sh','bash','ps1','htpasswd','netrc','npmrc','pypirc'}
CODE_EXTS       = {'py','js','ts','java','go','rb','php','c','cpp','cs','rs'}
CONFIG_EXTS     = {'json','yaml','yml','toml','xml','properties'}

def kw_count(text, kw_list):
    t = str(text).lower()
    return sum(k in t for k in kw_list)

def char_entropy(text):
    t = str(text)
    if not t: return 0.0
    freq = [t.count(c) / len(t) for c in set(t)]
    return -sum(p * math.log2(p) for p in freq if p > 0)

def has_url(text):
    return int(bool(re.search(r'https?://', str(text))))

def has_issue_ref(text):
    return int(bool(re.search(r'(#\d+|[A-Z]+-\d+)', str(text))))

def commits_in_last_1h(timestamps):
    result, window = [], deque()
    for ts in timestamps:
        cutoff = ts - 3600
        while window and window[0] < cutoff:
            window.popleft()
        result.append(len(window))
        window.append(ts)
    return result

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['utc_offset']         = df['commit_utc_offset_hours'].fillna(0)
    df['abs_utc_offset']     = df['utc_offset'].abs()
    df['is_unusual_tz']      = (~df['utc_offset'].isin(
                                [-8,-7,-6,-5,-4,-3,0,1,2,3,5.5,8,9])).astype(int)
    df['commit_datetime']    = pd.to_datetime(df['author_timestamp'], unit='s')
    df['local_hour']         = ((df['commit_datetime'].dt.hour + df['utc_offset']) % 24).astype(int)
    df['commit_dow']         = df['commit_datetime'].dt.dayofweek
    df['is_weekend']         = (df['commit_dow'] >= 5).astype(int)
    df['is_night_commit']    = df['local_hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)
    df['is_off_hours']       = df['local_hour'].apply(lambda h: 1 if h < 8 or h > 18 else 0)
    df['total_changes']      = df['n_additions'] + df['n_deletions']
    df['change_ratio']       = df['n_additions'] / (df['n_deletions'] + 1)
    df['net_change']         = df['n_additions'] - df['n_deletions']
    df['log_total_change']   = np.log1p(df['total_changes'])
    df['is_large_commit']    = (df['total_changes'] > 150).astype(int)
    df['message_length']     = df['subject'].apply(lambda x: len(str(x)))
    df['word_count']         = df['subject'].apply(lambda x: len(str(x).split()))
    df['sensitive_kw_count'] = df['subject'].apply(lambda x: kw_count(x, SENSITIVE_KEYWORDS))
    df['vague_kw_count']     = df['subject'].apply(lambda x: kw_count(x, VAGUE_KEYWORDS))
    df['msg_entropy']        = df['subject'].apply(char_entropy)
    df['msg_has_url']        = df['subject'].apply(has_url)
    df['msg_has_issue_ref']  = df['subject'].apply(has_issue_ref)
    df['is_empty_message']   = (df['message_length'] <= 3).astype(int)
    df['file_ext']           = df['filename'].apply(
        lambda x: str(x).rsplit('.', 1)[-1].lower() if '.' in str(x) else 'none')
    df['is_high_risk_file']  = df['file_ext'].apply(lambda e: int(e in HIGH_RISK_EXTS))
    df['is_code_file']       = df['file_ext'].apply(lambda e: int(e in CODE_EXTS))
    df['is_config_file']     = df['file_ext'].apply(lambda e: int(e in CONFIG_EXTS))
    df['is_hidden_file']     = df['filename'].apply(lambda x: int(str(x).startswith('.')))
    df['path_depth']         = df['filename'].apply(lambda x: str(x).count('/'))
    df['hash_entropy']       = df['commit_hash'].apply(char_entropy)

    artifact = load_model()
    known = artifact['label_encoder'].classes_
    df['file_ext_encoded'] = artifact['label_encoder'].transform(
        df['file_ext'].where(df['file_ext'].isin(known), known[0])
    )

    df = df.sort_values(['author_id', 'author_timestamp']).reset_index(drop=True)
    author_stats = df.groupby('author_id').agg(
        author_total_commits    = ('author_timestamp',   'count'),
        author_avg_changes      = ('total_changes',      'mean'),
        author_night_rate       = ('is_night_commit',    'mean'),
        author_sensitive_kw_sum = ('sensitive_kw_count', 'sum'),
        author_weekend_rate     = ('is_weekend',         'mean'),
        author_high_risk_rate   = ('is_high_risk_file',  'mean'),
        author_avg_entropy      = ('msg_entropy',        'mean'),
    ).reset_index()
    df = df.merge(author_stats, on='author_id', how='left')
    df['prev_ts']         = df.groupby('author_id')['author_timestamp'].shift(1)
    df['time_since_last'] = (df['author_timestamp'] - df['prev_ts']).fillna(99999)
    df['is_burst_commit'] = (df['time_since_last'] < 300).astype(int)
    df['commits_last_1h'] = df.groupby('author_id')['author_timestamp'].transform(
        lambda x: commits_in_last_1h(x.values)
    )
    return df

def predict_df(df: pd.DataFrame) -> list[CommitResult]:
    artifact  = load_model()
    features  = artifact['features']
    threshold = artifact['threshold']
    model     = artifact['model']
    X         = df[features].fillna(0)
    prob      = model.predict_proba(X)[:, 1]
    flag      = (prob >= threshold).astype(bool)

    results = []
    for i, row in df.iterrows():
        p = float(prob[i])
        results.append(CommitResult(
            commit_hash = str(row['commit_hash']),
            subject     = str(row['subject'])[:120],
            author_id   = int(row['author_id']),
            suspicious  = bool(flag[i]),
            probability = round(p, 4),
            risk_level  = 'HIGH' if p > 0.75 else ('MEDIUM' if p > 0.45 else 'LOW')
        ))
    return results

# =============================================================================
# 4. GITHUB GRAPHQL FETCHER
# =============================================================================
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

def build_query(owner: str, repo: str, last_n: int, branch: str) -> str:
    return f"""
    {{
      repository(owner: "{owner}", name: "{repo}") {{
        object(expression: "{branch}") {{
          ... on Commit {{
            history(first: {min(last_n, 100)}) {{
              nodes {{
                oid
                committedDate
                message
                additions
                deletions
                author {{
                  user {{ databaseId }}
                  email
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """

async def fetch_commits(
    owner: str, repo: str, last_n: int, token: str, branch: str
) -> list[dict]:

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type" : "application/json",
    }
    query = build_query(owner, repo, last_n, branch)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GITHUB_GRAPHQL_URL,
            json    = {"query": query},
            headers = headers
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code = resp.status_code,
            detail      = f"GitHub API error: {resp.text}"
        )

    data = resp.json()

    if "errors" in data:
        raise HTTPException(
            status_code = 400,
            detail      = f"GraphQL error: {data['errors']}"
        )

    if not data.get('data') or not data['data'].get('repository'):
        raise HTTPException(
            status_code = 400,
            detail      = "Repo nahi mila — owner/repo check karo"
        )

    repo_data = data['data']['repository']

    if not repo_data.get('object'):
        raise HTTPException(
            status_code = 400,
            detail      = f"Branch '{branch}' nahi mili — sahi branch name dalo"
        )

    nodes = repo_data['object']['history']['nodes']

    commits = []
    for node in nodes:
        ts = int(datetime.fromisoformat(
            node['committedDate'].replace('Z', '+00:00')).timestamp())

        author    = node.get('author', {})
        user      = author.get('user') or {}
        author_id = user.get('databaseId', 0) or 0

        commits.append({
            'author_timestamp'        : ts,
            'commit_hash'             : node['oid'],
            'commit_utc_offset_hours' : 0.0,
            'filename'                : 'unknown',
            'n_additions'             : node.get('additions', 0),
            'n_deletions'             : node.get('deletions', 0),
            'subject'                 : node['message'].split('\n')[0][:200],
            'author_id'               : author_id,
        })

    return commits

# =============================================================================
# 5. ROUTES
# =============================================================================

@app.get("/health")
def health():
    model_ok = os.path.exists(MODEL_PATH)
    return {
        "status"    : "ok" if model_ok else "model missing",
        "model_path": MODEL_PATH,
        "model_ok"  : model_ok,
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(req: RepoRequest):
    raw_commits = await fetch_commits(
        req.owner, req.repo, req.last_n, req.token, req.branch
    )

    if not raw_commits:
        raise HTTPException(status_code=404, detail="Koi commit nahi mila")

    df               = pd.DataFrame(raw_commits)
    df               = engineer_features(df)
    results          = predict_df(df)
    suspicious_count = sum(1 for r in results if r.suspicious)

    return AnalyzeResponse(
        repo             = f"{req.owner}/{req.repo}",
        branch           = req.branch,
        total_analyzed   = len(results),
        suspicious_count = suspicious_count,
        suspicious_pct   = round(suspicious_count / len(results) * 100, 2),
        results          = results,
    )

# @app.post("/predict/single", response_model=CommitResult)
# def predict_single(req: SingleCommitRequest):
#     df      = pd.DataFrame([req.dict()])
#     df      = engineer_features(df)
#     results = predict_df(df)
#     return results[0]
