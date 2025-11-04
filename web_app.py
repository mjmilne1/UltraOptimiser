from flask import Flask, jsonify, render_template_string
import subprocess
import json
import os
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraOptimiser Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        .button { background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
        .button:hover { background: #45a049; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .running { background: #d4edda; color: #155724; }
        .results { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }
        pre { background: #282c34; color: #abb2bf; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 UltraOptimiser Dashboard</h1>
        <div class="status running">
            ✅ Service Status: <strong>RUNNING</strong>
        </div>
        <a href="/optimize" class="button">Run Portfolio Optimization</a>
        <a href="/results" class="button">View Latest Results</a>
        <a href="/api/status" class="button">API Status</a>
        <div class="results">
            <h2>Quick Stats</h2>
            <p>Last Run: {{ last_run }}</p>
            <p>Optimizations Today: {{ run_count }}</p>
        </div>
    </div>
</body>
</html>
"""

run_count = 0
last_run = "Never"

@app.route('/')
def home():
    return render_template_string(DASHBOARD_HTML, last_run=last_run, run_count=run_count)

@app.route('/optimize')
def optimize():
    global run_count, last_run
    run_count += 1
    last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Run the optimization
    result = subprocess.run(['python', 'examples_real_world.py'], 
                          capture_output=True, text=True)
    
    return f"""
    <html>
    <head>
        <title>Optimization Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            pre {{ background: #282c34; color: #abb2bf; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            a {{ background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Optimization Results</h1>
            <p>Run at: {last_run}</p>
            <pre>{result.stdout}</pre>
            <br>
            <a href="/">Back to Dashboard</a>
        </div>
    </body>
    </html>
    """

@app.route('/results')
def results():
    # Try to read the CSV file if it exists
    try:
        df = pd.read_csv('outputs/portfolio_allocation.csv')
        result_html = df.to_html(classes='table table-striped')
    except:
        result_html = "<p>No results available yet. Run an optimization first.</p>"
    
    return f"""
    <html>
    <head><title>Results</title></head>
    <body style="font-family: Arial; margin: 40px;">
        <h1>Latest Portfolio Allocation</h1>
        {result_html}
        <br><a href="/">Back</a>
    </body>
    </html>
    """

@app.route('/api/status')
def api_status():
    return jsonify({
        "status": "running",
        "service": "UltraOptimiser",
        "version": "3.0.0",
        "last_run": last_run,
        "run_count": run_count,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    global run_count, last_run
    run_count += 1
    last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    result = subprocess.run(['python', 'examples_real_world.py'], 
                          capture_output=True, text=True)
    
    return jsonify({
        "status": "success",
        "timestamp": last_run,
        "output": result.stdout
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
