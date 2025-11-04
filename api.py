"""
FastAPI wrapper for UltraOptimiser
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
from core.optimizer_v2 import UltraOptimiser, Goal

app = FastAPI(title="UltraOptimiser API", version="2.0.0")


class GoalRequest(BaseModel):
    name: str
    target_amount: float
    time_horizon: float
    risk_tolerance: float
    priority: float


class OptimizationRequest(BaseModel):
    n_assets: int
    expected_returns: List[float]
    covariance_matrix: List[List[float]]
    goals: List[GoalRequest]
    constraints: Optional[Dict] = None


@app.get("/")
def read_root():
    return {
        "name": "UltraOptimiser API",
        "version": "2.0.0",
        "description": "Advanced portfolio optimization with multi-objective Lagrangian"
    }


@app.post("/optimize")
def optimize(request: OptimizationRequest):
    try:
        # Initialize optimizer
        optimizer = UltraOptimiser(request.n_assets)
        
        # Convert goals
        goals = [
            Goal(
                g.name,
                g.target_amount,
                g.time_horizon,
                g.risk_tolerance,
                g.priority
            )
            for g in request.goals
        ]
        
        # Convert to numpy arrays
        returns = np.array(request.expected_returns)
        cov_matrix = np.array(request.covariance_matrix)
        
        # Run optimization
        result = optimizer.optimize(
            returns,
            cov_matrix,
            goals,
            constraints=request.constraints
        )
        
        return {
            "success": True,
            "result": result.to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}
