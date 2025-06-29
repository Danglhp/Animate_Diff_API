from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import os
from typing import Optional
import logging

from pipeline import PoemToImagePipeline
from schemas import PoemRequest, PoemResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Poem to Image Animation API",
    description="API for converting Vietnamese poems to animated images using PhoCLIP and AnimateDiff",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None

# Store for background tasks
tasks = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    global pipeline
    try:
        logger.info("Initializing Poem-to-Image pipeline...")
        pipeline = PoemToImagePipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Poem to Image Animation API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }

@app.post("/generate", response_model=PoemResponse)
async def generate_animation(poem_request: PoemRequest, background_tasks: BackgroundTasks):
    """Generate animation from poem"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename
    if poem_request.output_filename:
        output_filename = f"{poem_request.output_filename}.gif"
    else:
        output_filename = f"animation_{task_id}.gif"
    
    output_path = output_dir / output_filename
    
    # Store task info
    tasks[task_id] = {
        "status": "processing",
        "output_path": str(output_path),
        "poem": poem_request.poem,
        "prompt_generation_mode": poem_request.prompt_generation_mode
    }
    
    # Add background task
    background_tasks.add_task(process_poem, task_id, poem_request.poem, str(output_path), poem_request.prompt_generation_mode)
    
    return PoemResponse(
        task_id=task_id,
        status="processing",
        message="Animation generation started",
        output_path=None
    )

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "output_path": task.get("output_path"),
        "poem": task.get("poem", ""),
        "prompt_generation_mode": task.get("prompt_generation_mode", "analysis_to_vietnamese")
    }

@app.get("/download/{task_id}")
async def download_animation(task_id: str):
    """Download the generated animation"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    output_path = Path(task["output_path"])
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Animation file not found")
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Animation generation not completed")
    
    return FileResponse(
        path=output_path,
        filename=output_path.name,
        media_type="image/gif"
    )

async def process_poem(task_id: str, poem: str, output_path: str, prompt_generation_mode: str = "analysis_to_vietnamese"):
    """Background task to process poem and generate animation"""
    try:
        logger.info(f"Processing poem for task {task_id} with mode: {prompt_generation_mode}")
        
        # Update task status
        tasks[task_id]["status"] = "processing"
        
        # Process the poem with the specified prompt generation mode
        result_path = pipeline.process(poem, output_path, prompt_generation_mode)
        
        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["output_path"] = result_path
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": task["status"],
                "output_path": task.get("output_path"),
                "poem": task.get("poem", "")[:100] + "..." if len(task.get("poem", "")) > 100 else task.get("poem", ""),
                "prompt_generation_mode": task.get("prompt_generation_mode", "analysis_to_vietnamese")
            }
            for task_id, task in tasks.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 