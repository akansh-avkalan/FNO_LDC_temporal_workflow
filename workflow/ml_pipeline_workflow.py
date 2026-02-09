# workflow\ml_pipeline_workflow.py
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy 
from typing import Dict, Tuple
import asyncio

with workflow.unsafe.imports_passed_through():
    from shared import (
        MLPipelineInput,
        EvaluateConfig
    )
    from activities.dataset_download import dataset_download
    from activities.train_FNO import train_FNO
    from activities.evaluate_FNO import evaluate_FNO

@workflow.defn
class MLPipelineWorkflow: 
    @workflow.run
    async def run(self, inputs: MLPipelineInput) -> Dict:
        # Unpack the config
        dataset_config = inputs.dataset_config
        
        workflow.logger.info("INFO: Starting ML pipeline workflow")
        
        # Step 1: Download dataset
        path: str = await workflow.execute_activity(
            dataset_download, 
            dataset_config, 
            start_to_close_timeout=timedelta(minutes=20),
            retry_policy=RetryPolicy(maximum_attempts=3),
            result_type=str,
        )
        workflow.logger.info(f"INFO: Dataset available at {path}")
        
        # Step 2: Train model
        train_config = inputs.train_config
        
        model_path: str = await workflow.execute_activity(
            train_FNO,
            train_config,
            start_to_close_timeout=timedelta(minutes=20),
            retry_policy=RetryPolicy(maximum_attempts=3),
            result_type=str,
        )
        workflow.logger.info(f"INFO: Model trained and saved at {model_path}")
        
        # Step 3: Create evaluate config with the trained model path
        evaluate_config = EvaluateConfig(
            model_path=model_path,
            train_config=train_config
        )
        
        # Step 4: Evaluate model
        result: Tuple[float, str] = await workflow.execute_activity(
            evaluate_FNO,
            evaluate_config, 
            start_to_close_timeout=timedelta(minutes=20),
            retry_policy=RetryPolicy(maximum_attempts=3),
            result_type=Tuple[float, str],
        )
        test_mse, plot_dir = result
        
        workflow.logger.info(f"INFO: Evaluation complete. Test MSE: {test_mse:.6e}")
        workflow.logger.info(f"INFO: Evaluation plots saved at {plot_dir}")
        
        # Return summary of the pipeline
        return {
            "dataset_path": path,
            "model_path": model_path,
            "test_mse": test_mse,
            "plot_dir": plot_dir,
            "status": "success"
        }
