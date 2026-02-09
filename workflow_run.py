# workflow_run.py

import asyncio
from temporalio.client import Client
from shared import MLPipelineInput, DataSetConfig, TrainConfig, TASK_QUEUE
from workflow.ml_pipeline_workflow import MLPipelineWorkflow


async def main():
    client = await Client.connect("localhost:7233")

    result = await client.execute_workflow(
        MLPipelineWorkflow.run,
        MLPipelineInput(
            dataset_config=DataSetConfig(
                repo_id="BGLab/FlowBench",
                folder="LDC_NS_2D/128x128",
                output_dir="LDC_128",
            ),
            train_config=TrainConfig(
                dataset_path="LDC_128/LDC_NS_2D/128x128",
                batch_size=16,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42,
                num_workers=0,
                fno_width=64,   
                modes1=16,   
                modes2=16,
                in_channels=3,
                out_channels=4,
                epochs=200,
                log_dir="experiments/fno/",
                checkpoint_frequency=100,
                learning_rate=1e-3,
            ),
            # evaluate_config is optional and defaults to None
        ),
        id="fno-ldc-128",             
        task_queue=TASK_QUEUE,       
    )

    print("Workflow completed successfully!")
    print(f"Results: {result}")

if __name__ == "__main__":
    asyncio.run(main())
