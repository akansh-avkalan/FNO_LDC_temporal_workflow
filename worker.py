import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
import logging
from shared import TASK_QUEUE
from workflow.ml_pipeline_workflow import MLPipelineWorkflow

from activities.dataset_download import dataset_download
from activities.train_FNO import train_FNO
from activities.evaluate_FNO import evaluate_FNO


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MLPipelineWorkflow],
        activities=[
            dataset_download,
            train_FNO,
            evaluate_FNO,
        ],
    )

    print("Worker started, listening on task queue:", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
