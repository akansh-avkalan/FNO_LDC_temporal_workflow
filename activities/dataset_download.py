from temporalio import activity 
import os 
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from shared import DataSetConfig


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

@activity.defn
async def dataset_download(dataset_config: DataSetConfig): 
    logger = activity.logger
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not found in environment variables")

    # logger.info(
    #     f"Starting download | repo={dataset_config.repo_id}, folder={dataset_config.folder}, output={dataset_config.output_dir}"
    # )

    try:
        snapshot_download(
            repo_id=dataset_config.repo_id,
            repo_type="dataset",
            local_dir=dataset_config.output_dir,
            allow_patterns=[f"{dataset_config.folder}/*"],
            token=HF_TOKEN
        )
    except Exception as e:
        logger.exception("Dataset download failed")
        raise 

    # logger.info("INFO:  Dataset download completed successfully")
    return f"{dataset_config.output_dir}/{dataset_config.folder}"
