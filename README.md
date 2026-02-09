# FNO LDC Temporal Workflow

A Temporal-based workflow system for training and deploying Fourier Neural Operators (FNO) on Lid-Driven Cavity (LDC) flow simulations.

## Overview

This project orchestrates machine learning workflows for physics-informed neural networks using Temporal's durable execution framework. It applies FNO models to predict fluid dynamics in lid-driven cavity problems.

## Architecture

```
Folder Structure:
model_workflow/
├── worker.py            # Temporal Worker process
├── workflow_run.py      # Workflow execution entry point
├── shared.py            # Shared configuration and constant    
├── LDC_128              # Dataset 
|    └── LDC_NS_2D\128x128
│
├── workflows/                   # Workflow definition
│	 ├── __init__.py
│    └── ml_pipeline_workflow.py # Sequence and failure handling code
│
├── activities/                  # Temporal activities
│    ├── __init__.py
│    ├── dataset_download.py      # Download the dataset 
│    ├── train_fno.py             # training module 
│	 └── evaluate_FNO.py          # Inference code
│
├── utils/
│    ├── dataset.py               # Dataset, Split the data, Dataloader
│    ├── metrics.py               # L2_norm and LInf_Norm
│    ├── trainer.py               # helper function : Train model 
│    └── visulization.py          # Comparision chart creation 
│
└── requirements.txt 
```

## Key Components

- **Workflows**: Orchestrate the complete ML pipeline (data preparation, training, inference)
- **Activities**: Individual tasks (data loading, model training, evaluation) with automatic retry logic
- **Models**: FNO neural network implementations for fluid dynamics
- **Worker**: Executes workflow and activity code

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start


1. **Start the Temporal:**
   ```bash
   temporal server start-dev
   ```

2. **Start the Temporal worker:**
   ```bash
   python worker.py
   ```

3. **Run a workflow:**
   ```bash
   python workflow_run.py
   ```


## Failure handling: 
**Retry Policies** - Exponential backoff with configurable attempts and intervals  
**Activity Timeouts** - Four types: Start-to-Close, Schedule-to-Close, Heartbeat, and Schedule-to-Start  
**Heartbeats** - Progress tracking and faster failure detection for long ML training jobs  
**Error Handling** - Non-retryable vs retryable errors with the Saga pattern for compensation



## Features

- Durable execution with automatic retries
- Fault-tolerant ML pipeline orchestration
- FNO-based fluid dynamics prediction
- LDC flow simulation handling
- Distributed task execution

## What is FNO?

Fourier Neural Operators learn mappings between function spaces using spectral convolutions, making them efficient for solving partial differential equations like those governing fluid flow.

## What is LDC?

The Lid-Driven Cavity is a benchmark CFD problem where fluid motion is driven by a moving lid, commonly used to validate numerical methods and ML models.


## References: 
```
@misc{li2021fourierneuraloperatorparametric,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2021},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2010.08895}, 
}
```

```
@misc{tali2024flowbenchlargescalebenchmark,
      title={FlowBench: A Large Scale Benchmark for Flow Simulation over Complex Geometries}, 
      author={Ronak Tali and Ali Rabeh and Cheng-Hau Yang and Mehdi Shadkhah and Samundra Karki and Abhisek Upadhyaya and Suriya Dhakshinamoorthy and Marjan Saadati and Soumik Sarkar and Adarsh Krishnamurthy and Chinmay Hegde and Aditya Balu and Baskar Ganapathysubramanian},
      year={2024},
      eprint={2409.18032},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn},
      url={https://arxiv.org/abs/2409.18032}, 
}
```