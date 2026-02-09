# FNO LDC Temporal Workflow

A Temporal-based workflow system for training and deploying Fourier Neural Operators (FNO) on Lid-Driven Cavity (LDC) flow simulations.

## Overview

This project orchestrates machine learning workflows for physics-informed neural networks using Temporal's durable execution framework. It applies FNO models to predict fluid dynamics in lid-driven cavity problems.

## Architecture

```
├── activities/      # Temporal activities (atomic tasks)
├── models/         # FNO model definitions
├── utils/          # Helper functions and utilities
├── workflow/       # Temporal workflow definitions
├── worker.py       # Temporal worker process
├── workflow_run.py # Workflow execution entry point
└── shared.py       # Shared configurations and constants
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

1. **Start the Temporal worker:**
   ```bash
   python worker.py
   ```

2. **Run a workflow:**
   ```bash
   python workflow_run.py
   ```

## Features

- ✅ Durable execution with automatic retries
- ✅ Fault-tolerant ML pipeline orchestration
- ✅ FNO-based fluid dynamics prediction
- ✅ LDC flow simulation handling
- ✅ Distributed task execution

## What is FNO?

Fourier Neural Operators learn mappings between function spaces using spectral convolutions, making them efficient for solving partial differential equations like those governing fluid flow.

## What is LDC?

The Lid-Driven Cavity is a benchmark CFD problem where fluid motion is driven by a moving lid, commonly used to validate numerical methods and ML models.

## Use Cases

- Training surrogate models for CFD simulations
- Real-time flow prediction from initial conditions
- Parameter optimization in fluid dynamics
- Reduced-order modeling for engineering applications

## License

See repository license file for details.
