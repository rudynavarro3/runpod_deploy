# Hugging Face Model Fine-tuning with Docker & RunPod

A complete containerized solution for fine-tuning Hugging Face models with GPU acceleration on RunPod.io.

## Features

- 🚀 **Containerized Training**: Complete Docker setup for reproducible model training
- 🔥 **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- 🤗 **Hugging Face Integration**: Seamless model loading and pushing to Hub
- ☁️ **RunPod Ready**: Pre-configured for deployment on RunPod.io
- 📊 **Flexible Training**: Support for custom datasets and training parameters
- 🛠️ **Production Ready**: Includes logging, error handling, and monitoring

## Quick Start

### 1. Project Setup

```bash
# Clone or create your project directory
mkdir hf-model-trainer && cd hf-model-trainer

# Run the setup script
python setup.py --all

# Copy environment template and configure
cp .env.template .env
# Edit .env with your credentials
```

### 2. Configure Your Training

Edit `config/train_config.json`:

```json
{
  "model_name": "microsoft/DialoGPT-small",
  "output_model_name": "your-username/my-finetuned-model",
  "training_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "learning_rate": 5e-5
  }
}
```

### 3. Add Training Data

Place your training data in `data/` directory or use the provided sample data:

```json
[
  "What is machine learning?",
  "Machine learning is a subset of AI...",
  "How do neural networks work?",
  "Neural networks process information..."
]
```

### 4. Local Development

```bash
# Build the Docker image
make build

# Run training locally
make run

# Or run directly with custom parameters
docker run --gpus all -v $(pwd):/workspace hf-model-trainer \
  python train.py \
  --model_name microsoft/DialoGPT-small \
  --output_model_name your-username/my-model \
  --epochs 3
```

## RunPod Deployment

### Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **Docker Registry**: Push your image to Docker Hub or another registry
3. **Hugging Face Token**: Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Step-by-Step Deployment

#### 1. Build and Push Image

```bash
# Build image
docker build -t your-username/hf-model-trainer .

# Push to registry
docker push your-username/hf-model-trainer
```

#### 2. Deploy to RunPod

```bash
# Generate RunPod configuration
python deploy_runpod.py \
  --api_key YOUR_RUNPOD_API_KEY \
  --image_name your-username/hf-model-trainer \
  --model_name microsoft/DialoGPT-small \
  --output_model_name your-username/my-finetuned-model \
  --hf_token YOUR_HF_TOKEN \
  --build_image
```

#### 3. Manual RunPod Setup

1. **Create a Pod**:
   - Go to RunPod console
   - Select GPU type (RTX 3090, A100, etc.)
   - Use your Docker image
   - Set container disk size: 50GB+
   - Add persistent volume: 100GB+

2. **Environment Variables**:
   ```
   HF_TOKEN=your_huggingface_token
   TRANSFORMERS_CACHE=/workspace/.cache
   HF_HOME=/workspace/.cache
   ```

3. **Docker Command**:
   ```bash
   python train.py --model_name microsoft/DialoGPT-small --output_model_name your-username/my-model --epochs 3 --batch_size 4
   ```

## Project Structure

```
hf-model-trainer/
├── train.py              # Main training script
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Local development
├── requirements.txt      # Python dependencies
├── deploy_runpod.py      # RunPod deployment
├── setup.py             # Project setup script
├── Makefile             # Common commands
├── .env.template        # Environment template
├── config/
│   └── train_config.json # Training configuration
├── data/
│   └── sample_data.json  # Training data
├── results/             # Training outputs
├── models/              # Model cache
└── scripts/             # Additional utilities
```

## Training Script Usage

```bash
python train.py \
  --model_name microsoft/DialoGPT-small \
  --output_model_name your-username/my-finetuned-model \
  --hf_token YOUR_TOKEN \
  --dataset_name squad \
  --custom_data_file data/my_data.json \
  --epochs 3 \
  --batch_size 2
```

### Parameters

- `--model_name`: Base model from Hugging Face Hub
- `--output_model_name`: Name for your fine-tuned model
- `--hf_token`: Hugging Face authentication token
- `--dataset_name`: Dataset from Hugging Face datasets
- `--custom_data_file`: Path to custom training data (JSON)
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size

## Custom Data Format

### Text Data (JSON)
```json
[
  "First training example text",
  "Second training example text",
  "Third training example text"
]
```

### Conversation Data
```json
[
  {
    "text": "Human: Hello! Assistant: Hi there, how can I help?"
  },
  {
    "text": "Human: What's the weather? Assistant: I don't have real-time weather data."
  }
]
```

## Advanced Configuration

### Training Arguments

Modify `config/train_config.json` for advanced settings:

```json
{
  "training_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 100,
    "save_steps": 500,
    "eval_steps": 500,
    "fp16": true,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "linear"
  }
}
```

### Memory Optimization

For large models or limited GPU memory:

```python
# Enable gradient checkpointing
gradient_checkpointing=True

# Use fp16 training
fp16=True

# Reduce batch size
per_device_train_batch_size=1

# Use gradient accumulation
gradient_accumulation_steps=4
```

## Monitoring and Logging

### TensorBoard

```bash
# Start TensorBoard (if running locally)
tensorboard --logdir results/runs

# View at http://localhost:6006
```

### Weights & Biases

Add to requirements.txt and configure:

```python
import wandb

# In training script
wandb.init(project="hf-finetuning")
```

## Troubleshooting

### Common Issues

**Out of Memory Errors**:
- Reduce batch size: `--batch_size 1`
- Enable gradient checkpointing
- Use fp16 training
- Reduce sequence length

**Authentication Errors**:
- Verify HF_TOKEN is set correctly
- Check Hugging Face Hub permissions
- Ensure model name follows format: `username/model-name`

**Docker Issues**:
- Ensure NVIDIA Docker runtime is installed
- Check GPU availability: `nvidia-smi`
- Verify CUDA compatibility

**RunPod Issues**:
- Check container logs for errors
- Ensure sufficient disk space
- Verify environment variables are set

### Debugging

```bash
# Run container interactively
docker run -it --gpus all hf-model-trainer bash

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test Hugging Face authentication
python -c "from huggingface_hub import login; login()"
```

## Performance Tips

### GPU Optimization

1. **Batch Size**: Start with small batch size and increase gradually
2. **Mixed Precision**: Use fp16 for faster training
3. **Gradient Accumulation**: Simulate larger batches
4. **Model Parallelism**: For very large models

### Cost Optimization on RunPod

1. **Spot Instances**: Use cheaper spot pricing when available
2. **Auto-shutdown**: Set up automatic shutdown after training
3. **Persistent Volumes**: Save models to persistent storage
4. **Monitor Usage**: Track GPU utilization and costs

## Examples

### Fine-tune DialoGPT for Customer Service

```bash
python train.py \
  --model_name microsoft/DialoGPT-medium \
  --output_model_name company/customer-service-bot \
  --custom_data_file data/customer_conversations.json \
  --epochs 5 \
  --batch_size 4
```

### Fine-tune GPT-2 for Creative Writing

```bash
python train.py \
  --model_name gpt2 \
  --output_model_name writer/creative-gpt2 \
  --dataset_name writing_prompts \
  --epochs 3 \
  --batch_size 2
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- 📖 [Hugging Face Documentation](https://huggingface.co/docs)
- 🏃 [RunPod Documentation](https://docs.runpod.io)
- 🐛 [Issue Tracker](https://github.com/your-repo/issues)
- 💬 [Discussions](https://github.com/your-repo/discussions)

## Acknowledgments

- Hugging Face team for the transformers library
- RunPod for GPU infrastructure
- NVIDIA for CUDA support