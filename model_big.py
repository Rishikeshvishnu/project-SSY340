#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from typing import Optional,Union
from dataclasses import dataclass
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt
from typing import Dict

from data_preprocessing import *
from model_layers_new import *

import matplotlib.pyplot as plt
import time
from collections import defaultdict
# In[2]:


from math import sqrt

class MusicTransformerDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        # Save config parameters
        self.d_model = config['d_model']
        self.max_abs_position = config['max_abs_position']
        self.vocab_size = config['vocab_size']
        
        # Input embedding layer
        self.input_embedding = nn.Embedding(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['d_model']
        )
        
        # Create positional encoding and register as buffer
        pos_encoding = sinusoidal_positional_encoding(
            config['max_abs_position'], 
            config['d_model']
        )
        self.register_buffer('positional_encoding', pos_encoding)
        
        # Input dropout
        self.input_dropout = nn.Dropout(config['dropout'])
        
        # Decoder layers
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim_model=config['d_model'],
                num_heads=config['num_heads'],
                dim_ff=config['d_ff'],
                max_rel_dist=config['max_rel_dist'],
                dropout=config['dropout'],
                eps=config['layernorm_eps']
            ) for _ in range(config['num_layers'])
        ])
        
        # Final layer norm (to match TransformerDecoder behavior)
        self.final_norm = nn.LayerNorm(config['d_model'], eps=config['layernorm_eps'])
        
        # Output projection to vocabulary
        self.final = nn.Linear(config['d_model'], config['vocab_size'])
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Music Transformer Decoder.
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            mask: Optional mask tensor for masked attention
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Input embedding and scaling
        x = self.input_embedding(x) * sqrt(self.d_model)
        
        # Add positional encoding if configured
        if self.max_abs_position > 0:
            x = x + self.positional_encoding[:, :x.shape[-2], :]
        
        # Apply input dropout
        x = self.input_dropout(x)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder:
            x = decoder_layer(x, tgt_mask=mask)
        
        # Apply final normalization (to match TransformerDecoder)
        x = self.final_norm(x)
        
        # Final projection to vocabulary
        return self.final(x)


# In[3]:


def transformer_learning_rate(
    dim_model: int,
    current_step: Union[int, torch.Tensor],
    warmup_steps: int = 4000
) -> Union[float, torch.Tensor]:
    """
    Implements the transformer learning rate schedule with warmup and decay.
    
    This scheduler increases the learning rate linearly for the first warmup_steps,
    and then decreases it proportionally to the inverse square root of the step number.
    Formula: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    Args:
        dim_model: Model's hidden dimension size
        current_step: Current training step number
        warmup_steps: Number of warmup steps (default: 4000)
    
    Returns:
        Calculated learning rate for the current step
    """
    # Ensure warmup_steps is positive
    if warmup_steps <= 0:
        current_step = current_step + 4000
        warmup_steps = 4000
    
    # Add small epsilon to avoid division by zero
    step = current_step + 1e-8
    
    # Calculate model dimension scaling factor
    dim_scale = dim_model ** -0.5
    
    # Calculate step-based scaling factor
    if isinstance(step, torch.Tensor):
        step_scale = torch.min(
            step ** -0.5,
            step * (warmup_steps ** -1.5)
        )
    else:
        step_scale = min(
            step ** -0.5,
            step * (warmup_steps ** -1.5)
        )
    
    return dim_scale * step_scale


# In[4]:


def masked_loss_function(logits, ground_truth, loss_metric=F.cross_entropy):
    """
    Compute average loss excluding padded elements (zeros) in the target sequence
    Args:
        logits: model predictions to evaluate
        ground_truth: expected output values
        loss_metric: base loss function to apply before masking
    Returns:
        average loss value over non-padded positions
    """
    # Generate padding mask (True for valid positions, False for padding)
    valid_positions = torch.not_equal(ground_truth, 0)
    
    # Calculate per-element losses
    element_losses = loss_metric(logits, ground_truth, reduction='none')
    
    # Convert boolean mask to match loss dtype for multiplication
    valid_positions = valid_positions.type(element_losses.dtype)
    
    # Apply mask and calculate mean over valid positions
    masked_losses = element_losses * valid_positions
    num_valid = torch.sum(valid_positions)
    
    return torch.sum(masked_losses) / num_valid


# In[5]:


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, pad_token: int = 0) -> Dict[str, float]:
    """
    Calculate core evaluation metrics for music generation with corrected F1 calculation.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        targets: Ground truth labels of shape (batch_size, seq_len)
        pad_token: Token used for padding sequences
        
    Returns:
        Dictionary containing accuracy, F1 score, and perplexity metrics
    """
    metrics = {}
    
    # Get predicted token indices
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for non-padding positions
    valid_positions = torch.ne(targets, pad_token)
    num_valid = torch.sum(valid_positions)
    
    # Calculate correct predictions
    correct_predictions = (predictions == targets) & valid_positions
    total_correct = torch.sum(correct_predictions.float())
    
    # 1. Accuracy
    metrics['accuracy'] = (total_correct / num_valid).item()
    
    # 2. F1 Score with proper precision and recall calculation
    n_classes = logits.size(-1)
    precisions, recalls = [], []
    
    for class_idx in range(n_classes):
        if class_idx == pad_token:
            continue  # Skip padding token class
        
        true_positives = torch.sum((predictions == class_idx) & (targets == class_idx) & valid_positions).float()
        predicted_positives = torch.sum((predictions == class_idx) & valid_positions).float()
        actual_positives = torch.sum((targets == class_idx) & valid_positions).float()
        
        if predicted_positives > 0:
            precisions.append(true_positives / predicted_positives)
        
        if actual_positives > 0:
            recalls.append(true_positives / actual_positives)
    
    # Macro F1 score
    if precisions and recalls:
        avg_precision = torch.mean(torch.stack(precisions))
        avg_recall = torch.mean(torch.stack(recalls))
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)
        metrics['f1_score'] = f1.item()
    else:
        metrics['f1_score'] = 0.0
    
    # 3. Perplexity
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid_positions
    avg_log_prob = torch.sum(token_log_probs) / num_valid
    metrics['perplexity'] = torch.exp(-avg_log_prob).item()
    
    return metrics


# In[6]:


# uncomment to train the model

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"The device being used is {device}")

# # Load and move data to device
# data = torch.load('sqip.pt').to(device)
# print(f"Data shape: {data.shape}")

# # Data splitting and setup
# total_samples = data.shape[0]
# train_size = int(0.85 * total_samples)
# val_size = total_samples - train_size

# train_data = data[:train_size]
# val_data = data[train_size:]

# print(f"Training samples: {len(train_data)}")
# print(f"Validation samples: {len(val_data)}")

# # Create datasets and dataloaders
# train_ds = TensorDataset(train_data[:, :-1], train_data[:, 1:])
# val_ds = TensorDataset(val_data[:, :-1], val_data[:, 1:])

# batch_size = 32
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# # Initialize model with config
# config = {
#     "d_model": 256,
#     "num_layers": 6,
#     "num_heads": 8,
#     "d_ff": 1024,
#     "max_rel_dist": 1024,
#     "max_abs_position": 0,
#     "vocab_size": vocab_size,
#     "dropout": 0.1,
#     "layernorm_eps": 1e-6
# }

# model = MusicTransformerDecoder(config).to(device)

# # Training setup with transformer learning rate schedule
# optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98))
# warmup_steps = 2000
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer,
#     lambda step: transformer_learning_rate(
#         config['d_model'],
#         step,
#         warmup_steps
#     )
# )

# num_epochs = 50

# # Initialize metrics storage
# metrics_history = {
#     'train': defaultdict(list),
#     'val': defaultdict(list)
# }
# best_val_loss = float('inf')

# print("Beginning training...")
# print(time.strftime("%Y-%m-%d %H:%M"))

# model = torch.compile(model)
# torch.set_float32_matmul_precision("high")

# for epoch in range(num_epochs):
#     start_time = time.time()
    
#     # Training phase
#     model.train()
#     epoch_train_metrics = defaultdict(list)
    
#     for batch_inp, batch_tar in train_dl:
#         batch_inp, batch_tar = batch_inp.to(device), batch_tar.to(device)
        
#         # Create attention mask
#         mask = create_mask(batch_inp, n=4)
#         logits = model(batch_inp, mask=mask)

#         # Forward pass
#         optimizer.zero_grad()
        
#         # Compute loss and metrics
#         loss = masked_loss_function(logits.transpose(-1, -2), batch_tar)
#         batch_metrics = calculate_metrics(logits, batch_tar)
#         batch_metrics['loss'] = loss.item()
        
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#         # Store batch metrics
#         for metric, value in batch_metrics.items():
#             epoch_train_metrics[metric].append(value)
    
#     # Validation phase
#     model.eval()
#     epoch_val_metrics = defaultdict(list)
    
#     with torch.no_grad():
#         for batch_inp, batch_tar in val_dl:
#             batch_inp, batch_tar = batch_inp.to(device), batch_tar.to(device)
            
#             mask = create_mask(batch_inp, n=max(batch_inp.dim() + 2, 2))
#             logits = model(batch_inp, mask=mask)
            
#             loss = masked_loss_function(logits.transpose(-1, -2), batch_tar)
#             batch_metrics = calculate_metrics(logits, batch_tar)
#             batch_metrics['loss'] = loss.item()
            
#             for metric, value in batch_metrics.items():
#                 epoch_val_metrics[metric].append(value)
    
#     # Calculate average metrics for epoch
#     train_metrics = {metric: sum(values)/len(values) for metric, values in epoch_train_metrics.items()}
#     val_metrics = {metric: sum(values)/len(values) for metric, values in epoch_val_metrics.items()}
    
#     # Store metrics history
#     for metric in train_metrics:
#         metrics_history['train'][metric].append(train_metrics[metric])
#         metrics_history['val'][metric].append(val_metrics[metric])
    
#     # Save best model based on validation loss
#     if val_metrics['loss'] < best_val_loss:
#         best_val_loss = val_metrics['loss']
#         best_model_state = model.state_dict()
    
#     epoch_time = time.time() - start_time
#     current_lr = scheduler.get_last_lr()[0]
    
#     # Print progress with all metrics
#     print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - LR: {current_lr:.2e}")
#     for metric in train_metrics:
#         print(f"{metric}: train={train_metrics[metric]:.4f}, val={val_metrics[metric]:.4f}")
#     print("-" * 80)

# # Save the final model and metrics
# save_file = {
#     "state_dict": best_model_state,
#     "config": config,
#     "metrics_history": metrics_history,
#     "best_val_loss": best_val_loss
# }

# torch.save(save_file, "music_transformer_model.pt")
# print("Model saved successfully!")
# print(time.strftime("%Y-%m-%d %H:%M"))

# # Plot training history for all metrics
# metric_names = list(metrics_history['train'].keys())
# num_metrics = len(metric_names)
# fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 4))

# for i, metric in enumerate(metric_names):
#     ax = axes[i]
#     ax.plot(metrics_history['train'][metric], label=f'Train {metric}')
#     ax.plot(metrics_history['val'][metric], label=f'Val {metric}')
#     ax.set_title(f'Training and Validation {metric.capitalize()}')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel(metric.capitalize())
#     ax.legend()

# plt.tight_layout()
# plt.savefig('training_history.png')
# plt.show()


# # In[ ]:




