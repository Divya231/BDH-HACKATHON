"""
Baby Dragon Hatchling (BDH) Model Integration
Track B: BDH-driven continuous narrative reasoning

This module provides:
1. BDH model wrapper for HuggingFace transformers
2. Stateful processing for long narratives
3. Sparse attention mechanisms
4. Integration with consistency classification
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from loguru import logger
from transformers import AutoModel, AutoConfig, AutoTokenizer
import numpy as np


class BDHConfig:
    """Configuration for BDH model"""
    
    def __init__(self, config_dict: Dict):
        self.vocab_size = config_dict.get('vocab_size', 50257)
        self.hidden_dim = config_dict.get('hidden_dim', 768)
        self.num_layers = config_dict.get('num_layers', 12)
        self.num_heads = config_dict.get('num_heads', 12)
        self.ff_dim = config_dict.get('ff_dim', 3072)
        self.max_seq_length = config_dict.get('max_seq_length', 2048)
        self.dropout = config_dict.get('dropout', 0.1)
        
        # BDH-specific
        self.state_dim = config_dict.get('state_dim', 768)
        self.memory_size = config_dict.get('memory_size', 512)
        self.sparsity_threshold = config_dict.get('sparsity_threshold', 0.1)
        self.update_mechanism = config_dict.get('update_mechanism', 'selective')
        
        # Validation
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}). "
                f"Current head_dim would be {self.hidden_dim}/{self.num_heads} = {self.hidden_dim/self.num_heads:.2f}"
            )
        
        logger.info(f"BDHConfig initialized: {self.hidden_dim}D, {self.num_layers}L")


class StatefulAttention(nn.Module):
    """
    Stateful attention mechanism inspired by BDH
    Maintains persistent state across sequences
    """
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Memory state
        self.memory_k = nn.Parameter(torch.randn(config.memory_size, config.hidden_dim))
        self.memory_v = nn.Parameter(torch.randn(config.memory_size, config.hidden_dim))
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with stateful attention
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]
            past_state: (past_k, past_v) for incremental processing
            
        Returns:
            output: [batch, seq_len, hidden_dim]
            new_state: (new_k, new_v) for next iteration
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate with memory if using stateful processing
        if past_state is not None:
            past_k, past_v = past_state
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
            # Extend attention mask to match concatenated sequence length
            if attention_mask is not None:
                past_len = past_k.size(2)
                # Create mask for past states (all ones - attend to all past)
                past_mask = torch.ones(batch_size, past_len, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([past_mask, attention_mask], dim=1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply sparsity threshold (BDH characteristic)
        if self.config.sparsity_threshold > 0:
            threshold = torch.quantile(scores, self.config.sparsity_threshold, dim=-1, keepdim=True)
            scores = torch.where(scores < threshold, torch.full_like(scores, float('-inf')), scores)
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(output)
        
        # Store state for next iteration (selective update)
        new_state = (k[:, :, -self.config.memory_size:, :], v[:, :, -self.config.memory_size:, :])
        
        return output, new_state


class BDHLayer(nn.Module):
    """Single BDH transformer layer with stateful attention"""
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.attention = StatefulAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_state: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, new_state = self.attention(hidden_states, attention_mask, past_state)
        hidden_states = residual + attn_output
        
        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output
        
        return hidden_states, new_state


class BDHModel(nn.Module):
    """
    Baby Dragon Hatchling Model
    
    Key features:
    - Persistent state across sequences
    - Sparse attention patterns
    - Incremental belief formation
    - Suitable for long-context narrative reasoning
    """
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        
        # BDH layers
        self.layers = nn.ModuleList([
            BDHLayer(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"BDHModel initialized with {config.num_layers} layers")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_states: Optional[List[Tuple]] = None,
        return_states: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BDH model
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            past_states: List of (k, v) tuples for each layer
            return_states: Whether to return states for next iteration
            
        Returns:
            Dictionary with:
                - hidden_states: Final hidden states
                - past_states: States for incremental processing
                - pooled_output: Pooled representation
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Initialize past states if not provided
        if past_states is None:
            past_states = [None] * self.config.num_layers
        
        # Process through layers
        new_states = []
        for i, layer in enumerate(self.layers):
            hidden_states, new_state = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_state=past_states[i]
            )
            new_states.append(new_state)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Pooled output (mean pooling over sequence)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_hidden / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        outputs = {
            'hidden_states': hidden_states,
            'pooled_output': pooled_output,
        }
        
        if return_states:
            outputs['past_states'] = new_states
        
        return outputs
    
    def process_long_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 512
    ) -> torch.Tensor:
        """
        Process long sequences incrementally using BDH's stateful nature
        Critical for 100k+ word narratives
        
        Args:
            input_ids: [batch, long_seq_len]
            attention_mask: [batch, long_seq_len]
            chunk_size: Size of chunks to process
            
        Returns:
            Final pooled representation
        """
        batch_size, total_len = input_ids.shape
        num_chunks = (total_len + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing long sequence: {total_len} tokens in {num_chunks} chunks")
        
        past_states = None
        all_pooled = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_len)
            
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None
            
            outputs = self.forward(
                chunk_ids,
                attention_mask=chunk_mask,
                past_states=past_states,
                return_states=True
            )
            
            past_states = outputs['past_states']
            all_pooled.append(outputs['pooled_output'])
        
        # Aggregate representations from all chunks
        final_pooled = torch.stack(all_pooled, dim=1).mean(dim=1)
        
        return final_pooled


class BDHForConsistencyClassification(nn.Module):
    """
    BDH model wrapper for binary consistency classification
    Track B main model
    """
    
    def __init__(self, config: BDHConfig, num_classes: int = 2):
        super().__init__()
        self.bdh = BDHModel(config)
        self.config = config
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        logger.info("BDHForConsistencyClassification initialized")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_long_context: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification
        
        Args:
            input_ids: Tokenized input [batch, seq_len]
            attention_mask: Attention mask
            labels: Ground truth labels [batch]
            use_long_context: Use incremental processing for long sequences
            
        Returns:
            Dictionary with logits, loss (if labels provided), predictions
        """
        # Process through BDH
        if use_long_context and input_ids.shape[1] > self.config.max_seq_length:
            pooled = self.bdh.process_long_sequence(input_ids, attention_mask)
        else:
            outputs = self.bdh(input_ids, attention_mask)
            pooled = outputs['pooled_output']
        
        # Classification
        logits = self.classifier(pooled)
        predictions = torch.argmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'predictions': predictions,
            'pooled_representation': pooled
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
        
        return result


def load_bdh_model(config_dict: Dict, pretrained_path: Optional[str] = None) -> BDHForConsistencyClassification:
    """
    Load BDH model for consistency classification
    
    Args:
        config_dict: Configuration dictionary
        pretrained_path: Path to pretrained checkpoint (optional)
        
    Returns:
        BDH model ready for training/inference
    """
    bdh_config = BDHConfig(config_dict)
    model = BDHForConsistencyClassification(bdh_config)
    
    if pretrained_path and pretrained_path.lower() != 'none':
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Pretrained weights loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    return model


def main():
    """Test BDH model"""
    # Test configuration
    config_dict = {
        'vocab_size': 50257,
        'hidden_dim': 768,
        'num_layers': 6,
        'num_heads': 12,
        'ff_dim': 3072,
        'max_seq_length': 512,
        'dropout': 0.1,
        'state_dim': 768,
        'memory_size': 256,
        'sparsity_threshold': 0.1,
        'update_mechanism': 'selective'
    }
    
    # Create model
    model = load_bdh_model(config_dict)
    
    # Test forward pass
    batch_size = 2
    seq_len = 256
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size,))
    
    outputs = model(input_ids, attention_mask, labels)
    
    logger.info(f"Test outputs:")
    logger.info(f"  Logits shape: {outputs['logits'].shape}")
    logger.info(f"  Predictions: {outputs['predictions']}")
    logger.info(f"  Loss: {outputs['loss'].item():.4f}")


if __name__ == "__main__":
    main()
