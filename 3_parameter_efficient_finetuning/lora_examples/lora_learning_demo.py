#!/usr/bin/env python3
"""
Demo: How LoRA Learns to Decompose Updates
This shows how LoRA discovers important patterns through training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_lora_learning():
    """Show how LoRA learns to approximate a target update matrix."""
    
    # Simulate a "target" update that we want to learn
    # In real training, this would come from gradients
    torch.manual_seed(42)
    d = 100  # Matrix dimension
    
    # Create a target update with some structure (not random)
    # This simulates what the model "wants" to learn
    target_update = torch.zeros(d, d)
    
    # Add some structured patterns (like what might emerge in real training)
    # Pattern 1: Diagonal emphasis
    target_update += 0.5 * torch.eye(d)
    
    # Pattern 2: Block structure (like attention heads)
    target_update[0:25, 0:25] += 0.3
    target_update[25:50, 25:50] += 0.2
    
    # Pattern 3: Some random noise
    target_update += 0.1 * torch.randn(d, d)
    
    print("🎯 Target Update Matrix (what we want to learn):")
    print(f"   Shape: {target_update.shape}")
    print(f"   Norm: {target_update.norm():.4f}")
    print()
    
    # Now let's see how different ranks can approximate this
    ranks_to_try = [1, 2, 4, 8, 16, 32]
    
    for r in ranks_to_try:
        # Initialize LoRA matrices
        A = torch.randn(d, r) * 0.1
        B = torch.randn(r, d) * 0.1
        
        # Simple optimization to find best A, B
        A.requires_grad_(True)
        B.requires_grad_(True)
        
        optimizer = torch.optim.Adam([A, B], lr=0.01)
        
        # Train to approximate the target
        for step in range(1000):
            optimizer.zero_grad()
            
            # Current approximation
            current_approx = A @ B
            
            # Loss: how far are we from target?
            loss = torch.nn.functional.mse_loss(current_approx, target_update)
            
            loss.backward()
            optimizer.step()
            
            if step % 200 == 0:
                print(f"   Rank {r:2d}, Step {step:4d}: Loss = {loss.item():.6f}")
        
        # Final approximation quality
        final_approx = A @ B
        final_error = torch.nn.functional.mse_loss(final_approx, target_update)
        
        # Calculate compression ratio
        original_params = d * d
        lora_params = d * r + r * d
        compression_ratio = original_params / lora_params
        
        print(f"✅ Rank {r:2d} Results:")
        print(f"   Final Error: {final_error.item():.6f}")
        print(f"   Parameters: {lora_params:,} vs {original_params:,} (compression: {compression_ratio:.1f}x)")
        print(f"   Approximation Quality: {(1 - final_error.item()) * 100:.1f}%")
        print()

def show_what_lora_discovers():
    """Show what patterns LoRA matrices learn to capture."""
    
    print("🔍 What LoRA Matrices Learn to Represent:")
    print()
    
    # Simulate learned A and B matrices
    torch.manual_seed(123)
    d, r = 50, 4
    
    # These would be learned through training
    A = torch.randn(d, r)
    B = torch.randn(r, d)
    
    print(f"Matrix A ({d}x{r}): Maps full space → {r}-dimensional space")
    print("Each column of A represents a 'direction' that matters for the task")
    print()
    
    print(f"Matrix B ({r}x{d}): Maps {r}-dimensional space → full space")  
    print("Each row of B represents how to 'expand' each direction back")
    print()
    
    # Show what each rank dimension captures
    reconstruction = A @ B
    
    print("🎨 What Each Rank Dimension Captures:")
    for i in range(r):
        # Contribution of rank dimension i
        contribution = torch.outer(A[:, i], B[i, :])
        contribution_strength = contribution.norm().item()
        
        print(f"   Rank {i+1}: Strength = {contribution_strength:.3f}")
        print(f"           Captures patterns like: {contribution[0, :5].tolist()}")
    
    print()
    print("💡 Key Insight: LoRA learns to find the most 'important' directions")
    print("   automatically through gradient descent!")

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 How LoRA Learns What Elements to Keep")
    print("=" * 60)
    print()
    
    demonstrate_lora_learning()
    print("\n" + "=" * 60)
    show_what_lora_discovers()
