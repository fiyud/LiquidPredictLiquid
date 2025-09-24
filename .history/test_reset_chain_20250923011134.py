"""
Test script to verify the reset_liquid_state method chain works properly.
"""
import torch
import sys
import os

# Add the current directory to path to import LiquidMamba
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LiquidMamba import MultiScaleLiquidMamba

def test_reset_chain():
    """Test that the reset_liquid_state method chain works without errors."""
    print("Testing reset_liquid_state method chain...")
    
    # Create a MultiScaleLiquidMamba instance
    model = MultiScaleLiquidMamba(
        d_model=64,
        num_scales=3,
        state_size=16,
        expand_factor=2
    )
    
    # Test that we can call reset_liquid_state without errors
    try:
        model.reset_liquid_state()
        print("✅ MultiScaleLiquidMamba.reset_liquid_state() - SUCCESS")
    except Exception as e:
        print(f"❌ MultiScaleLiquidMamba.reset_liquid_state() - FAILED: {e}")
        return False
    
    # Test that each branch has the reset method
    try:
        for i, branch in enumerate(model.mamba_branches):
            branch.reset_liquid_state()
            print(f"✅ Branch {i}.reset_liquid_state() - SUCCESS")
    except Exception as e:
        print(f"❌ Branch reset_liquid_state() - FAILED: {e}")
        return False
    
    # Test forward pass still works after reset
    try:
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64)
        output = model(x)
        print(f"✅ Forward pass after reset - SUCCESS, output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass after reset - FAILED: {e}")
        return False
    
    print("\n🎉 All tests passed! The reset chain functionality is working correctly.")
    return True

if __name__ == "__main__":
    test_reset_chain()