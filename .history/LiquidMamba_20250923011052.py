import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat

class LiquidTimeConstantScaler(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        state_size: int = 16,
        tau_init: float = 1.0,
        dt: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.dt = dt
        
        # Time constants cho selective mechanism (learnable parameters)
        self.tau = nn.Parameter(torch.ones(state_size) * tau_init)
        
        # Hidden state cho LTC dynamics
        self.register_buffer('hidden_state', torch.zeros(1, state_size))
        
        # Input và recurrent weights cho LTC
        self.W_input = nn.Linear(d_model, state_size)
        self.W_recurrent = nn.Linear(state_size, state_size, bias=False)
        
        # Selection scaling projection
        self.selection_projection = nn.Linear(state_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            selection_factors: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Extract global context từ input sequence
        global_context = x.mean(dim=1)  # (batch, d_model)
        
        # Input cho LTC
        input_signal = self.W_input(global_context)  # (batch, state_size)
        
        # Update hidden state với LTC dynamics
        if self.hidden_state.size(0) != batch_size:
            # Create new hidden state with correct batch size
            self.hidden_state = self.hidden_state[:1].expand(batch_size, -1).contiguous()
        
        current_hidden = self.hidden_state
        
        # Recurrent connection
        recurrent_input = self.W_recurrent(current_hidden)
        
        # Total input
        total_input = input_signal + recurrent_input
        
        # LTC dynamics: dh/dt = (-h + tanh(input)) / tau
        tau_clamped = torch.clamp(self.tau, min=0.1, max=10.0)
        dh_dt = (-current_hidden + torch.tanh(total_input)) / tau_clamped.unsqueeze(0)
        
        # Euler integration
        new_hidden = current_hidden + self.dt * dh_dt
        
        # Update stored hidden state
        self.hidden_state = new_hidden.detach()
        
        # Generate dynamic selection factors
        selection_factors = torch.sigmoid(self.selection_projection(new_hidden))  # (batch, d_model)
        selection_factors = selection_factors.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, d_model)
        
        return selection_factors
    
    def reset_state(self):
        """Reset hidden state"""
        self.hidden_state.zero_()


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model inspired by Mamba architecture.
    Sử dụng input-dependent parameters để selective propagate information.
    """
    def __init__(
        self,
        d_model: int,
        state_size: int = 16,
        dt_init_min: float = 0.001,
        dt_init_max: float = 0.1,
        use_liquid_scaling: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.use_liquid_scaling = use_liquid_scaling
        
        # SSM parameters - input dependent
        self.delta_proj = nn.Linear(d_model, d_model)  # Δ (timestep)
        self.B_proj = nn.Linear(d_model, state_size)   # B (input matrix)
        self.C_proj = nn.Linear(d_model, state_size)   # C (output matrix)
        
        # A matrix (evolution matrix) - learnable but shared
        # Initialize log-magnitudes to avoid NaNs and ensure A is negative and well-conditioned
        self.A_log = nn.Parameter(torch.zeros(d_model, state_size))
        
        # Delta initialization
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_init_max) - math.log(dt_init_min))
            + math.log(dt_init_min)
        )
        self.dt_bias = nn.Parameter(torch.log(dt))
        
        # LTC-based scaling
        if use_liquid_scaling:
            self.liquid_scaler = LiquidTimeConstantScaler(
                d_model=d_model,
                state_size=state_size
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply LTC scaling if enabled
        if self.use_liquid_scaling:
            selection_factors = self.liquid_scaler(x)
            x_scaled = x * selection_factors
        else:
            x_scaled = x
        
        # Generate input-dependent SSM parameters
        delta = F.softplus(self.delta_proj(x_scaled) + self.dt_bias.unsqueeze(0))  # (B, L, D)
        B = self.B_proj(x_scaled)  # (B, L, N)
        C = self.C_proj(x_scaled)  # (B, L, N)
        
        # A matrix (shared across time)
        A = -torch.exp(self.A_log)  # (D, N)
        
        # Discretize continuous SSM with numerical stability
        A_discrete = torch.exp(A.unsqueeze(0) * delta.unsqueeze(-1))  # (B, L, D, N)
        denom = A.unsqueeze(0).unsqueeze(0)  # (1, 1, D, N)
        factor = torch.where(
            denom.abs() < 1e-6,
            delta.unsqueeze(-1),  # limit (exp(A*dt)-1)/A -> dt when A->0
            (A_discrete - 1) / denom
        )  # (B, L, D, N)
        B_discrete = factor * B.unsqueeze(2)  # (B, L, D, N)
        
        # Selective scan - simplified implementation
        outputs = []
        h = torch.zeros(batch_size, d_model, self.state_size, device=x.device)  # (B, D, N)
        
        for t in range(seq_len):
            # Update state: h = A_discrete * h + B_discrete * x
            h = A_discrete[:, t] * h + B_discrete[:, t] * x_scaled[:, t].unsqueeze(-1)  # (B, D, N)
            
            # Compute output: y = C * h
            y = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1)  # (B, D)
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)  # (B, L, D)
        return output


class LiquidMambaAttention(nn.Module):
    """
    Liquid Mamba Attention: Kết hợp Selective SSM với LTC scaling.
    Thay thế Multi-Head Attention bằng Mamba's approach.
    """
    def __init__(
        self,
        d_model: int,
        state_size: int = 16,
        expand_factor: int = 2,
        conv_kernel_size: int = 4,
        dropout: float = 0.1,
        use_liquid_scaling: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.expand_factor = expand_factor
        self.d_inner = d_model * expand_factor
        
        # Input projections (như trong Mamba)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution layer (local processing)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel_size,
            groups=self.d_inner,
            padding=conv_kernel_size - 1
        )
        
        # Selective SSM
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            state_size=state_size,
            use_liquid_scaling=use_liquid_scaling
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:  # None để tương thích với attention interface
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask (không sử dụng trong Mamba)
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: None (Mamba không có explicit attention weights)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Input projection và split
        x_proj = self.in_proj(x)  # (B, L, 2*d_inner)
        x1, x2 = x_proj.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Gate mechanism
        x1 = F.silu(x1)  # SiLU activation
        
        # Convolution (local processing)
        x1_conv = rearrange(x1, 'b l d -> b d l')
        x1_conv = self.conv1d(x1_conv)[:, :, :seq_len]  # Trim padding
        x1_conv = rearrange(x1_conv, 'b d l -> b l d')
        
        # Apply activation
        x1_conv = F.silu(x1_conv)
        
        # Selective SSM (global processing with liquid scaling)
        ssm_output = self.ssm(x1_conv)
        
        # Element-wise product với gate
        output = ssm_output * x2
        
        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, None
    
    def reset_liquid_state(self):
        """Reset the liquid state in the SSM component."""
        if hasattr(self.ssm, 'reset_liquid_state'):
            self.ssm.reset_liquid_state()

class MultiScaleLiquidMamba(nn.Module):
    """
    Advanced version với multiple time scales và parallel processing.
    Sử dụng multiple Mamba branches với different liquid time constants.
    """
    def __init__(
        self,
        d_model: int,
        num_scales: int = 3,
        state_size: int = 16,
        expand_factor: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        
        # Multiple Mamba branches với different time scales
        self.mamba_branches = nn.ModuleList([
            LiquidMambaAttention(
                d_model=d_model,
                state_size=state_size,
                expand_factor=expand_factor,
                use_liquid_scaling=True
            ) for _ in range(num_scales)
        ])
        
        # Set different tau values cho different branches
        for i, branch in enumerate(self.mamba_branches):
            if hasattr(branch.ssm, 'liquid_scaler'):
                # Different time constants: fast, medium, slow
                tau_value = 0.5 * (2 ** i)  # 0.5, 1.0, 2.0, ...
                nn.init.constant_(branch.ssm.liquid_scaler.tau, tau_value)
        
        # Mixing weights
        self.scale_mixer = nn.Linear(d_model, num_scales)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, d_model = x.shape
        
        # Process through multiple scales
        branch_outputs = []
        for branch in self.mamba_branches:
            output, _ = branch(x, mask)
            branch_outputs.append(output)
        
        # Stack outputs
        stacked_outputs = torch.stack(branch_outputs, dim=-1)  # (B, L, D, num_scales)
        
        # Compute mixing weights based on input
        mixing_weights = torch.softmax(self.scale_mixer(x), dim=-1)  # (B, L, num_scales)
        mixing_weights = mixing_weights.unsqueeze(-2)  # (B, L, 1, num_scales)
        
        # Mix outputs
        mixed_output = torch.sum(stacked_outputs * mixing_weights, dim=-1)  # (B, L, D)
        
        # Final projection
        output = self.output_proj(mixed_output)
        
        return output, None
    
    def reset_liquid_state(self):
        """Reset liquid states for all branches"""
        for branch in self.mamba_branches:
            if hasattr(branch, 'reset_liquid_state'):
                branch.reset_liquid_state()
    

# Example usage và comparison
if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 128, 512
    x = torch.randn(batch_size, seq_len, d_model)

    print("=== Testing LiquidMambaAttention ===")
    liquid_mamba = LiquidMambaAttention(
        d_model=d_model,
        state_size=16,
        use_liquid_scaling=True
    )
    
    output, _ = liquid_mamba(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nTesting liquid dynamics over multiple steps...")
    liquid_mamba.reset_liquid_state()
    
    for step in range(5):
        output, _ = liquid_mamba(x)
        mean_output = output.mean().item()
        print(f"Step {step}: Mean output = {mean_output:.6f}")
    
    print("\n=== Testing MultiScaleLiquidMamba ===")
    multi_scale_mamba = MultiScaleLiquidMamba(
        d_model=d_model,
        num_scales=3,
        state_size=16
    )
    
    output, _ = multi_scale_mamba(x)
    print(f"Multi-scale output shape: {output.shape}")

    num_params_mamba = sum(p.numel() for p in liquid_mamba.parameters())
    num_params_multi = sum(p.numel() for p in multi_scale_mamba.parameters())
    
    print(f"LiquidMamba parameters: {num_params_mamba:,}")
    print(f"MultiScaleMamba parameters: {num_params_multi:,}")
    
    print("\n=== Testing scalability ===")
    for test_seq_len in [64, 128, 256, 512]:
        test_x = torch.randn(1, test_seq_len, d_model)
        
        import time
        start_time = time.time()
        output, _ = liquid_mamba(test_x)
        end_time = time.time()
        
        print(f"Seq len {test_seq_len}: {(end_time - start_time)*1000:.2f}ms")