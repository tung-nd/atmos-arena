import argparse
import torch
from torch.utils.flop_counter import FlopCounterMode
from atmos_arena.climax_arch import ClimaX
from atmos_arena.stormer_arch import Stormer
from atmos_arena.unet_arch import Unet

def count_parameters(model):
    """Count total parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def format_count(count):
    """Format count in billions (B) or millions (M)."""
    if count >= 1e9:
        return f"{count/1e9:.3f}B"
    elif count >= 1e6:
        return f"{count/1e6:.3f}M"
    else:
        return f"{count/1e3:.3f}K"

def main(model):
    device = 'cuda'
    if model == 'climax':
        default_vars = [
            "land_sea_mask",
            "orography",
            "lattitude",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "geopotential_50",
            "geopotential_250",
            "geopotential_500",
            "geopotential_600",
            "geopotential_700",
            "geopotential_850",
            "geopotential_925",
            "u_component_of_wind_50",
            "u_component_of_wind_250",
            "u_component_of_wind_500",
            "u_component_of_wind_600",
            "u_component_of_wind_700",
            "u_component_of_wind_850",
            "u_component_of_wind_925",
            "v_component_of_wind_50",
            "v_component_of_wind_250",
            "v_component_of_wind_500",
            "v_component_of_wind_600",
            "v_component_of_wind_700",
            "v_component_of_wind_850",
            "v_component_of_wind_925",
            "temperature_50",
            "temperature_250",
            "temperature_500",
            "temperature_600",
            "temperature_700",
            "temperature_850",
            "temperature_925",
            "relative_humidity_50",
            "relative_humidity_250",
            "relative_humidity_500",
            "relative_humidity_600",
            "relative_humidity_700",
            "relative_humidity_850",
            "relative_humidity_925",
            "specific_humidity_50",
            "specific_humidity_250",
            "specific_humidity_500",
            "specific_humidity_600",
            "specific_humidity_700",
            "specific_humidity_850",
            "specific_humidity_925",
        ]
        model = ClimaX(
            default_vars=default_vars,
            img_size=(128, 256),
            patch_size=4,
            embed_dim=1024,
            depth=8,
            decoder_depth=2,
            num_heads=16,
            mlp_ratio=4,
            drop_path=0.1,
            drop_rate=0.1,
        ).to(device)
        x = torch.randn((1, len(default_vars), 128, 256)).to(device)
        lead_time = torch.Tensor([72]).to(device)
        
        # Count and print parameters
        param_count = count_parameters(model)
        print(f"\nClimaX Parameters: {format_count(param_count)}")
        
        # Count and print FLOPs
        flop_counter = FlopCounterMode(model, depth=2)
        with flop_counter:
            preds = model(x, lead_time, default_vars, default_vars)
            
    elif model == 'stormer':
        in_variables = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "geopotential_50",
            "geopotential_100",
            "geopotential_150",
            "geopotential_200",
            "geopotential_250",
            "geopotential_300",
            "geopotential_400",
            "geopotential_500",
            "geopotential_600",
            "geopotential_700",
            "geopotential_850",
            "geopotential_925",
            "geopotential_1000",
            "u_component_of_wind_50",
            "u_component_of_wind_100",
            "u_component_of_wind_150",
            "u_component_of_wind_200",
            "u_component_of_wind_250",
            "u_component_of_wind_300",
            "u_component_of_wind_400",
            "u_component_of_wind_500",
            "u_component_of_wind_600",
            "u_component_of_wind_700",
            "u_component_of_wind_850",
            "u_component_of_wind_925",
            "u_component_of_wind_1000",
            "v_component_of_wind_50",
            "v_component_of_wind_100",
            "v_component_of_wind_150",
            "v_component_of_wind_200",
            "v_component_of_wind_250",
            "v_component_of_wind_300",
            "v_component_of_wind_400",
            "v_component_of_wind_500",
            "v_component_of_wind_600",
            "v_component_of_wind_700",
            "v_component_of_wind_850",
            "v_component_of_wind_925",
            "v_component_of_wind_1000",
            "temperature_50",
            "temperature_100",
            "temperature_150",
            "temperature_200",
            "temperature_250",
            "temperature_300",
            "temperature_400",
            "temperature_500",
            "temperature_600",
            "temperature_700",
            "temperature_850",
            "temperature_925",
            "temperature_1000",
            "specific_humidity_50",
            "specific_humidity_100",
            "specific_humidity_150",
            "specific_humidity_200",
            "specific_humidity_250",
            "specific_humidity_300",
            "specific_humidity_400",
            "specific_humidity_500",
            "specific_humidity_600",
            "specific_humidity_700",
            "specific_humidity_850",
            "specific_humidity_925",
            "specific_humidity_1000",
        ]
        model = Stormer(
            in_variables=in_variables,
            in_img_size=(128, 256),
            patch_size=2,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
        ).to(device)
        x = torch.randn((1, len(in_variables), 128, 256)).to(device)
        lead_time = torch.Tensor([6]).to(device)
        
        # Count and print parameters
        param_count = count_parameters(model)
        print(f"\nStormer Parameters: {format_count(param_count)}")
        
        # Count and print FLOPs
        flop_counter = FlopCounterMode(model, depth=2)
        with flop_counter:
            preds = model(x, lead_time, in_variables)
            
    elif model == 'unet':
        model = Unet(
            in_channels=69,
            out_channels=69,
            history=1,
            hidden_channels=128,
            activation="leaky",
            norm=True,
            dropout=0.1,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, False],
            mid_attn=False,
            n_blocks=2,
        ).to(device)
        x = torch.randn((1, 69, 128, 256)).to(device)
        
        # Count and print parameters
        param_count = count_parameters(model)
        print(f"\nUNet Parameters: {format_count(param_count)}")
        
        # Count and print FLOPs
        flop_counter = FlopCounterMode(model, depth=2)
        with flop_counter:
            preds = model(x, None, None, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    main(args.model)