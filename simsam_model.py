# # simsiam_model.py
# """
# SimSiam implementation for small-scale SSL training.
# Follows the exact structure of ssl_model.py for compatibility.
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Tuple, Optional
# import yaml
# from pathlib import Path
# from copy import deepcopy
# from device import get_device_manager
#
#
#
#
# def create_simsiam_model(
#         encoder: nn.Module,
#         projection_head: nn.Module,  # Ignored, but kept for interface compatibility
#         config_path: str = 'configs/config.yaml'
# ) -> SimSiamModel:
#     """Create SimSiam model with configuration."""
#     return SimSiamModel(encoder, projection_head, config_path)
#
#
# # ============= TEST FUNCTION =============
#
#
#
#
# if __name__ == "__main__":
#     test_simsiam()