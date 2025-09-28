#!/usr/bin/env python3
"""
General script to extract encoder from SSL checkpoint
Handles various checkpoint formats from different training stages
"""

import torch
import argparse
from pathlib import Path
import sys


def extract_encoder(checkpoint_path, output_path=None):
    """
    Extract encoder weights from various checkpoint formats.

    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Where to save encoder (default: same dir as checkpoint)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # Determine checkpoint format
    print(f"Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {checkpoint.keys()}")

    # Extract encoder based on format
    encoder_state = None
    metadata = {}

    if isinstance(checkpoint, dict):
        # Standard SSL checkpoint format
        if 'encoder_state_dict' in checkpoint:
            print("Found encoder_state_dict directly")
            encoder_state = checkpoint['encoder_state_dict']
            metadata['epoch'] = checkpoint.get('epoch', 'unknown')
            metadata['loss'] = checkpoint.get('best_loss', checkpoint.get('loss', 'unknown'))

        # Online encoder from momentum methods
        elif 'online_encoder_state_dict' in checkpoint:
            print("Found online_encoder_state_dict (momentum method)")
            encoder_state = checkpoint['online_encoder_state_dict']
            metadata['epoch'] = checkpoint.get('epoch', 'unknown')
            metadata['loss'] = checkpoint.get('best_loss', checkpoint.get('loss', 'unknown'))

        # Full model state dict
        elif 'model_state_dict' in checkpoint:
            print("Found model_state_dict, extracting encoder weights")
            model_state = checkpoint['model_state_dict']
            encoder_state = {}

            for k, v in model_state.items():
                # Handle various prefixes
                if 'encoder' in k:
                    # Remove common prefixes
                    new_k = k
                    for prefix in ['module.', 'model.', 'online_encoder.', 'encoder.']:
                        if k.startswith(prefix):
                            new_k = k[len(prefix):]
                            break
                    encoder_state[new_k] = v

            metadata['epoch'] = checkpoint.get('epoch', 'unknown')
            metadata['loss'] = checkpoint.get('loss', 'unknown')

        # State dict directly in checkpoint
        elif 'state_dict' in checkpoint:
            print("Found state_dict, extracting encoder weights")
            state_dict = checkpoint['state_dict']
            encoder_state = {}

            for k, v in state_dict.items():
                if 'encoder' in k:
                    new_k = k.replace('module.', '').replace('model.', '')
                    encoder_state[new_k] = v

            metadata['epoch'] = checkpoint.get('epoch', 'unknown')

        else:
            # Try to find any encoder-related keys
            print("Looking for encoder weights in checkpoint keys...")
            encoder_state = {}
            for k, v in checkpoint.items():
                if isinstance(v, torch.Tensor) and 'encoder' in k.lower():
                    encoder_state[k] = v

    else:
        # Checkpoint is directly the state dict
        print("Checkpoint appears to be a direct state dict")
        if hasattr(checkpoint, 'keys'):
            encoder_state = checkpoint
        else:
            print("Error: Cannot extract encoder from this checkpoint format")
            sys.exit(1)

    if not encoder_state:
        print("Error: No encoder weights found in checkpoint")
        print("Available keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "N/A")
        sys.exit(1)

    # Clean up encoder state dict keys
    cleaned_encoder_state = {}
    for k, v in encoder_state.items():
        # Remove any remaining prefixes
        clean_k = k
        for prefix in ['encoder.', 'online_encoder.', 'module.encoder.']:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
        cleaned_encoder_state[clean_k] = v

    print(f"Extracted {len(cleaned_encoder_state)} encoder parameters")

    # Show first few parameter names for verification
    param_names = list(cleaned_encoder_state.keys())[:5]
    print(f"Sample parameters: {param_names}")

    # Determine output path
    if output_path is None:
        output_path = checkpoint_path.parent / 'encoder.pt'
    else:
        output_path = Path(output_path)

    # Save encoder with metadata
    save_dict = {
        'encoder_state_dict': cleaned_encoder_state,
        'source_checkpoint': str(checkpoint_path),
        **metadata
    }

    torch.save(save_dict, output_path)
    print(f"\nEncoder saved to: {output_path}")
    print(f"Metadata: epoch={metadata.get('epoch', 'unknown')}, loss={metadata.get('loss', 'unknown')}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Extract encoder from SSL checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--output', '-o', type=str, help='Output path for encoder (default: same dir)')
    parser.add_argument('--verify', action='store_true', help='Verify encoder can be loaded')

    args = parser.parse_args()

    # Extract encoder
    encoder_path = extract_encoder(args.checkpoint, args.output)

    # Verify if requested
    if args.verify:
        print("\nVerifying encoder...")
        try:
            encoder_checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
            encoder_state = encoder_checkpoint['encoder_state_dict']
            print(f"✓ Encoder loaded successfully with {len(encoder_state)} parameters")

            # Check parameter shapes
            total_params = sum(p.numel() for p in encoder_state.values())
            print(f"✓ Total parameters: {total_params:,}")

        except Exception as e:
            print(f"✗ Error verifying encoder: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()