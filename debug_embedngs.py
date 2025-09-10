# debug_model.py
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from device import get_device_manager
from evaluate import DownstreamEvaluator, create_evaluation_dataset


def visualize_embeddings(self, dataset, save_dir: str = None):
    """Visualize and analyze embeddings to debug poor performance."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns

    # Extract embeddings if not cached
    embeddings, labels_list, participant_ids = self.extract_embeddings(
        dataset, aggregate_by_participant=True
    )

    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS")
    print("=" * 60)

    # 1. Check embedding statistics
    print("\n1. Embedding Statistics:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.4f}")
    print(f"   Std: {embeddings.std():.4f}")
    print(f"   Min: {embeddings.min():.4f}")
    print(f"   Max: {embeddings.max():.4f}")

    # Check for dead neurons (always zero)
    dead_neurons = np.sum(np.abs(embeddings).max(axis=0) < 1e-6)
    print(f"   Dead neurons: {dead_neurons}/{embeddings.shape[1]}")

    # Check for collapsed embeddings (all similar)
    pairwise_dist = np.std([np.linalg.norm(embeddings[i] - embeddings[j])
                            for i in range(min(100, len(embeddings)))
                            for j in range(i + 1, min(100, len(embeddings)))])
    print(f"   Embedding diversity (std of distances): {pairwise_dist:.4f}")

    # 2. Extract labels for visualization
    ages = []
    sexes = []
    bmis = []

    for label_dict in labels_list:
        if isinstance(label_dict, dict):
            ages.append(label_dict.get('age', -1))
            sexes.append(label_dict.get('sex', -1))
            bmis.append(label_dict.get('bmi', -1))

    ages = np.array(ages)
    sexes = np.array(sexes)
    bmis = np.array(bmis)

    # Filter valid samples
    valid_mask = (ages > 0) & (sexes >= 0) & (bmis > 0)
    embeddings_valid = embeddings[valid_mask]
    ages_valid = ages[valid_mask]
    sexes_valid = sexes[valid_mask]
    bmis_valid = bmis[valid_mask]

    print(f"\n2. Valid samples for visualization: {valid_mask.sum()}/{len(embeddings)}")

    if valid_mask.sum() < 10:
        print("   ⚠️ Too few valid samples for visualization")
        return

    # 3. Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # PCA visualization
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(embeddings_valid)

    print(f"\n3. PCA explained variance: {pca.explained_variance_ratio_}")

    # Plot PCA colored by age
    scatter1 = axes[0, 0].scatter(emb_pca[:, 0], emb_pca[:, 1],
                                  c=ages_valid, cmap='viridis', alpha=0.6)
    axes[0, 0].set_title('PCA - Colored by Age')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.colorbar(scatter1, ax=axes[0, 0])

    # Plot PCA colored by sex
    axes[0, 1].scatter(emb_pca[:, 0], emb_pca[:, 1],
                       c=sexes_valid, cmap='coolwarm', alpha=0.6)
    axes[0, 1].set_title('PCA - Colored by Sex')
    axes[0, 1].set_xlabel(f'PC1')
    axes[0, 1].set_ylabel(f'PC2')

    # Plot PCA colored by BMI
    scatter3 = axes[0, 2].scatter(emb_pca[:, 0], emb_pca[:, 1],
                                  c=bmis_valid, cmap='plasma', alpha=0.6)
    axes[0, 2].set_title('PCA - Colored by BMI')
    axes[0, 2].set_xlabel(f'PC1')
    axes[0, 2].set_ylabel(f'PC2')
    plt.colorbar(scatter3, ax=axes[0, 2])

    # t-SNE visualization (if not too many samples)
    if len(embeddings_valid) <= 1000:
        print("\n4. Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_valid) - 1))
        emb_tsne = tsne.fit_transform(embeddings_valid)

        # t-SNE colored by age
        scatter4 = axes[1, 0].scatter(emb_tsne[:, 0], emb_tsne[:, 1],
                                      c=ages_valid, cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('t-SNE - Colored by Age')
        plt.colorbar(scatter4, ax=axes[1, 0])

        # t-SNE colored by sex
        axes[1, 1].scatter(emb_tsne[:, 0], emb_tsne[:, 1],
                           c=sexes_valid, cmap='coolwarm', alpha=0.6)
        axes[1, 1].set_title('t-SNE - Colored by Sex')

        # t-SNE colored by BMI
        scatter6 = axes[1, 2].scatter(emb_tsne[:, 0], emb_tsne[:, 1],
                                      c=bmis_valid, cmap='plasma', alpha=0.6)
        axes[1, 2].set_title('t-SNE - Colored by BMI')
        plt.colorbar(scatter6, ax=axes[1, 2])

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'embedding_visualization.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n5. Visualization saved to {save_path}")

    plt.show()

    # 6. Check correlation between embeddings and labels
    print("\n6. Embedding-Label Correlations:")
    from scipy.stats import pearsonr

    # Calculate correlation for each embedding dimension with age
    age_correlations = []
    for i in range(min(10, embeddings_valid.shape[1])):  # Check first 10 dims
        corr, p_val = pearsonr(embeddings_valid[:, i], ages_valid)
        age_correlations.append(abs(corr))

    print(f"   Max correlation with age: {max(age_correlations):.4f}")
    print(f"   Mean correlation with age: {np.mean(age_correlations):.4f}")

    # 7. Check if embeddings are collapsed
    print("\n7. Embedding Collapse Check:")
    distances = []
    n_samples = min(100, len(embeddings_valid))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances.append(np.linalg.norm(embeddings_valid[i] - embeddings_valid[j]))

    print(f"   Mean pairwise distance: {np.mean(distances):.4f}")
    print(f"   Std pairwise distance: {np.std(distances):.4f}")

    if np.std(distances) < 0.1:
        print("   ⚠️ WARNING: Embeddings may be collapsed (very low variance)")

    return embeddings_valid, ages_valid, sexes_valid, bmis_valid
def debug_encoder(encoder_path: str, dataset_type: str = 'vitaldb'):
    """Debug encoder to understand poor performance."""

    print("\n" + "=" * 60)
    print("ENCODER DEBUGGING")
    print("=" * 60)

    # Create evaluator
    evaluator = DownstreamEvaluator(
        encoder_path=encoder_path,
        device_manager=get_device_manager()
    )

    # Create dataset
    dataset = create_evaluation_dataset(
        dataset_type=dataset_type,
        modality='ppg',
        split='test'
    )

    # 1. Check model weights
    print("\n1. Model Weight Analysis:")
    model = evaluator.encoder

    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_stats = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'max': param.data.max().item(),
                'min': param.data.min().item(),
                'zeros': (param.data == 0).sum().item() / param.data.numel()
            }
            if weight_stats['std'] < 0.01:
                print(f"   ⚠️ {name}: Very low std = {weight_stats['std']:.6f}")
            if weight_stats['zeros'] > 0.5:
                print(f"   ⚠️ {name}: {weight_stats['zeros']:.1%} zeros")

    # 2. Visualize embeddings
    print("\n2. Generating embedding visualizations...")
    embeddings, ages, sexes, bmis = evaluator.visualize_embeddings(
        dataset,
        save_dir=f"debug_output/{dataset_type}"
    )

    # 3. Test with random input
    print("\n3. Testing with random input:")
    random_input = torch.randn(10, 1, 640).to(evaluator.device)
    with torch.no_grad():
        random_embeddings = evaluator.encoder(random_input)

    print(f"   Random input embeddings std: {random_embeddings.std().item():.4f}")

    # 4. Test with actual signals
    print("\n4. Testing with actual signals:")
    real_signals = []
    for i in range(min(10, len(dataset))):
        seg1, seg2, *_ = dataset[i]
        real_signals.append(seg1)

    real_batch = torch.stack(real_signals).to(evaluator.device)
    with torch.no_grad():
        real_embeddings = evaluator.encoder(real_batch)

    print(f"   Real signal embeddings std: {real_embeddings.std().item():.4f}")

    # Compare random vs real
    random_mean = random_embeddings.mean(dim=0)
    real_mean = real_embeddings.mean(dim=0)
    similarity = torch.cosine_similarity(random_mean, real_mean, dim=0).item()
    print(f"   Cosine similarity (random vs real): {similarity:.4f}")

    if similarity > 0.95:
        print("   ⚠️ WARNING: Embeddings for random and real signals are too similar!")

    return evaluator
if __name__ == "__main__":
    import sys

    # Path to your trained encoder
    encoder_path = "data/outputs/checkpoints/vitaldb_ppg_20250901_064928/encoder.pt"

    # Debug pre-trained model on VitalDB
    print("=" * 60)
    print("DEBUGGING PRE-TRAINED MODEL ON VITALDB")
    print("=" * 60)
    evaluator = debug_encoder(encoder_path, dataset_type='vitaldb')

    # Debug fine-tuned model
    encoder_path_finetuned = "data/outputs/checkpoints/finetune_but_ppg_ppg_20250910_185818/encoder.pt"

    print("\n" + "=" * 60)
    print("DEBUGGING FINE-TUNED MODEL ON BUT PPG")
    print("=" * 60)
    evaluator_finetuned = debug_encoder(encoder_path_finetuned, dataset_type='but_ppg')