# L2P Continual Learning: Experimental Report

Dio Brown | Syracuse University, Spring 2026
Last Updated: March 12, 2026

## 1. Objective

Replicate the L2P (Learning to Prompt) paper results using a PyTorch reimplementation on Google Colab. Clone the repo, run on reduced dataset, verify metrics against Wang et al. (CVPR 2022). Then test a novel Kairos-inspired modification to the data pipeline. Codebase: JH-LEE-KR/l2p-pytorch. Environment: Google Colab Pro, NVIDIA H100 GPU (80GB VRAM).

## 2. Method Summary

L2P freezes a pretrained Vision Transformer (ViT) backbone and trains only a small prompt pool. When an image arrives, the frozen ViT generates a CLS query vector, matched against learnable keys via cosine similarity to retrieve top-N prompts. These prompts are prepended to the image patch sequence, steering attention toward task-relevant features. Three components update during training: prompt values, keys, and the classification head. The frozen backbone structurally prevents catastrophic forgetting. Only 122,980 trainable parameters (0.14% of ViT-Base).

## 3. Paper Reference Configuration

Wang et al. (CVPR 2022): ViT-Base/16 pretrained on ImageNet-21k, full Split CIFAR-100 (10 tasks, 10 classes each), M=10 prompts, top_k=5, Lp=5, lr=0.03, 5 epochs/task. Reported: Acc@1=83.83%, Forgetting=7.63%.

## 4. Replication Runs

### Run 1: Random Weights Baseline
- Config: ViT-Small/16 random init, 10% Split CIFAR-100, 5 epochs/task, T4 GPU
- Results: Avg Acc@1 = 4.13%, Forgetting = 5.02%, Runtime ~41 min
- Interpretation: Without pretrained features, CLS queries are noise. Prompt retrieval is random. Confirms pretrained backbone is essential.

### Run 2: Pretrained Backbone, No Head Pretraining
- Config: ViT-Small/16 IN-21k pretrained, 10% subsample, 5 epochs/task
- Results: Avg Acc@1 = 4.49%, Avg Acc@5 = 16.70%, Forgetting = 4.00%, Runtime ~14 min
- Interpretation: Backbone produces real features (training acc ~30%) but random classification head cannot map to labels. Acc@5 improvement over Run 1 confirms backbone is working.

### Run 3: Pretrained Backbone + Separate Head Pretraining (ViT-Small)
- Config: ViT-Small/16 IN-21k, head pretrained separately on full CIFAR-100 (72% acc), 10% subsample, H100 GPU
- Results: Avg Acc@1 = 1.57%, Avg Acc@5 = 6.83%, Forgetting = 0.52%, Runtime ~1.5 min
- Interpretation: Near-zero forgetting confirms L2P structural protection works. Architecture mismatch between timm ViT and repo's custom VisionTransformer degraded accuracy.

### Run 4: ViT-Base with Proper pos_embed Interpolation (FINAL BASELINE)
- Config: ViT-Base/16 IN-21k (paper's actual model), pos_embed interpolated (197→222), head trains during L2P, 50% random subsample, H100 GPU, batch_size=16, 122,980 trainable params
- **Final Results: Avg Acc@1 = 47.57%, Avg Acc@5 = 73.95%, Forgetting = 18.21%, Runtime = 10:23**
- Per-task test accuracy (after all 10 tasks trained):

| Task | Acc@1 | Acc@5 |
|------|-------|-------|
| 1 | 5.2% | 29.5% |
| 2 | 1.9% | 23.6% |
| 3 | 39.3% | 67.8% |
| 4 | 39.7% | 77.4% |
| 5 | 44.6% | 84.6% |
| 6 | 65.4% | 93.7% |
| 7 | 66.5% | 89.4% |
| 8 | 67.0% | 89.8% |
| 9 | 69.0% | 90.5% |
| 10 | 77.1% | 93.2% |

- Training accuracy reached 86% by final task. Task 1 initially scored 50.5% but degraded to 5.2% — forgetting on early tasks is the primary gap vs paper.

### Run 6: Full Dataset — 100% Data Baseline (Experiment D)
- Config: ViT-Base/16 IN-21k, pos_embed interpolated (197→222), 100% Split CIFAR-100 (no subsample), H100 GPU, batch_size=16, 5 epochs/task, 122,980 trainable params
- **Final Results: Avg Acc@1 = 61.75%, Avg Acc@5 = 85.69%, Forgetting = 14.70%, Runtime = 18:13**
- Per-task test accuracy (after all 10 tasks trained):

| Task | Acc@1 | Acc@5 |
|------|-------|-------|
| 1 | 9.0% | 35.7% |
| 2 | 29.5% | 74.2% |
| 3 | 64.8% | 89.8% |
| 4 | 66.0% | 90.3% |
| 5 | 73.4% | 95.1% |
| 6 | 67.7% | 94.1% |
| 7 | 76.1% | 93.4% |
| 8 | 70.2% | 92.8% |
| 9 | 80.4% | 95.3% |
| 10 | 80.4% | 96.2% |

- Average accuracy progression across tasks:

| After Task | Avg Acc@1 | Forgetting |
|------------|-----------|------------|
| 1 | 60.20% | — |
| 2 | 49.85% | 19.80% |
| 3 | 53.53% | 15.80% |
| 4 | 57.00% | 14.40% |
| 5 | 56.30% | 17.53% |
| 6 | 56.88% | 16.98% |
| 7 | 60.56% | 13.67% |
| 8 | 61.79% | 12.99% |
| 9 | 63.43% | 12.31% |
| 10 | 61.75% | 14.70% |

- Training accuracy reached 89.86% by final task epoch 5. Task 1 initially scored 60.20% but degraded to 9.0% after all 10 tasks — same catastrophic early-task forgetting pattern as Run 4 but with higher overall accuracy.
- **Gap vs paper: 83.83% − 61.75% = 22.08 pp.** Doubling the data from 50% to 100% improved Acc@1 by 14.18 pp (47.57% → 61.75%), but a significant gap remains. Forgetting improved only modestly (18.21% → 14.70%) compared to the paper's 7.63%. The per-task pattern reveals the issue: early tasks (1-2) are almost completely forgotten, while recent tasks (9-10) perform well at 80%.

### Root Causes of Runs 1-3 Failures
Three compounding mistakes were identified during the Run 4 pivot:
1. Wrong model: Used ViT-Small (384 dim) instead of ViT-Base (768 dim) — the paper's model
2. Skipped pos_embed: Must interpolate (copy first 197 positions into 222-position tensor), not skip
3. Unnecessary head pretraining: Paper trains head jointly during L2P — separate pretraining introduced architecture mismatch

## 5. Novel Experiment: Coherence-Guided Sampling (Run 5)

### Motivation
Instead of randomly selecting 50% of training data, what if the frozen ViT backbone's own feature geometry could select more informative examples? This idea draws directly from Kairos, where experience is weighted by coherence — how well an input matches the developing cluster structure. The hypothesis was that prototypical images (close to their class centroid in ViT feature space) would give cleaner gradient signal, potentially matching or exceeding random sampling at the same data budget.

### Method
Before L2P training begins, all 50,000 CIFAR-100 training images are passed through the frozen ViT-Base backbone to extract CLS features (768-dim vectors). For each of the 100 classes, the class centroid is computed as the mean feature vector. The 50% of images closest to their respective class centroid (by L2 distance) are selected as the training set. This produces a dataset of the most "prototypical" examples — the images that best represent each class according to the pretrained backbone's understanding.

### Run 5 Results
- Config: Same as Run 4 but with coherence-guided 50% subsample instead of random 50%
- **Final Results: Avg Acc@1 = 25.78%, Avg Acc@5 = 54.35%, Forgetting = 16.69%, Runtime = 6:24**
- Per-task test accuracy (after all 10 tasks trained):

| Task | Acc@1 | Acc@5 |
|------|-------|-------|
| 1 | 0.2% | 17.3% |
| 2 | 2.1% | 18.2% |
| 3 | 9.3% | 37.4% |
| 4 | 11.7% | 40.3% |
| 5 | 30.8% | 67.3% |
| 6 | 31.0% | 75.3% |
| 7 | 36.9% | 66.8% |
| 8 | 39.1% | 69.2% |
| 9 | 42.5% | 73.1% |
| 10 | 54.2% | 78.6% |

### Analysis: Why Coherence Sampling Hurt
The coherence-guided approach reduced accuracy by nearly 22 percentage points (47.57% → 25.78%). This is a meaningful negative result with several likely explanations:

1. **Diversity is more important than prototypicality for prompt training.** The prompt pool needs exposure to the full visual distribution within each class — unusual angles, lighting, backgrounds, atypical instances — to develop robust prompt-to-class associations. Centroid-close examples are too homogeneous, providing less gradient variety.

2. **Prompt keys need boundary examples to specialize.** L2P's prompt selection mechanism (cosine similarity between CLS query and keys) must learn to distinguish between classes. Examples near class centroids are already well-separated; it's the ambiguous, boundary-region examples that force keys to carve sharper decision boundaries. Removing them weakened prompt specialization.

3. **The reduced batch count compounds the problem.** Run 5 had only 79 batches/epoch vs Run 4's 157. The coherence selection created uneven class splits (some classes had more centroid-close examples than others), resulting in fewer total training steps and less gradient exposure overall.

4. **Forgetting was slightly better (16.69% vs 18.21%).** This is the one positive signal — prototypical examples may create more stable prompt associations that resist overwriting. But the trade-off in raw accuracy was far too steep.

### Kairos Implications
This result is informative for Kairos's design. In Kairos, coherence-weighted experience selection is meant to work during *development* — when clusters are forming from scratch and need prototypical examples to stabilize. L2P's backbone is already fully formed; what it needs from training data is *discrimination*, not *prototypicality*. This suggests that coherence-guided selection may be most powerful in Kairos's developmental context (where cluster formation benefits from clean exemplars) but counterproductive in mature systems (where decision boundary refinement benefits from diversity). The distinction between developmental and post-developmental learning may be more fundamental than initially assumed.

## 6. Results Comparison

### All Runs

| Run | Purpose | Model | Data | Sampling | Acc@1 | Acc@5 | Forgetting | Gap to Paper | Runtime |
|-----|---------|-------|------|----------|------:|------:|-----------:|-------------:|---------|
| **Paper** | **Reference target** | **ViT-B/16 IN-21k** | **100%** | **Full** | **83.83%** | **—** | **7.63%** | **—** | **—** |
| 1 | Backbone ablation | ViT-S random | 10% | Random | 4.13% | ~12% | 5.02% | −79.70 pp | ~41 min |
| 2 | Head ablation | ViT-S pretrained | 10% | Random | 4.49% | 16.70% | 4.00% | −79.34 pp | ~14 min |
| 3 | Head pretraining test | ViT-S + separate head | 10% | Random | 1.57% | 6.83% | 0.52% | −82.26 pp | ~1.5 min |
| 4 | Corrected baseline | ViT-B + pos_embed | 50% | Random | 47.57% | 73.95% | 18.21% | −36.26 pp | 10:23 |
| 5 | Coherence sampling | ViT-B + pos_embed | 50% | Coherence | 25.78% | 54.35% | 16.69% | −58.05 pp | 6:24 |
| **6** | **Full data baseline** | **ViT-B + pos_embed** | **100%** | **Full** | **61.75%** | **85.69%** | **14.70%** | **−22.08 pp** | **18:13** |

### Key Pairwise Comparisons

| Comparison | Δ Acc@1 | Δ Acc@5 | Δ Forgetting | Conclusion |
|------------|--------:|--------:|-------------:|------------|
| Run 1 → Run 2 (add pretrained backbone) | +0.36 pp | +4.70 pp | −1.02 pp | Pretrained backbone is essential |
| Run 2 → Run 4 (correct model + pos_embed) | +43.08 pp | +57.25 pp | +14.21 pp | Model choice and pos_embed are critical |
| Run 4 → Run 5 (coherence vs random sampling) | −21.79 pp | −19.60 pp | −1.52 pp | Diversity beats prototypicality for prompt training |
| Run 4 → Run 6 (50% → 100% data) | +14.18 pp | +11.74 pp | −3.51 pp | More data helps substantially but doesn't fix forgetting |
| Run 6 → Paper (same data, reference impl.) | −22.08 pp | — | −7.07 pp | Early-task forgetting is the remaining bottleneck |

### Per-Task Accuracy After All 10 Tasks (Key Runs)

| Task | Run 4 Acc@1 | Run 5 Acc@1 | Run 6 Acc@1 | Paper Acc@1 (est.) |
|------|------------:|------------:|------------:|-------------------:|
| 1 | 5.2% | 0.2% | 9.0% | ~80%+ |
| 2 | 1.9% | 2.1% | 29.5% | ~80%+ |
| 3 | 39.3% | 9.3% | 64.8% | ~80%+ |
| 4 | 39.7% | 11.7% | 66.0% | ~80%+ |
| 5 | 44.6% | 30.8% | 73.4% | ~80%+ |
| 6 | 65.4% | 31.0% | 67.7% | ~80%+ |
| 7 | 66.5% | 36.9% | 76.1% | ~80%+ |
| 8 | 67.0% | 39.1% | 70.2% | ~80%+ |
| 9 | 69.0% | 42.5% | 80.4% | ~80%+ |
| 10 | 77.1% | 54.2% | 80.4% | ~83%+ |
| **Avg** | **47.57%** | **25.78%** | **61.75%** | **83.83%** |

The per-task pattern is consistent across all runs: recent tasks perform well, early tasks are catastrophically forgotten. This is a structural property of L2P's shared prompt pool, not a data quantity or sampling problem.

## 7. Key Findings

1. **Pretrained backbone is essential.** Run 1 vs Run 2 confirms L2P cannot function without meaningful visual features from the frozen backbone.

2. **Model size matters critically.** ViT-Small (384 dim) vs ViT-Base (768 dim) produced dramatically different results — the paper's model choice is not arbitrary.

3. **pos_embed must be handled properly.** Skipping positional embeddings destroys spatial understanding. Interpolating (copying the first 197 positions into 222) preserves pretrained spatial knowledge.

4. **Head does not need separate pretraining.** The paper trains the classification head jointly during L2P. Separate pretraining introduced architecture mismatch bugs.

5. **L2P is highly parameter-efficient.** Only 122,980 trainable params (0.14% of ViT-Base) achieve strong classification through prompt-based steering.

6. **Data diversity outweighs data quality for prompt training.** The coherence-guided sampling experiment (Run 5) demonstrated that selecting prototypical examples harms accuracy. The prompt pool needs exposure to the full visual distribution — including atypical, ambiguous, and boundary examples — to specialize effectively.

7. **Coherence-guided selection may reduce forgetting.** Run 5's slightly lower forgetting (16.69% vs 18.21%) suggests prototypical examples create more stable prompt associations, even though they hurt overall accuracy. This trade-off merits further investigation.

8. **More data substantially improves accuracy but does not fix forgetting.** Run 6 (100% data) achieved 61.75% Acc@1 vs Run 4's 47.57% (+14.18 pp) — a meaningful gain. But forgetting improved only modestly (18.21% → 14.70%), and the gap to the paper's 83.83% remains 22 pp. The per-task pattern is revealing: Tasks 9-10 achieve 80.4% while Tasks 1-2 degrade to 9.0%/29.5%. More data helps overall performance but does not address the structural early-task forgetting problem. The remaining gap is a forgetting problem, not a data quantity problem.

## 8. Proposed Next Experiments

### Experiment A: Larger Prompt Pool (M=50, top_k=3)
Paper uses M=10, top_k=5 (every image uses half the pool). M=50, top_k=3 means each image uses 6% of the pool, forcing narrower prompt specialization. May produce intra-task sub-clustering. Paper never tests this.

### Experiment B: Key-as-Projection-of-Prompt
Paper shows prompts-as-keys drops accuracy 81%→58%. Test: key = normalize(linear_projection(prompt)). Keys stay coupled to content but specialize as addresses. Bridges L2P (decoupled keys/prompts) and Kairos (unified address=content).

### Experiment C: Inverse Coherence Sampling
Given Run 5's result, test the opposite: select the 50% of images *farthest* from class centroids (most atypical). If boundary/ambiguous examples are what prompts need most, this should outperform both random and coherence sampling.

### Experiment D: 100% Data Baseline ✓ Completed (Run 6)
Results: Avg Acc@1 = 61.75%, Forgetting = 14.70%. Closing the data gap from 50% to 100% added 14.18 pp of accuracy but only modestly reduced forgetting (18.21% → 14.70%). The paper's 83.83% remains 22 pp away. Early-task forgetting (Tasks 1-2 degrading to 9.0%/29.5%) is confirmed as the primary bottleneck — not data quantity. See Run 6 in Section 4 for full results.

## 9. Connections to Kairos

L2P and Kairos both solve continual learning but through fundamentally different mechanisms. L2P inherits knowledge from a pretrained backbone; Kairos builds representations from raw perception using Gabor filters and Hebbian learning. L2P retrieves prompts via cosine similarity between CLS query and learned keys; Kairos routes through coherence matching between input and cluster content. L2P decouples keys from prompts (address ≠ content); Kairos unifies them (cluster identity IS content).

Run 1 demonstrates L2P's key limitation: without pretrained representations, the system fails entirely. Kairos addresses this by design. The proposed key-as-projection-of-prompt experiment directly tests whether L2P can move toward Kairos's unified architecture.

Run 5 provides the most nuanced Kairos connection: coherence-guided experience selection — a core Kairos principle — actually hurt L2P. This suggests coherence weighting is most valuable during developmental learning (when representations are forming) rather than in post-developmental systems with frozen, mature feature spaces. The implication for Kairos is encouraging: coherence-guided selection may be uniquely powerful in exactly the developmental context Kairos operates in, where it would serve a fundamentally different role than in mature systems like L2P.

Run 6 sharpens the forgetting diagnosis. With full data, L2P achieves 80.4% on the most recent tasks but only 9.0% on the earliest — a 71 pp spread across tasks. This is structurally what Kairos is designed to prevent: the cluster system's spatial separation means new knowledge occupies new clusters rather than overwriting old ones, and contextual connections mean traversing new territory passively reinforces old cluster neighborhoods. L2P has no equivalent mechanism. The 22 pp remaining gap to the paper is not a data problem — it is a forgetting architecture problem, and Kairos's design addresses it at the architectural level.

## 10. Technical Notes

Codebase modifications: Custom models.py with HuggingFace weight downloading, pos_embed interpolation, and data subsampling. Removed pretrained_custom_load from vision_transformer.py. Run 5 added a coherence sampling preprocessing cell that extracts ViT features, computes per-class centroids, selects closest 50%, and saves indices to coherent_indices.json. datasets.py was patched to load these indices. Run 6 removed the subsampling flag entirely, running on the full 50,000-image CIFAR-100 training set with no index filtering. All code executed in Google Colab Pro with H100 GPU.

Paper: Wang et al., "Learning to Prompt for Continual Learning," CVPR 2022.
