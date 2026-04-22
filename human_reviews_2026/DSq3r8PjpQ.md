# QLIP: A Dynamic Quadtree Vision Prior Enhances MLLM Performance Without Retraining

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Multimodal Large Language Models (MLLMs) encode images into visual tokens, aligning visual and textual signals within a shared latent space to facilitate cross-modal representation learning. The CLIP model is a widely adopted foundational vision language model whose vision encoder has played a critical role in the development of MLLMs such as LLaVA. However, the CLIP vision encoder suffers from notable limitations including being constrained to only handling fixed input resolutions and a failure to produce separated embeddings for dissimilar images. Replacing the vision encoder of an existing model typically incurs substantial computational costs because such a change often necessitates retraining the entire model pipeline.

In this work, we identify two factors which underlie the limitations of the CLIP vision encoder: mesoscopic bias and interpolation bias. To address these issues, we propose QLIP, a drop-in replacement for CLIP that can be seamlessly integrated with existing MLLMs with only a few lines of code and can enhance both coarse-grained and fine-grained visual understanding, without re-training. QLIP is designed around an image quadtree which replaces the standard uniform grid patches with a novel content aware patchification.

Our experimental results demonstrate that QLIP improves the general visual question answering accuracy of the LLaVA-1.5 model series across various model sizes—without requiring retraining or fine-tuning of the full MLLM. Notably, QLIP boosts detailed understanding performance on the challenging $V^*$ benchmark by up to 13.6%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes Q-LIP, a plug-and-play CLIP replacement that integrates into MLLMs with minimal code (no full retraining) and uses content-aware quadtree patchification to boost visual understanding.​ Experiments show QLIP improves LLaVA-1.5’s performance in several benchmarks, without full MLLM retraining/fine-tuning.

### Strengths
1. QLIP fixes CLIP’s mesoscopic/interpolation biases with quadtree patchification and a small MLP, handling arbitrary resolutions without encoder weight changes.
2. As a drop-in CLIP replacement, it boosts LLaVA’s  benchmark accuracy by 13.6% without MLLM retraining/fine-tuning.
3. QLIP improves token efficiency (higher accuracy with fewer tokens) and reduces hallucination, performing well on several benchmarks.

### Weaknesses
1. The application of the proposed method is limited. Currently, many large models natively support input image at arbitrary resolutions.
2. The generalization of the method is not validated. This study only conducts experiments on LLaVa-1.5 (7B and 13B), and experiments on additional models should be added.
3. It is advisable to add comparisons between the proposed quadtree patchification and some token merging/pruning methods.

### Questions
Refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces QLIP, a novel, lightweight, and drop-in modification for the CLIP vision encoder designed to enhance the performance of Multimodal Large Language Models (MLLMs), such as LLaVA, without requiring expensive retraining of the entire MLLM backbone. The authors attribute limitations of the standard CLIP encoder—specifically its fixed-resolution nature and poor handling of fine-grained details—to two biases: Mesoscopic Bias and Interpolation Bias. QLIP addresses these issues via two components: Quadtree Patchification (QtP) and Coordinate-Based MLP. Experiments show that QLIP achieves substantial performance improvements, notably a +13.6% accuracy boost on the challenging $V^{*}$ benchmark for fine-grained VQA, and successfully mitigates model hallucination.

### Strengths
1. Practical & Cost-Effective Drop-in Solution: QLIP’s design as a "drop-in replacement" is highly impactful. By significantly enhancing the visual signal without necessitating the re-training or fine-tuning of the entire MLLM pipeline, it offers a practical, low-cost path to upgrading existing MLLMs.

2. Clear Theoretical Motivation: The paper clearly and quantitatively identifies two specific, fundamental biases in the CLIP vision encoder (mesoscopic and interpolation bias). The proposed solutions are elegantly tailored to directly address these theoretical deficiencies.

3. Content-Aware Patchification: The use of Quadtree Patchification (QtP) with a simple gradient-based criterion provides an effective, training-free mechanism for adaptively merging semantically similar regions. This mechanism also has the beneficial side effect of reducing input token count, which is shown to decrease hallucination.

### Weaknesses
1. The base model is out of data. LLaVA suffers limited performance. I suggest the authors to conduct evaluation on more SOTA models, such as InternVL-3.5 or QwenVL2.5 to truly demonstrate the effectiveness.

2. I wonder the performance on visual grounding benchmarks, such as refcoco, since it also requires fine-grained region information.

### Questions
The QtP process involves a dynamic computation (quadtree traversal and gradient calculation). Can the authors quantify the computational overhead of the QtP stage in terms of wall-clock time? How does this overhead scale with very high input resolutions? Is the total prefill time (QtP + CLIP encoding) still significantly faster than a simple, high-resolution baseline that uses linear interpolation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
CLIP is employed as the vision encoder for many MLLMs, yet its absolute positional encoding and fixed-resolution pre-training severely degrade performance when images are supplied at non-native resolutions.
The paper introduces QLIP, a drop-in replacement that can be inserted into any CLIP-based MLLM without architectural changes.  Instead of uniform patchification, QLIP first performs a content-aware partition that keeps salient regions at higher resolution and fuses the resulting variable-length token sequence into the original embedding dimension.  Extensive experiments show up to 13.6 % absolute improvement on a battery of cross-resolution benchmarks.

### Strengths
Simple, training-free at inference time, and demonstrably effective.

### Weaknesses
**Rapidly moving baseline.**  The community is already shifting from CLIP to newer vision backbones (e.g. InternVL, SigLIP-2) that use RoPE or 2-D absolute + relative encoders and are pre-trained with native multi-resolution recipes.  It is unclear whether QLIP retains any advantage when the underlying encoder itself is resolution-robust.  A head-to-head comparison with such models is missing.

**Limited to CLS-level bias.**  QLIP is optimised to deliver a single, high-quality CLS token; it does not guarantee per-token fidelity.  For dense-prediction tasks (segmentation, object detection) that require spatially accurate patch-wise features, the content-aware patchification may discard positional details and harm downstream accuracy.

### Questions
No

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper targets the limitations of the standard CLIP vision encoder used in Multimodal Large Language Models (MLLMs), specifically its fixed-resolution training and uniform patch grid1. The authors diagnose two key issues: "mesoscopic bias" (a preference for features at a specific training scale ) and "interpolation bias" (an inability to generalize positional encodings to new resolutions ). They propose QLIP, a "drop-in" replacement that combines two components: a content-aware "quadtree patchification" (QtP) mechanism to dynamically merge low-information patches based on image gradients , and a small, trained MLP to interpolate positional encodings for arbitrary coordinates. The central claim is that QLIP can be integrated into existing MLLMs like LLaVA without retraining the full model , leading to significant performance gains, most notably a +13.6% accuracy boost on the fine-grained $V^{*}$ benchmark.

### Strengths
Excellent Problem Diagnosis: The paper provides a clear and valuable conceptual framework by identifying and naming "mesoscopic bias" and "interpolation bias". This diagnosis of why CLIP fails at high resolutions is a useful contribution to the field.Strong Target-Task Performance: The +13.6% gain on the $V^{*}$ benchmark is extremely significant. 

This is a challenging benchmark designed to test the exact fine-grained failures of MLLMs, so this result strongly suggests the method is effective for its intended purpose.Token Efficiency: The quadtree mechanism (QtP) not only prunes tokens but can also improve performance with fewer tokens than the baseline. 

This is a highly desirable outcome, demonstrating a successful, content-aware pruning strategy that removes noise/redundancy rather than just information.Low-Cost Intervention (in theory): The core idea of replacing a small part of the vision encoder without retraining the entire MLLM is highly practical and computationally appealing.

### Weaknesses
1. The paper's components are not fundamentally new. Quadtrees are a classic data structure, and their application to vision transformers and dynamic tokenization has been explored. Similarly, the token pruning/merging field is already well-established. The idea of replacing static positional embeddings with a dynamic, coordinate-based MLP is also a known technique (e.g., in NeRF). The novelty lies in the combination for this specific MLLM problem, but the components themselves are iterative.

2. The method's impressive gains are not general. The +13.6% boost on $V*$ is met with a performance decrease on MM-Bench (-2.8% for 7B) and stagnation or minor losses on several other benchmarks (e.g., Sci-QA, CV-Bench). This strongly suggests the method is an overspecialization, it trades general VQA capability for fine-grained acuity. The gains do not generalize.Misleading Baselines: The paper highlights outperforming the $S^{2}$ model. However, it also cites SEAL, which achieves a much higher 75.4% on $V^{*}$ overall. While SEAL requires full retraining, this massive 17-point performance gap shows that QLIP is far from the state-of-the-art in solving fine-grained VQA, even if its method (drop-in) is cheaper.

3. "Plug-and-Play" Claim is Misleading: The method is not "plug-and-play." It introduces highly sensitive new hyperparameters, namely the image size and the quadtree pruning threshold $\alpha$. The paper states that the reported results are the "best score from our sweeps". Figure 6 clearly shows a chaotic and non-monotonic relationship between $\alpha$, image size, and performance. A user cannot simply "drop in" QLIP; they must perform an expensive, multi-axis hyperparameter sweep for their specific task to have any hope of replicating the results, which undermines the entire premise of a simple, no-cost intervention."

4. No Retraining" Claim is Misleading: The paper repeatedly claims "no re-training". However, the core MLP interpolation network must be trained. The authors trained it for 11 hours on four NVIDIA L40S GPUs. While this is vastly cheaper than retraining an MLLM, it is not "no retraining." It is a separate, required training step that introduces a dependency on a new dataset (Imagenette) and process.

### Questions
1. On Performance Inconsistency: The performance on $V*$ is excellent, but the model gets worse on MM-Bench. Do the authors have a hypothesis for this trade-off? Does the QtP's gradient-based pruning accidentally discard global, "mesoscopic" context that is necessary for general VQA benchmarks but less important for $V^{*}$?

2. On Hyperparameter Sensitivity: Given the extreme sensitivity to $\alpha$ and image size shown in Figure 6, how do you justify the "plug-and-play" claim? What concrete, general-purpose default values would you recommend to a user who cannot afford to run the "sweeps" you performed?On MLP Training: The MLP was trained on Imagenette. 

3. How confident are you that this MLP generalizes to image domains outside of natural images (e.g., medical scans, line drawings, or thermal imagery)? Does the "drop-in" claim only apply if the target domain is similar to Imagenet?

4. On Inference Overhead: What is the latency overhead of the quadtree patchification step during inference? Calculating the gradient map and recursively building the tree seems computationally non-trivial compared to a simple uniform grid patch.

### Soundness
3

### Presentation
3

### Contribution
3
