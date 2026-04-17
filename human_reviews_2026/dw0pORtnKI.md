# MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Reasoning-augmented machine learning systems have shown improved performance in various domains, including image generation. However, existing reasoning-based methods for image generation either restrict reasoning to a single modality (image or text) or rely on high-quality reasoning data for fine-tuning. To tackle these limitations, we propose MILR, a test-time method that jointly reasons over image and text in a unified latent vector space. Reasoning in MILR is performed by searching through vector representations of discrete image and text tokens. Practically, this is implemented via the policy gradient method, guided by an image quality critic.
We instantiate MILR within the unified multimodal understanding and generation (MUG) framework that natively supports language reasoning before image synthesis and thus facilitates cross-modal reasoning. The intermediate model outputs, which are to be optimized, serve as the unified latent space, enabling MILR to operate entirely at test time. We evaluate MILR on GenEval, T2I-CompBench, and WISE, achieving state-of-the-art results on all benchmarks. Notably, on knowledge-intensive WISE, MILR attains an overall score of 0.63, improving over the baseline by 80%. Our further analysis indicates that joint reasoning in the unified latent space is the key to its strong performance. Moreover, our qualitative studies reveal MILR's non-trivial ability in temporal and cultural reasoning, highlighting the efficacy of our reasoning method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces MILR, a test-time reasoning method that enhances image generation by jointly reasoning over image and text in a unified latent vector space. Utilizing a policy gradient method guided by an image quality critic, MILR operates within the Multimodal Understanding and Generation (MUG) framework, optimizing intermediate model outputs at test time without fine-tuning.

### Strengths
# Strengths:
(+) MILR achieves superior performance across all tested benchmarks, outperforming both training-based and test-time reasoning models.

(+) The method's test-time optimization avoids the need for curated reasoning data or model fine-tuning, making it cost-effective and practical.

(+) Qualitative studies highlight MILR's nontrivial abilities in geometric, temporal, and cultural reasoning, enhancing its versatility.

### Weaknesses
# Weakness:

(-) The reliance on a reward model for optimization may introduce bias, potentially limiting exploration of the generative capacity of MUG.

(-) The method's efficiency and performance depend on empirical hyperparameter settings (e.g., λt = 0.2, λv = 0.02), which may not generalize across all scenarios.

(-) The paper lacks a detailed discussion on computational costs and scalability, which could be a concern for large-scale applications. Besides, this paper claims latent test-time reasoning. I wonder about its cost (GPU Memory, Per Time, Reward Loss across time and step).

# Minor Weakness:

(-) Fig. 2 appears to have an issue, as it suggests the reward model only receives the final image, while the text tokens should also be input to assess compatibility accurately.

(-) The format of data input to the reward model is unclear; if it takes in data format like final text and final image, the computational load seems excessive—why not directly calculate the loss based on the latent vector z?

### Questions
# Questions:
1. How does MILR handle cases where the reward model provides inconsistent feedback across iterations?

2. What are the potential impacts of varying λt and λv values on different types of image generation tasks?

3. Can MILR be adapted to work with other MUG frameworks beyond Janus-Pro, and if so, what modifications would be required?

4. How does the method perform on real-time image generation tasks with strict latency constraints?

5. I read the code in the supplementary material. Are there plans to release the code to reproduce this work?


Overall, I think this work is interesting. If the author could address my concerns, I am willing to increase my rating to 8.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces MILR, a test-time optimization method for text-to-image generation that performs joint reasoning over text and image in a unified latent space. Concretely, the authors take intermediate hidden states z = [z(t); z(v)] from a unified multimodal understanding-and-generation (MUG) model (Janus-Pro; Chen et al., 2025) and iteratively update a prefix of the text and image token latents using REINFORCE-style policy gradients (Williams, 1992) to maximize a reward measuring text–image compatibility. The approach leaves model weights frozen and only optimizes latents at inference time. The paper reports strong gains on three benchmarks—GenEval (Ghosh et al., 2023), T2I-CompBench (Huang et al., 2023), and WISE (Niu et al., 2025)—and analyzes hyperparameters (prefix lengths λt≈0.2, λv≈0.02; steps up to 16) and ablations (text-only, image-only).

  Empirically, Janus-Pro-7B+MILR achieves 0.95 overall on GenEval (vs. 0.78 base Janus-Pro-7B; larger category jumps include Counting +0.34, Position +0.21, Attribute Binding +0.27), outperforms test-time strategies like PARM and ReflectionFlow (both ~0.91), and approaches training-time RL methods such as Flow-GRPO (0.95). On T2I-CompBench, MILR improves overall from 0.3921 to 0.5325; on WISE, from 0.35 to 0.63 (the paper states +80% over its base and +16.7% over T2I-R1). The authors discuss failure modes (textual/visual reasoning collapse, reward hacking) and note reliance on reward models, sometimes using the benchmark’s evaluator (“oracle” reward).

### Strengths
- Clear technical core: optimizing a prefix of multimodal token latents z(t), z(v) via policy-gradient updates at test time, without finetuning parameters. The optimization target and the forward path are precisely stated (Eq. (3)–(6)).
- Strong empirical gains across three widely used evaluations: GenEval (NeurIPS 2023; object-focused alignment), T2I-CompBench (NeurIPS 2023; compositionality), and WISE (Niu et al., 2025; knowledge-intensive prompts). Results are specific and category-level improvements are reported.
- Solid ablations: text-only vs. image-only vs. joint, token prefix ratios (λt, λv), and step counts; joint optimization consistently performs best and compute scaling is characterized (best around 16 steps).
- Practicality: method is test-time-only and uses a single A100 in reporting; no curated reasoning data required, contrasting with GRPO/DPO-style training (e.g., Flow-GRPO; Liu et al., 2025a; T2I-R1; Jiang et al., 2025).
- Thoughtful discussion of failure modes (reasoning collapse, reward hacking) with qualitative evidence; acknowledges reward-model dependence.

### Weaknesses
- Reward reliance / oracle leakage: Many claims hinge on using the benchmark’s evaluator as the reward (“OracleReward”), which risks overfitting/reward hacking and inflating benchmark scores (authors themselves show mis-evaluations in spatial relations). More evidence with non-oracle, public reward models would strengthen claims of generality beyond GenEval’s metric.
- Base-model coupling: All main results use Janus-Pro. The approach should be validated on a diffusion-based MUG (e.g., Show-o or a diffusion-tokenizer system) to test portability; current evidence is limited to one AR MUG paradigm.
- Baseline fairness and compute: Best-of-N and ReflectionFlow/PARM comparisons need tighter compute normalization. The paper mentions “comparable compute (N=T=20)” with early stopping, but wall-clock, sample counts, and variance should be reported per benchmark to rule out search budget confounds.
- Unified-latent novelty vs. prior latent/test-time reasoning: Prior latent-space test-time computation and “latent reasoning” lines (e.g., Geiping et al., 2025; Hao et al., 2024; Shen et al., 2025; Dao & Gu, 2024) are cited but the empirical isolation of “unified cross-modal latent” as the key driver remains partial. The ablation “w/o image” vs. “w/o text” helps, but an ablation that replaces unified latents with modality-specific latents in separate loops would more directly test the unified-space hypothesis.
- Reward models and robustness: MixedReward and other non-oracle critics improve over baseline but still lag the oracle. Robustness across unseen prompts/domains (outside the three benchmarks) is not demonstrated.
 - Tuning on GenEval validation: λt/λv tuned on a GenEval split (then used elsewhere) may introduce slight bias. Cross-validated tuning or tuning-once on a separate development set would reduce concerns.

### Questions
- Compute fairness: For Best-of-N, PARM, and ReflectionFlow, please report per-sample wall-clock time, total forward passes, and early-stop statistics. Are the search budgets strictly matched across methods and benchmarks?
  - Generality to diffusion MUG: Can you show MILR with a diffusion-based unified generator (e.g., Show-o or a diffusion-tokenizer model) to demonstrate portability beyond Janus-Pro?
  - Reward robustness: Can you report full results with only non-oracle rewards (e.g., MixedReward) across all three benchmarks, including failure modes and qualitative examples? Any signs of reward hacking under these critics?
  - Unified vs. separated latent loops: Could you add an ablation that optimizes text and image latents in separate spaces without sharing a unified latent layer, to isolate the benefit of the unified representation more directly?
  - Sensitivity to prefix choice: Beyond contiguous prefixes, did you try structured subsets (e.g., entropy or attention-based selection for z(v)) that may better target global-structure tokens? You mention random subsets are worse; a principled selection could help.
  - Out-of-benchmark generalization: Any tests on prompts outside the evaluator’s training (for OracleReward), or user study-aligned metrics (e.g., paired preference) to complement automated scores?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MILR, a novel method designed to improve text-to-image generation. The core contribution is a test-time optimization technique that reasons jointly over text and image representations within a unified latent space. Instead of refining raw text or image pixels, MILR operates on the intermediate latent vectors of a pre-trained MUG model. It employs a policy gradient method to iteratively update these latent vectors before they are decoded into the final output.

### Strengths
Novel and elegant proposal for test-time latent reasoning, which enhances powerful pre-trained models without requiring any fine-tuning.  

Impressive empirical results across multiple challenging benchmarks.

### Weaknesses
The term "reasoning" is debatable here. The process is more accurately described as a guided latent space search that optimizes a latent vector to maximize a reward, rather than a structured, logical thought process (like reasoning defination in LLM).

The most significant flaw is that the headline, state-of-the-art results reported in Tables 1 and 2 are achieved using the benchmark's own evaluation toolkit as the reward model. This  setup is not representative of any realistic application. In practice, a perfect reward function is never available.

### Questions
Could you elaborate on the choice of REINFORCE as the optimization algorithm and the choice of the pre-trained model?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce MILR, a novel method for improving multimodal image generation during test time. In this approach, they perform reasoning in latent space jointly over unified image and text. The use REINFORCE method to optimize the intermediate latent vectors, with the reward model that assesses how well the generated image aligns with the input.
They show that MILR significantly improves the performance of Janus-Pro models on several benchmarks, achieving state of the art results.

### Strengths
- Their approach of performing joint reasoning in latent space is very interesting and novel, as well as, achieves state of the art performance.
- The paper is well written, and the experiments are thorough.

### Weaknesses
- The authors did not discuss about latency. Given that their technique involves iterative optimization at test time, there will be significant increase in latency.
- This method is heavily dependent on the reward function. There is a possibility of reward hacking, as authors also mentioned. One way to resolve it would be to add a regularizer like penalize semantic incoherence.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4
