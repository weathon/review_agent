# Unified Vision-Language-Action Model

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Vision-language-action models (VLAs) have garnered significant attention for their potential in advancing robotic manipulation.
However, previous approaches predominantly rely on the general comprehension capabilities of vision-language models (VLMs) to generate action signals, often overlooking the rich temporal and causal structure embedded in visual observations. In this paper, we present UniVLA, a unified and native multimodal VLA model that autoregressively models vision, language, and action signals as discrete token sequences. This tokenized formulation naturally supports flexible multimodal task learning, particularly from large-scale video data, and further demonstrates that generative vision supervision can significantly enhance visual understanding. By incorporating world modeling during post-training, UniVLA captures causal dynamics from videos, facilitating effective transfer to downstream policy learning—especially for long-horizon tasks. Our approach sets new state-of-the-art results across several widely used simulation benchmarks, including CALVIN, LIBERO, and Simplenv-Bridge, substantially outperforming prior methods. For example, UniVLA achieves 95.5% average success rate on LIBERO benchmark, surpassing π₀-FAST's 85.5%. We further demonstrate its broad applicability through experiments on real-world ALOHA manipulation tasks and autonomous driving scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper “UniVLA: Unified Vision-Language-Action Model” introduces a fully tokenized, autoregressive architecture that jointly models vision, language, and action within a single transformer-based sequence framework. Unlike conventional VLA systems that fuse pretrained visual encoders with language-conditioned action heads, UniVLA treats all three modalities as discrete tokens within a shared vocabulary, enabling unified cross-modal reasoning and flexible task conditioning.
The framework operates in two stages: (1) Post-training (world modeling): learning causal video dynamics and temporal prediction from large-scale robot-centric videos without action supervision. (2) Fine-tuning (policy learning): conditioning on interleaved visual and textual tokens to predict action tokens.

The approach achieves state-of-the-art results across CALVIN (4.63 average length vs. 4.49 for RoboVLMs), LIBERO (95.5% vs. 85.5% for π₀-FAST), and SimplerEnv (69.8% vs. 42.7%). It also transfers to real-world ALOHA robotic tasks and autonomous driving (NAVSIM), showing versatility of the unified token-based modeling.

### Strengths
- UniVLA’s key novelty lies in representing vision, language, and action as discrete tokens under a shared autoregressive framework. This eliminates separate encoders and heads, resulting in a conceptually elegant, flexible design. The framework’s success on both robotics and driving underscores its generality.

- The paper demonstrates that pretraining on large-scale video prediction (without action labels) significantly enhances downstream policy efficiency and long-horizon reasoning. Table 4’s analysis is particularly convincing, showing +45–70% relative improvement over action-only post-training.

- UniVLA consistently outperforms prior VLA baselines (π₀, FAST, OpenVLA, GR-1, RoboVLMs) across three simulation suites and two real-world domains. The inclusion of both robotic manipulation and driving tasks makes the evaluation unusually broad for a single model.

- The paper systematically analyzes data efficiency (Table 5a), training efficiency (Table 5b), world model variants, and historical context. The insights—for instance, diminishing returns beyond 2-step history—are practical for future VLA design.

- Demonstrating performance gains on the physical ALOHA robot (pouring, folding) and quantifying inference latency (≈3s per 20-step chunk) is commendable. Few VLA papers provide such detailed real-world evaluation.

### Weaknesses
- The model borrows heavily from existing architectures (Emu3 for transformer backbone, FAST for action tokenization, VQ-VAE for image tokens) and primarily unifies them under one autoregressive umbrella. While impactful, the approach feels more like a pragmatic consolidation than a paradigm shift.

- The paper includes numerous benchmarks, ablations, and cross-domain studies—sometimes at the expense of clarity. A concise ablation visualization or conceptual diagram summarizing findings could make the narrative more digestible.

- The authors claim that token unification enables stronger “causal reasoning,” but this is not analytically substantiated. There’s little formal justification for why autoregressive modeling over discrete tokens inherently captures causal structure better than multimodal fusion.

- The method’s scalability depends on massive video post-training (622k videos, 32×A100 GPUs for 5 days). The claim of “data efficiency” holds mainly at fine-tuning, not at pretraining. Real-world feasibility for smaller labs remains questionable.

- Although the paper reports strong quantitative results, it does not deeply analyze potential loss of control granularity introduced by discrete action tokenization, especially for fine-grained manipulation tasks (e.g., folding clothes).

### Questions
- How sensitive is UniVLA to the specific choice of discretization (VQ encoder, DCT-based FAST)? Could learned continuous latent codes outperform these fixed tokenizers?

- During post-training, how much of the improvement stems from video scale vs. the world model objective itself?

- Can UniVLA generalize to reinforcement learning settings where rewards guide action prediction rather than imitation?

- How does token-level autoregression scale with longer action horizons (e.g., >1,000 tokens per rollout)?

- Is there any observed degradation when transferring from simulated to real-world data, especially under domain shifts in lighting and textures?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel architecture, UniVLA, which encodes three modalities—image, text, and action—into discrete tokens. This enables joint modeling in an autoregressive manner, achieving better cross-modal alignment. Based on this multi-modal discrete token autoregressive paradigm, UniVLA can unify the world model training and visuomotor policy within a single model. It also demonstrates that world model training improves policy learning, especially in long-horizon tasks and OOD scenarios. The authors achieved promising results on classic VLA benchmarks, including CALVIN, LIBERO, and Simpler-Env, demonstrating the effectiveness of their method.

### Strengths
- The UniVLA architecture is meaningful: it discretizes images, text, and actions into tokens and trains them under a unified autoregressive paradigm. This enables stronger cross-modal alignment via diverse proxy tasks, such as world-model training, policy learning, and multimodal understanding.
- The authors demonstrate that world-model training is an effective auxiliary task that benefits downstream policy learning.
- UniVLA achieves competitive results on several standard benchmarks, showing improvements over some classic VLA models.

### Weaknesses
- Concern about inference speed. The authors use Emu3-8.5B as the autoregressive base, which is non-trivial in size for a VLA backbone. Moreover, action prediction requires generating a sequence of tokens, which inevitably slows inference. This is especially problematic in real-robot experiments, where limited on-device compute leads to a low action-prediction rate. The authors should further discuss UniVLA's deployment in simulation and on real hardware, its baseline compute requirements, and the concrete inference frequency.
- Limitations of real-world experiments. As a VLA model, its performance in the real world is a crucial metric of its quality. However, the experimental section only outlines the basic setup of the real-robot evaluation in L454–457, without reporting UniVLA’s success rate on real tasks or comparing its real-world performance against other competitive VLA models. This omission hinders a fuller assessment of UniVLA’s contributions.
- The authors should discuss how much of the current performance gain stems from Emu3's inherent multimodal discrete-token modeling. While we acknowledge that modeling vision, language, and action uniformly as discrete tokens is a reasonable approach, would the performance remain comparable if this paradigm were transplanted to other backbones? Notably, OpenVLA and Pi-FAST were not pretrained with such multimodal discrete-tokenization prediction.

### Questions
- Have the authors analyzed and compared different action encoding strategies? Within the autoregressive action-prediction paradigm, there are many ways to encode discrete tokens (e.g., FAST, VQ-VAE), and one could also adopt continuous latent encodings.
- Did the authors incorporate multimodal understanding tasks within this multimodal discrete-token modeling paradigm? In the leftmost panel of Figure 1, we notice an illustration of a multimodal grounding task, yet Section 3.2 does not describe the corresponding training procedure.
- Further discussion on architecture. Modeling discrete tokens across the three modalities is meaningful for aligning them, but it may be insufficient for VLA. For example, in PI-0.5, the first training stage jointly models vision, language, and action (with images not discretized and actions discretized via FAST), yet the second stage still introduces an action expert to produce actions, ensuring precise action modeling. Have the authors examined the difference between predicting actions solely with a VLM and adopting a dual-system–style architecture?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors aim to demonstrate that existing VLA methods with language-centric paradigm have difficulty in learning of cross-modal represemtations, temporal and causal dependencies across the perception-action loop. The authors introduce UniVLA, which jointly models vision, language, and action through autoregressive sequence learning. The effectiveness of the method is validated through experiments in both simulation and real-world settings.

### Strengths
1. This paper proposes a novel paradigm-UniVLA-that models all three modalities as discrete tokens within a shared autoregressive sequence. This represents a significant conceptual advance beyond traditional late-fusion or modality-specific architectures, enabling deeper cross-modal interaction and joint representation learning.

2. The proposed world-model post-training via video-based supervision substantially enhances temporal reasoning and data efficiency, demonstrating a clear methodological contribution.

3. The model achieves state-of-the-art performance across multiple simulation benchmarks and demonstrates strong generalization to long-horizon tasks.

### Weaknesses
1. Although the simulation results are impressive, the study lacks a comparison against established baseline models on real-world tasks. Robustness under noisy sensory inputs remain insufficiently validated. 

2. Why does employing action prediction within this framework during post-training adversely affect model performance?

3. The 8.5B-parameter model exhibits significant latency in real-world deployment (evidenced by robotic arm stuttering in the video), critically impairing its capacity for real-time temporal information processing. Can this framework be successfully applied to other VLM with fewer parameters?

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a unified Vision-Language-Action (VLA) model that discretizes all modality data into tokens. Unlike previous VLAs that predict actions solely from current observations, this work predicts the next actions conditioned on interleaved visual-action histories. Furthermore, the paper introduces a world-model post-training strategy to effectively transfer knowledge from pretrained multimodal models to downstream robotic tasks. Extensive experiments across multiple simulation environments validate the effectiveness of the proposed approach.

### Strengths
1. The paper presents a unified VLA framework that discretizes all modalities into tokens, resulting in a shared representation across visual, textual, and action modalities.
2. Comprehensive experiments are conducted in both simulated and real-world scenarios, validating the effectiveness of the proposed framework and analyzing the transferability of different visual post-training strategies.
3. The model is further evaluated beyond robotic manipulation, including autonomous driving tasks, demonstrating the generalization ability of the proposed approach.

### Weaknesses
1. Inference speed – The model is built upon Emu3 with 8.5B parameters and employs next-token prediction for action generation. This results in slow inference and limits its ability to handle tasks requiring high responsiveness. The authors are encouraged to provide detailed inference speed comparisons with other VLAs such as OpenVLA, π₀, and UVA.
2. While the paper presents a strong discrete-token–based autoregressive framework, it lacks systematic comparisons or discussions with continuous-action (e.g., π₀, DexVLA[1]) or hybrid VLA models (e.g., HybridVLA[2]). Continuous or hybrid approaches often achieve smoother control and better precision, and an in-depth analysis of these trade-offs would strengthen the paper.
3. The proposed UniVLA predicts the next chunk of actions conditioned on interleaved observation and state history, while most baseline VLAs rely only on current observations. This may create an unfair comparison. The authors should include stronger baselines that also utilize history information (e.g., Long-VLA[3]) to ensure fair comparison.


[1]. DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control
[2]. HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model
[3]. Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation

### Questions
1. Although the proposed world-model post-training strategy appears effective, it requires 50K training steps on 32 A100 GPUs over 4–5 days. Could the authors conduct additional ablation studies to analyze how this stage affects downstream performance—for instance, by reducing the size of the post-training dataset and evaluating the resulting impact?

2. In Appendix D, the authors state: “This design enables policy learning for embodied control and spatial reasoning through language output.” Could the authors further clarify how UniVLA performs spatial reasoning tasks without being explicitly trained on such datasets? Is this ability inherited from the source EMU3 model after two-stage training, or were spatial reasoning datasets incorporated during post-training?

### Soundness
3

### Presentation
3

### Contribution
3
