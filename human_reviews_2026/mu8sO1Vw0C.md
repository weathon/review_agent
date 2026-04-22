# Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Recent advances in video generation have enabled thrilling experiences in producing realistic videos driven by scalable diffusion transformers. However, they usually fail to produce satisfactory outputs that are aligned to users' authentic demands and preferences. In this work, we introduce Dual-Iterative Optimization (Dual-IPO), an iterative paradigm that sequentially optimizes both the reward model and the video generation model for improved synthesis quality and human preference alignment. For the reward model, our framework ensures reliable and robust reward signals via CoT-guided reasoning, voting-based self-consistency, and preference certainty estimation. Given this, we optimize video foundation models with guidance of signals from reward model's feedback, thus improving the synthesis quality in subject consistency, motion smoothness and aesthetic quality, etc. The reward model and video generation model complement each other and are progressively improved in the multi-round iteration, without requiring tediously manual preference annotations. Comprehensive experiments demonstrate that the proposed Dual-IPO can effectively and consistently improve the video generation quality of base model with various architectures and sizes, even help a model with only 2B parameters surpass a 5B one. Moreover, our analysis experiments and ablation studies identify the rational of our systematic design and the efficacy of each component.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Dual-Iterative Preference Optimization (Dual-IPO), a framework that jointly and repeatedly improves both the reward model and the video generation model to better align diffusion-based T2V models with human preferences. The reward model is strengthened in each round via Chain-of-Thought guided reasoning, voting-based self-consistency, and a preference-certainty estimator to produce reliable pseudo-labels without extensive manual annotation. Using these refined signals, the video generator is iteratively optimized via DPO/KTO-style preference learning, and both models co-evolve over multiple rounds. Experiments across architectures and model sizes show consistent gains in subject consistency, motion smoothness, and aesthetics, including cases where a 2B model surpasses a 5B baseline.

### Strengths
- Dual-IPO attains a level of preference alignment that was not achievable with prior approaches. It is validated on state-of-the-art backbones (CogVideoX, Wan 2.1), and even a small model is shown to surpass the performance of a larger one in Table 5.
- The effectiveness of the constructed reward models is discussed, particularly centered around Table 1.
- Human evaluation results (e.g., Fig. 3b) are also included for complementary aspects, demonstrating that the improvements are not merely due to over-optimization against the evaluation model.

### Weaknesses
- The discussion on the number of iterative rounds is insufficient. It is unclear whether increasing the number of rounds continues to improve performance or saturates, and therefore, it is difficult to claim that an optimal recipe has been established.

- For the experiments on Wan, not all VBench metrics are reported. Since VBench is biased toward temporal consistency, it remains possible that the training procedure reduces temporal dynamics in videos. This should be explicitly addressed. The trade-off between temporal consistency and dynamics has been discussed in prior work (e.g., [1], [2]).

- Human Preferences are not shown for Wam 2.1 experiments. Is the performance gain for Wan 2.1 and more powerful models not efficient enough? To address this concern, I recommend that you show the results of human preference experiments for Wan 2.1 1.3B/14B.

[1] Liu, et al. VideoDPO: Omni-Preference Alignment for Video Diffusion Generation. CVPR2025.  
[2] Oshima, et al. Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search. NeurIPS2025.

### Questions
Please see the weakness for the major questions.

### Soundness
3

### Presentation
3

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
This paper presents Dual Iterative Preference Optimization (Dual-IPO), a novel post-training framework designed to enhance T2V generation models through iterative optimization of both the reward model and the video generation model. Unlike conventional alignment approaches such as DPO and KTO, which rely on fixed, human-annotated datasets, Dual-IPO introduces a bi-directional training loop. On the one hand, the reward model is iteratively refined via SRPO using CoT-guided reasoning, multi-path self-consistency voting, and a preference certainty estimator. The T2V model is updated based on dynamically evolving reward signals through Diffusion-DPO or Diffusion-KTO optimization. Experiments on CogVideoX and Wan show significant improvements in semantic alignment, motion smoothness, and aesthetic quality. The proposed framework achieves 81.33% human preference accuracy and strong VBench gains, establishing new SOTA results.

### Strengths
1.	The paper provides a motivation for addressing static reward limitations and distribution mismatch in preference alignment for generative video models.
2.	The proposed Dual-IPO is a novel optimization paradigm. The idea of jointly optimizing the reward model and generator in a feedback loop is interesting and novel. It addresses reward drift and distribution mismatch issues, which are common in previous methods like DPO or RLHF. It also reduce the requirements on large-scale human annotations.
3.	This paper proposes SRPO to train the reward model, which integrates CoT-guided reasoning, multi-path voting, PCE and SRPO loss. It supports both pairwise and pointwise supervision, which is flexible and unified for various alignment tasks.
4.	Dual-IPO achieves new state-of-of-art performance. Experiments on CogVideoX (2B/5B) and Wan (1.3B) show consistent gains (e.g., +3.0 on VBench), demonstrating robustness across architectures and parameter scales. The experiment validation is comprehensive and rigorous.

### Weaknesses
1.	The paper does not analyze the convergence of the dual-iterative process. There is no proof that the process will stably converge stably to an optimal point, rather than oscillating. The paper also does not clearly state when the iterative process should stop.
2.	Lack of ablation study. For example, the paper does not contain the complete ablation study of the three key parts of SRPO (CoT, self-consistency and PCE).
3.	The method is complicated and inefficient, which is pointed out by the authors. It requires 128 GPUs and approximately two weeks per iteration, making it extremely resource-intensive. The paper does not analyze the efficiency of each component, or explore the strategy to reduce computational cost.
4.	It would be better and clearer to illustrate SRPO and its components with graphs.
5.	Some typos (eg. in Figure 1 (c)).

### Questions
1.	What is the convergence criteria of the dual-iterative preference optimization? Will the excessive iterations cause overfitting? Is there a risk of reward hacking or mode collapse in later iterations?
2.	What is the bottleneck that restricts the computational efficiency? Have you explored any strategies to reduce computational requirements? How does the performance scale with reduced computational budgets?
3.	Could you provide the ablation study of CoT, self-consistency and PCE to prove their effectiveness individually? And what if replace SRPO with static DPO?
4.	Could Dual-IPO be generalized beyond T2V? The design seems extensible.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Dual-Iterative Preference Optimization (Dual-IPO), a post-training framework that iteratively refines both a reward model and a text-to-video (T2V) generator to better align video synthesis with human preferences.
The method combines:

- Self-Refined Preference Optimization (SRPO) — leveraging Chain-of-Thought (CoT) guided reasoning, self-consistency voting, and a Preference Certainty Estimator (PCE) for robust pseudo-labeling.

- Dual iterative optimization — alternately improving the critic (reward) model and the video generation model via both Diffusion-DPO and Diffusion-KTO objectives.
Experiments on CogVideoX, Wan, and VBench benchmarks show consistent gains in motion smoothness, aesthetic quality, and prompt consistency, with a 2B model surpassing a 5B baseline.

### Strengths
- Dual optimization design – The interplay between a self-refined reward model and iterative generator updates is conceptually elegant and empirically validated.

- Methodological soundness – The CoT-guided pseudo-labeling and PCE-weighted DPO/KTO training are carefully formulated and ablated.

- Strong experimental validation – Includes results across model sizes, architectures, and both automatic and human evaluations, showing consistent improvement.

- Clarity and completeness – Writing, figures, and methodology are clear; the paper includes reproducibility and ethics statements, and provides rich supplementary material.

### Weaknesses
- High computational cost – Each optimization round involves dual training of large models (VILA-40B, CogVideo-5B), making it unclear how scalable or practical Dual-IPO is for typical research labs.

- Dependence on synthetic preference labels – Despite PCE filtering, the pseudo-label quality may still drift; more human validation or robustness analysis would strengthen the claims.

### Questions
- How sensitive is Dual-IPO to the initial human preference dataset size and quality? Would small or noisy seeds destabilize the dual-loop training?

- Do you observe any reward overfitting or “reward hacking” effects in later iterations? If so, how does SRPO mitigate them?

- Could the authors provide a more quantitative analysis of the trade-off between computational cost and performance gains? Specifically, each iterative round seems to require large-scale training with multiple pseudo-labeling and re-optimization stages. How does performance improvement scale with training cost (e.g., GPU hours vs. VBench gain)? Is there an observed saturation point after which additional iterations bring diminishing returns?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Dual-IPO (Dual-Iterative Preference Optimization), a novel post-training framework for text-to-video (T2V) generation that enhances both synthesis quality and alignment with human preferences. The method involves a dual iterative paradigm that sequentially optimizes a reward model and a video generation model. The reward sign is refined through Chain-of-Thought (CoT) guided reasoning, voting-based self-consistency, and a preference certainty estimator to ensure reliable feedback. Using these improved reward signals, the video generation model is iteratively updated to enhance semantic consistency, motion smoothness, and aesthetic quality. Through multiple optimization rounds, both models are progressively improved without the need for extensive manual preference annotations. Experimental results show that Dual-IPO substantially improves video generation quality and allows smaller models to outperform larger baselines.

### Strengths
1) The authors introduce a comprehensive and well-motivated dual-iterative framework that jointly optimizes the reward and generation models, addressing limitations of static preference alignment methods.

2) The proposed approach demonstrates strong data efficiency, requiring only a small amount of human-annotated preference data to initiate the self-refinement process.

3) The paper effectively captures the evolving nature of human preferences, emphasizing that fixed offline datasets may lead to overfitting and reduced generalization.

4) Extensive experiments across multiple model architectures and scales (e.g., CogVideoX and Wan) are provided, supported by quantitative metrics (VBench) and qualitative analysis.

### Weaknesses
1) The construction of textual prompts used for generating training data is under-specified. The authors mention the use of structured elements (subjects, attributes, spatial relations, and actions), but the paper would benefit from more details regarding the **size, diversity, and balance of the prompt pool**. For instance, how many combinations were used per category, and how does this affect diversity and representativeness?

2) The construction of textual prompts may inadvertently introduce **bias** into the generated training data. Since prompts reflect the authors’ design choices, such as the selection of subjects, attributes, and actions. They could encode cultural, demographic, or semantic biases present in the prompt corpus. For example, certain object, action combinations might be overrepresented, or specific demographic groups may be underrepresented if not explicitly balanced. Such biases could propagate through the downstream model, affecting its generalization and fairness. The paper would benefit from a discussion of how prompt templates were curated, whether any bias detection or mitigation strategies (e.g., prompt balancing, debiasing filters, or controlled generation) were applied, and how residual biases were evaluated or monitored.

### Questions
My main concern regarding this paper is the diversity of the user-specified training data construction and the potential bias led by this procedure. If the authors could address my concern effectively. I'd be happy to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3
