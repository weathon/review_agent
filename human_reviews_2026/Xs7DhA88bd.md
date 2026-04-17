# Synergistic Absorption-Diffusion: Dual-branch Enhanced Continuous-Time Modeling for Parallel Token Generation

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recent advancements in diffusion models, such as global optimization and parallel token prediction, have enhanced global consistency compared to autoregressive Transformers. However, existing diffusion models exhibit unfavorable trade-offs between efficiency and quality, in which the multi-step iterative denoising processes particularly incur high computational costs. To address these issues, we propose a dual-branch synergistic absorption diffusion model. For efficiency-quality trade-offs, we design a dual-branch architecture, in which the Transformer branch generates local token chunks, and the diffusion branch optimizes global token blocks in fewer steps. To resolve the instability of discrete-time models, we further introduce the continuous-time diffusion process, which enhances parallel token generation and learning representations. Experiments conducted on multiple tasks, including text generation and structural reasoning tasks, demonstrate the state-of-the-art performance of the proposed model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a dual-branch continuous-time discrete diffusion model (CAD-DF) combining a Transformer autoregressive (AR) branch with a diffusion denoising branch. The goal is to achieve both global consistency (from diffusion) and local coherence (from AR) for parallel token generation. The method introduces a cross-attention fusion mechanism and mutual reinforcement between branches, along with denoising cross-entropy (t-DCE, λ-DCE) losses for stable training. Results show improved perplexity, accuracy, and inference speed on several datasets compared to GPT-2 and recent diffusion-based baselines.

### Strengths
1. Conceptually elegant integration of AR and diffusion paradigms.
2. Continuous-time formulation improves efficiency and stability.
3. Strong experimental results across both small and medium model scales.
4. Clear ablation studies showing the effect of each component.
5. Well-motivated and theoretically consistent with prior works.

### Weaknesses
1. Presentation is heavy and could be simplified for accessibility.
2. Theoretical explanation of mutual reinforcement could be supported with empirical visualization (e.g., information flow or convergence curves).
3. Missing analysis on failure or degradation cases.

### Questions
1. How does the dual-branch mechanism behave when scaling to very large models—do benefits persist or saturate?
2. Can the cross-attention fusion be replaced or approximated for efficiency at scale?
3. How sensitive are results to the λ/t-DCE hyperparameters?
4. Are there any stability trade-offs in the continuous-time inference regime?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a dual-branch synergistic absorption diffusion model that integrates a Transformer branch and a diffusion branch via cross-attention to address the efficiency-quality trade-off in parallel token generation.

### Strengths
- The dual-branch structure that synergistically integrates AR and diffusion paradigms via cross-attention is novel and well-motivated.   
- The model demonstrates state-of-the-art performance on quality metrics (Perplexity, Accuracy) across diverse text generation and structural reasoning tasks.

### Weaknesses
- The model is said to effectively address the efficiency-quality trade-off, but Table 4 shows the full model has lower throughput than the model "w/o Transformer Branch".  
- The novelty of the paper is not well described; some key concepts like time-independent reparameterization and DCE losses are adopted directly from prior work, but they are not well distinguished.
- Presentation quality is poor.
  - Figure 1 shows U-Net diffuser, but there is no explanation.
  - There are no explicit references to any tables in the main text. 
  - The citation style can be improved.

### Questions
- What do the authors mean by saying effectly addressing the efficiency-quality trade-offs? 
- What exactly is the contributions/novelties of the approach compared to existing models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a dual-branch synergistic architecture for parallel token generation in language models, termed the Synergistic Absorbing Diffusion (CAD-DF) framework. It integrates a Transformer-based autoregressive branch with a continuous-time discrete diffusion branch, coordinating them via cross-attention fusion and mutual reinforcement mechanisms. The approach is designed to address longstanding tradeoffs between generation efficiency and quality, leveraging analytical time-independent objectives and absorbing state diffusion for more robust and scalable sequence modeling. Experiments on six benchmark datasets across language modeling and structural reasoning tasks reportedly demonstrate superior performance in perplexity, accuracy, and inference efficiency compared to both autoregressive and diffusion-based baselines.

### Strengths
1. The paper addresses an important and active research problem—how to achieve both efficient and high-quality parallel token generation—by explicitly combining the strengths of autoregressive and diffusion models. The dual-branch framework is presented as a meaningful improvement over simple mixtures or sequential composition.
2. Extensive experiments on diverse, standard benchmarks (WikiText2, WikiText103, C4, FineWeb, Prolong, JSON-Mode-Eval) show substantial improvements. Table 1 and Table 2 (Pages 8) detail strong gains in both perplexity and accuracy compared to competitive baselines, including GPT-2, D3PM, SEDD, PLAID, and the RADD series.
3. The efficiency analysis in Table 3 (Page 9) demonstrates notable improvements in throughput and memory, while Table 4 (Page 9) provides ablations showing each major architectural component’s impact.

### Weaknesses
1. Lack of Baselines: While the experimental section is broad, there is no evidence of direct empirical comparison against block diffusion, ParaTAA, or speculative sampling frameworks, whose aims and underlying architectures are similar or overlapping with CAD-DF. The absence of these comparisons in Tables 1 and 2 limits the credibility of the “state-of-the-art” claims.
2. Mathematical Details and Notational Ambiguity in Cross-Attention Fusion: The description of the cross-attention fusion mechanism (see Equation, Page 6) lacks formal definitions for input/output dimensionalities, masking strategies, and how [M]-masked positions interact during fusion. Additionally, it is not entirely clear how (or whether) stability holds when merging outputs from two distinct temporal streams, especially if their distributions are not inherently aligned. This raises questions about the reliability of the mutual reinforcement during training.
3. Limited Ablation Depth and Lack of Qualitative Error Analyses: While Table 4 (Page 9) gives a simple ablation, there is no exploration of nuanced failure cases, qualitative generations, or breakdowns across context length and sequence complexity. For a method purporting to resolve global-local consistency, analyses showing (or failing) on long-range sequence dependencies or structural errors would be instructive.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Synergistic Absorbing Diffusion, a dual-branch architecture combining a continuous-time diffusion branch (for global distribution modeling) and a Transformer branch (for local conditional modeling), integrated via cross-attention. This design improves parallel generation, consistency, and efficiency, achieving state-of-the-art results on several NLP tasks.

### Strengths
1. The proposed Synergistic Absorbing Diffusion model not only enhances parallel generation capability and global consistency but also reduces the number of denoising steps and inference latency. This is of significant importance for efficient processing in practical application scenarios.

2. The authors conducted comprehensive experiments covering various natural language processing tasks, such as text generation and structured reasoning. The experimental results indicate that the Synergistic Absorbing Diffusion model achieves strong performance on these tasks.

### Weaknesses
1. When scaling up to medium-sized models, the method's advantage in Prolong PPL does not appear to be significant. It is uncertain whether it can be effectively scaled up to modern Large Language Models (LLMs).

2. The model has not been evaluated on downstream tasks.

### Questions
NA.

### Soundness
2

### Presentation
2

### Contribution
2
