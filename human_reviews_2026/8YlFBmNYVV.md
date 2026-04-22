# Round-trip Reinforcement Learning: Self-Consistent Training for Better Chemical LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Large Language Models (LLMs) are emerging as versatile foundation models for computational chemistry, handling bidirectional tasks like reaction prediction and retrosynthesis. However, these models often lack round-trip consistency. For instance, a state-of-the-art chemical LLM may successfully caption a molecule, yet be unable to accurately reconstruct the original structure from its own generated text. This inconsistency suggests that models are learning unidirectional memorization rather than flexible mastery. Indeed, recent work has demonstrated a strong correlation between a model's round-trip consistency and its performance on the primary tasks. This strong correlation reframes consistency into a direct target for model improvement. We therefore introduce Round-Trip Reinforcement Learning (RTRL), a novel framework that trains a model to improve its consistency by using the success of a round-trip transformation as a reward signal. We further propose an iterative variant where forward and reverse mappings alternately train each other in a self-improvement loop, a process that is highly data-efficient and notably effective with the massive amount of unlabelled data common in chemistry. Experiments demonstrate that RTRL significantly \textbf{boosts performance and consistency} over strong baselines across supervised, self-supervised, and synthetic data regimes. This work shows that round-trip consistency is not just a desirable property but a trainable objective, offering a new path toward more robust and reliable foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces **Round-Trip Reinforcement Learning (RTRL)**, a training framework designed to explicitly improve **round-trip consistency (RTC)** in chemical LLMs.
The authors observe that existing chemical LLMs perform well in one direction (e.g., for reaction prediction or retrosynthesis) but fail to reproduce the original input when performing the inverse mapping (e.g., reaction prediction and then retrosynthesis to get the raw input).

To address this, the authors propose to treat RTC not just as an evaluation metric but as a trainable RL objective. In RTRL, the model generates a forward output and is rewarded based on how well a backward model can reconstruct the original input. They employ GRPO for stable RL optimization and further introduce an iterative self-improvement loop where forward and backward tasks train each other alternately. Experiments on round-trip chemical tasks (e.g., **reaction prediction/retrosynthesis and molecule captioning/generation**) show up to ~52% gain in round-trip consistency and ~55% improvement in primary task metrics.

### Strengths
This paper focuses on some of the most important tasks in molecule-text modeling and achieves obvious performance improvements through the following advantages:
- **Novel training objective with domain grounding:** The paper innovatively transforms round-trip consistency from a diagnostic metric into a direct optimization target. This aligns well with the inherently bidirectional nature of many chemical phenomena.
- **Elegant self-improvement design:** The iterative RTRL loop enables mutual reinforcement between forward and backward mappings, leading to better bidirectional performance without relying solely on paired data.
- **Good writing quality and clarity:** The paper is logically structured, well written, and progresses in a clear, step-by-step manner—making complex RL-based ideas accessible to a broader audience.

### Weaknesses
Below are my concerns about this paper. My major concerns are about weaknesses 3 and 4: about the improvement of performances on forward and backward tasks (instead of the RTC ones). I would be happy to raise my score if the author can provide satisfactory response.
- **Framework figure complexity:** The framework figure in the paper (Figure 1) is visually cluttered and difficult to parse. Its layout and dense notation make it hard for readers to quickly understand the overall workflow and the interaction between the forward model, backward model, and RL loop.
- **Potential for reward hacking:** As with any RL setup, the model might exploit trivial output patterns that are easy to invert but chemically meaningless. For example, what if the forward LLM includes the molecular SMILES in some corner when generating its caption? This would make the backward task super easy.
- **Mixed experimental outcomes:** While consistency metrics improve significantly, the actual results do not hold across all tasks (e.g., some relatively low or even negative improvements in Tables 2 and 3) and are not particularly strong compared to baselines. This may limit enthusiasm for immediate practical adoption.
- **Weak baselines:** The baseline models included in this research are either general LLMs (GPT) or relatively old—the latest model is ChemDFM, which was proposed in Jan 2024. Comparing with stronger or more influential baseline models (e.g., Mol-LLM, InstructMol-GS, MolecularGPT, ReactXT) could better strengthen the findings in this paper.

[1] Mol-LLM: Multimodal generalist molecular LLM with improved graph utilization  
[2] MolecularGPT: Open large language model for few-shot molecular property prediction  
[3] InstructMol: Multi-modal integration for building a versatile and reliable molecular assistant in drug discovery, COLING 2025  
[4] ReactXT: Understanding Molecular “Reaction-ship” via Reaction-Contextualized Molecule-Text Pretraining, ACL 2024

### Questions
I'm interested in more discussion regarding the reward hacking phenomenon. As mentioned in the Weaknesses section, I don't think a single format reward calculated by RDKit can eliminate this problem in such an iterative self-improving setting.

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
3

### Summary
This paper addresses the lack of "Round-Trip Consistency" (RTC) in Large Language Models (LLMs) in chemistry when handling bidirectional tasks (such as reaction prediction and retrosynthesis, molecular description and generation). A novel training framework called "Round-Trip Reinforcement Learning" (RTRL) is proposed. The authors observe that existing models may successfully map from A to B, but fail to accurately reconstruct A from B, indicating that the model learns a unidirectional, shallow memory rather than a flexible understanding of chemical principles.

The core idea of ​​RTRL is to treat RTC as a trainable objective, using the success or failure of the ring transition as a reward signal, and optimizing the model through reinforcement learning (specifically the GRPO algorithm). The authors design an efficient reward function, using the conditional likelihood $p(x|y)$ of the original input $x$ at the intermediate output $y$ as the surrogate reward, avoiding expensive double generation and complex similarity metrics.

Furthermore, the framework supports Iterative RTRL, enabling forward and backward models (both powered by the same LLM) to mutually reinforce each other. This approach demonstrates effectiveness across various scenarios, including self-supervised (using only unlabeled data), supervised, and synthetic data. Experimental results show that RTRL significantly improves the model's RTC (up to 52%) and main task performance (up to 55%).

### Strengths
1. Conceptual Novelty. The most significant contribution of this paper lies in its novel perspective: transforming "Round-Trip Consistency" from a traditional evaluation metric into a directly optimizable training objective.

2. Efficiency. By using conditional log-likelihood $\log p_{\phi}(x|y, t_g)$ as a reward, this method (1) avoids the complete backward generation step, greatly improving computational efficiency; (2) by using LLM itself as a "referee", it bypasses the difficulty of designing effective text similarity measures (such as BLEU) in the field of chemistry (such as SMILES).

3. Comprehensive Evaluation. The authors conducted a series of well-designed experiments that strongly support their core arguments. Experiments cover two core bidirectional tasks: reaction prediction (retrosynthesis) and molecular description (text-based molecule generation).

### Weaknesses
1. Need more discussion on Reward Hacking. The authors mention the possibility of "reward hacking," where a model might learn to copy the input ($A \rightarrow A$) to easily obtain a high reward of $p(x|y)$. Authors propose a solution by introducing an optional "format score" $F(y)$ (Eq. 10). This solution is somewhat ad-hoc; the Authors do not discuss the sensitivity of the hyperparameter $\alpha$ in detail, nor do they explore whether such a simple format check is sufficient to prevent more sophisticated "cheating" behavior.

2. Stability and Scalability Concerns. This method relies on Reinforcement Learning (GRPO), and the training stability of applying RL to LLM is a well-known challenge. The authors mention that they had to freeze the parameters of the reward model $\phi$ to stabilize training, implying the sensitivity of the training process. Experiments were conducted on an 8B model. While the authors speculate that RTRL can be applied to larger models, this has not been confirmed. Furthermore, RTRL requires generating multiple completions per sample, which could incur high computational costs when scaling to larger models.

### Questions
Regarding the robustness of the reward mechanism and "reward hacking":

1. How sensitive is this method to the hyperparameter $\alpha$? Does the setting of $\alpha$ require fine-grained manual tuning for different tasks?

2. (Important) How can this simple format check guarantee against more "smart" cheating? For example, the model could learn to generate an intermediate output $y$ that is chemically meaningless or very mediocre, but structurally easy to reverse engineer. This would still maximize your surrogate reward $\log p_{\phi}(x|y)$, but completely defeat the purpose of improving task performance. How do you prove that RTRL incentivizes genuine task improvement, and not just an improvement in "reversibility"?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Round-Trip Reinforcement Learning (RTRL), a framework designed to address the critical issue of round-trip consistency (RTC) failure in chemical Large Language Models (LLMs). This inconsistency suggests models rely on shallow, unidirectional memorization rather than robust, bidirectional mastery of chemical principles. The experimental results show that the proposed RTRL method can help improve the round-trip consistency and also improve the performance of base model. However, the effectiveness of the proposed method is only validated on paired round-trip tasks and is not validated on general chemistry tasks. In addition, the performance comparison lacks significant SOTA on some tasks, making the conclusion to be less well-supported.

### Strengths
1. Fosters deeper bidirectional coherence

By treating consistency as a direct, trainable objective, RTRL encourages the model to move beyond shallow, unidirectional memorization. 

2. Efficient utilization of unlabeled data

RTRL employs a self-supervised objective that is highly data-efficient. Training only requires access to inputs from a single domain (X), such as a list of molecules, without needing corresponding paired, labeled data.

3. Measurable performance and consistency gains

RTRL provides measurable gains over strong baselines, significantly boosting model performance and consistency.

### Weaknesses
1. **Limited generalizability and validation scope**

The approach is demonstrated exclusively on bidirectional generative tasks (such as molecule captioning ⇌ generation and reaction prediction ⇌ retrosynthesis). The core conclusion that round-trip consistency is critical is based on these specific tasks, and the framework is not validated on general tasks that are not round-trip tasks. Furthermore, RTRL currently does not work with classification or regression tasks.

2. **Performance comparison not comprehensive enough** 

While RTRL significantly improves the base model it is applied to (ChemDFM), the resulting absolute performance figures on certain tasks may not represent the highest State-of-Art (SOTA) benchmarks. For example, on the USPTO-50K Retrosynthesis task in the self-supervised setting, the Exact Match accuracy achieved was 0.151, the SOTA in the paper achieved 0.202. But the Llasmol [1] has the exact match accuracy 0.329, significantly higher than the SOTA baseline in the paper and the proposed method. 

[1] Yu, Botao, et al. "Llasmol: Advancing large language models for chemistry with a large-scale, comprehensive, high-quality instruction tuning dataset." arXiv preprint arXiv:2402.09391 (2024).

### Questions
1.  Can the proposed method be helpful for other non-round-trip-paired tasks or other chemistry tasks?

2.  Augmenting the round-trip paired tasks with the same dataset has been explored well in chemistry such as [1]. Can the proposed method be compared with those baselines that are simply trained with multiple combined datasets like Llasmol? It will also be great to include some chemistry-specific method like [1]. 


[1] Tetko, Igor V., et al. "State-of-the-art augmented NLP transformer models for direct and single-step retrosynthesis." Nature communications 11.1 (2020): 5575.

### Soundness
2

### Presentation
3

### Contribution
1
