# Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs

- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training.
A growing body of evidence has shown that RL fine-tuning improves the capability of LLMs beyond what SFT alone achieves.
However, the underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored.
In this study, we draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning.
Our analysis across multiple model families and mathematical datasets shows two robust effects of online RL post-training: (i) an overall increase in average activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions. These changes suggest that RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in mathematical generalization. Notably, models fine-tuned with Direct Preference Optimization (DPO) deviate from these trends, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training. Together, our findings provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches.
Our code is open source at https://github.com/tsinghua-fib-lab/llm_rl_probing_analysis.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the underlying mechanisms by which RL fine-tuning enhances the capabilities of LLMs. Leveraging a graph-theoretic perspective and edge attribution patching (EAP), the authors analyze the internal information flow of LLMs by comparing models before (after supervised fine-tuning) and after RL-based post-training. The study finds that online RL algorithms (like PPO and GRPO) systematically alter the internal circuitry of LLMs in two key ways: Increased Activation Intensity (More internal pathways become active, and their signal strengths increase, suggesting a more engaged and robust information flow) and Increased Activation Diversity (Activation patterns become more varied and less concentrated, indicating that the model develops more flexible and diverse pathways for problem-solving). Overall, the work provides a unified, mechanistic explanation for the performance gains seen from RL fine-tuning, bridging the gap between external model behavior and internal circuit-level transformations.

### Strengths
S1: The paper’s detailed comparison between online reinforcement learning methods (such as PPO and GRPO) and preference-based techniques like DPO—especially regarding the distinct internal transformations they trigger—offers a new lens through which to understand current LLM fine-tuning paradigms.

S2: The main contributions and results of the study are presented with clarity. The notions of activation intensity and activation diversity are both intuitive and insightful, effectively summarizing the internal changes observed. The discussion linking these changes to improved robustness and adaptability in information processing is straightforward, helping demystify the complex inner workings of LLMs. This lucidity allows the paper’s substantial technical insights to be easily grasped and appreciated by a wide range of AI researchers.

### Weaknesses
W1: Although the paper effectively highlights what changes occur—namely, increased activation intensity and diversity—it does not adequately explain why these specific changes arise from online RL. The discussion mainly points to correlations between RL fine-tuning and the resulting internal shifts. To advance beyond correlation, the work should develop clear, testable hypotheses detailing the causal mechanisms by which RL signals (such as reward functions or policy gradient updates) drive alterations in edge attribution and activation dynamics.

W2:  The study focuses on 7B parameter models and primarily LLaMA-style Transformers. While this is a common scale, the generalizability of the findings to a broader spectrum of LLMs remains unclear.

### Questions
Q1: For W1, could you share any thoughts on the possible causal factors or how you plan to investigate them?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the internal effects of reinforcement learning (RL) fine-tuning on large language models (LLMs), using the Edge Attribution Patching (EAP) framework to analyze residual pathways before and after RL fine-tuning. The authors report two consistent effects across several model families: (1) increased activation intensity and (2) greater diversity of activation patterns. They argue that these internal changes help explain the superior generalization performance of RL-fine-tuned models, particularly under PPO, GRPO, and DPO.

### Strengths
1. The paper tackles how RL fine-tuning affects the internal circuitry of LLMs, which is underexplored.
2. The findings are consistent across multiple model families and datasets, offering an interesting mechanistic view of RL’s role in enhancing information flow diversity.

### Weaknesses
1. While the paper observes consistent tendency over the three metrices (Activation Intensity, Information Complexity, and Distribution Kurtosis) between SFT and SFT with RL, it does not further verify whether these internal changes causally contribute to downstream performance improvements compared to SFT.

2. The experiments and analyses are limited to mathematical reasoning tasks, raising concerns about the generalizability of the findings to other domains (e.g., natural language understanding, dialogue).

3. All experiments are conducted on 7B-parameter models; including smaller (e.g., 1B) and larger (e.g., 14B) models would strengthen the claim of generality across scales.

### Questions
1. Which dataset is used for the visualizations shown in Figure 3?
2. In Table 1, how was the hyperparameter (e.g., truncation scale $\alpha$) chosen?

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
3

### Summary
This paper uses edge attribution patching to analyze how RL fine-tuning changes LLM internals across four 7B model pairs on math tasks. They find online RL methods (PPO, GRPO) consistently increase activation intensity and diversity in internal pathways, suggesting more redundant and flexible information flow. DPO-trained models don't show these patterns, revealing key differences between preference-based and online RL approaches.

### Strengths
This paper makes a valuable contribution by systematically bridging mechanistic interpretability with RL post-training across four model families, three benchmarks, and multiple hyperparameters. The core findings are clear and consistent: online RL methods (PPO, GRPO) reliably increase activation intensity and diversity in internal pathways, while DPO behaves fundamentally differently—providing mechanistic support for existing empirical observations about these algorithms. The experimental design is rigorous with sound methodology, the EAP framework is computationally scalable and theoretically grounded, and the paper excels in reproducibility with open-source code and detailed specifications.

### Weaknesses
1. No Causal Link to Performance:
The paper establishes correlation without causation. While the authors demonstrate that RL training correlates with both altered activation patterns and improved performance, they fail to prove that these internal changes actually cause the capability improvements. Critical intervention experiments are missing: for instance, artificially inducing these activation patterns in SFT models to test whether performance improves, or constraining RL training to maintain low activation intensity to assess whether performance degrades. Without such causal validation, the observed patterns may be mere byproducts rather than drivers of RL's effectiveness.
2. Descriptive Rather Than Explanatory:
The paper describes what changes—higher activation intensity and greater diversity—but not why or how RL objectives produce these patterns. It essentially concludes "RL makes models different internally" without explaining the underlying mechanism. Critically, the work lacks any connection to RL theory: there is no discussion of how exploration, credit assignment, policy gradients, or reward shaping might drive these activation changes. This leaves the findings as empirical observations without mechanistic grounding.
3. Proposed Enhancement: 
Incorporating RLVR (RL with Verifiable Rewards) would substantially strengthen the paper. RLVR would complete the methodological space—contrasting online RL with verifiable rewards against learned rewards (PPO/GRPO) and offline preferences (DPO)—thereby testing whether the key distinction is truly online versus offline training. For mathematical reasoning tasks, RLVR provides cleaner binary reward signals that could yield more interpretable activation patterns. Importantly, relevant RLVR-trained models already exist (e.g., Qwen2.5-Math) and would be straightforward to analyze within the existing framework.
Combined with additional mechanistic experiments—tracking activation changes throughout training trajectories, ablating specific RL components, and formally connecting observed patterns to RL theory—this approach would transform the paper from a descriptive observation into an explanatory contribution.

### Questions
1. From part 2.1, H^(2ℓ) is used to mean two different things?
At the beginning of the paragraph: "Let H^(2ℓ) ∈ R^(B×P×d_model) denote the input hidden state to the ℓ-th layer"
In the next sentence: "The output of the ℓ-th layer, H^(2ℓ), is computed..." Does the same notation represent both the input and output of layer ℓ?

2. Have you analyzed the dynamics of these changes during RL training? At what point in training do activation intensity and diversity increase—early, late, or gradually throughout? Does this correlate with capability improvements?

3. All experiments focus on mathematical reasoning. Have you tested whether these patterns hold for other domains where RL is commonly applied (e.g., coding, creative writing, instruction following)? If not, how confident are you that these are general properties of RL fine-tuning rather than math-specific phenomena?

### Soundness
2

### Presentation
3

### Contribution
2
