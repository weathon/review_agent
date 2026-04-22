# When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 8, 2

## Abstract
Large language models (LLMs) rarely admit uncertainty, often producing fluent but misleading answers, rather than abstaining (i.e., refusing to answer). This weakness is even evident in temporal question answering (QA), where models frequently ignore time-sensitive evidence and conflate facts across different time-periods. In this paper, we present the first empirical study of training LLMs with abstention ability while reasoning about temporal QA. Existing approaches such as calibration might be unreliable in capturing uncertainty in complex reasoning. We instead frame abstention as a teachable skill and introduce pipelines including one that couples Chain-of-Thought (CoT) supervision with Reinforcement Learning (RL) guided by abstention-aware rewards. Our goal is to systematically analyze how different information types and training techniques affect temporal reasoning with abstention behavior in LLMs. Through extensive experiments studying various methods, we find that RL yields strong empirical gains on reasoning: a model initialized by Qwen2.5-1.5B-Instruct surpasses GPT-4o by 3.46% and 5.80% in Exact Match on TimeQA-Easy and -Hard, respectively. Moreover, it improves the True Positive rate on unanswerable questions by 20% over a pure supervised fine-tuned (SFT) variant. Beyond performance, our analysis shows that SFT induces overconfidence and harms reliability, while RL improves prediction accuracy but exhibits similar risks. Finally, by comparing implicit reasoning cues (e.g., original context, temporal sub-context, knowledge graphs) with explicit CoT supervision, we find that implicit information provides limited benefit for reasoning with abstention.  Our study presents new insights into how abstention and reasoning can be jointly optimized, providing a foundation for building more reliable LLMs. Dataset and code is publicly released https://github.com/Blackzxy/AbstentionTemporalQA.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies abstention-aware temporal question answering with LLMs. It proposes a two-stage pipeline: (i) distill CoT rationales with GPT‑o1 and SFT the policy; (ii) continue training with GRPO using a simple, rule-based abstention-aware reward. The work also explores implicit (time-filtered sub-context, KG snippets) and explicit CoT reasoning signals. On TimeQA, the best model exceeds GPT‑4o with the original context by +3.46 and +5.80 EM, respectively. The paper further analyzes prompt designs, number of KGs, reward variants, and out‑of‑distribution generalization.

### Strengths
1. Timely focus on abstention + temporal reasoning.
2. Comparison across input settings (question only, full context, time-filtered sub-context, KGs), model scales, SFT vs RL, and prompt variants sheds insights into the domain.
3. The experimental setup is detailed nicely for reproduction.
4. Some interesting analysis is performed, including:
    4.1. SFT increases overconfidence
    4.2. Increasing unanswerable questions in training can collapse the model
    4.3. Impact of KG on abstention

### Weaknesses
1. **Lack of Benchmarks** - Results are confined to TimeQA. Other temporal reasoning sets (e.g., [1,2]) would better validate generality. The OOD experiments (Table 4, p. 9) focus on non‑temporal datasets and show very poor transfer after RL (e.g., TP -> 0 on RL+c), which underscores brittleness.

2. **Heavy reliance on GPT-o1 for CoT** - CoT collection use GPT‑o1. This raises questions about measuring knowledge distillation from larger models, rather than assessing the impact of suggested training and potential subtle leakage or bias from those systems.

3. **Training Methodology isn't novel** - The training stack (CoT‑SFT + GRPO) and rule‑based reward (Eq. 2) are standard training paradigms; the main novelty lies in the selected domain and results rather than new learning algorithms.

4. **Lack of LRMs** - Recent LRMs (eg, o4-mini, Gemini-2.5-Pro, or open-source LRMs -- DeepSeek-R1, Qwen-Thinking) have shown substantial reasoning improvements in temporal domains. Using them will shed more insights into state-of-the-art models and performance.

[1] Uddin, Md Nayem, et al. "UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization." arXiv preprint arXiv:2407.03525 (2024).
[2] Fatemi, Bahare, et al. "Test of time: A benchmark for evaluating llms on temporal reasoning." arXiv preprint arXiv:2406.09170 (2024).

### Questions
See weaknesses above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the important question of how to teach large language models (LLMs) the skill of abstention: not answering a question. This work focuses on questions involving a temporal dimension. The authors explore various forms of SFT and RL (using GRPO) to induce abstention. The authors show training a model with chain-of-thought supervised finetuning followed by reinforcement learning can boost abstention. The authors highlight that generalizing outside of the TimeQA benchmark to out of domain benchmarks such as MMLU is still challenging.

### Strengths
The authors tackle the important open problem of the best approach to teach models the skill of abstention for temporal questions. The authors cover a reasonable set of closed and open models as well as explore various setups for inducing abstention, including various approaches to including context. The authors make reasonable choices in terms of post-training methods (GRPO, SFT) and adapt them for abstention. I commend the authors on the perspective that abstention is a learnable skill and the thorough exploration in post-training approaches to induce it.

I appreciate the authors were careful about evaluation by including both correct abstention and over-abstention in the experiments. I also appreciate the evaluation of both in-domain TimeQA as well as other out-of-domain benchmarks to assess generalization. I also appreciate the authors' proactive inclusion of limitations such as model size.

The findings are quite interesting and offer a practical recipe for inducing abstention for temporal questions. The finding regarding the importance of data mix (answerable versus unaswerable) is also quite neat! The authors also highlight the important open problem of teaching LLMs the skill of abstention more generally, as well as the limitations of only teaching abstention using SFT.

### Weaknesses
The authors offer some nice findings comparing post-training approaches for abstention, including the lack of success of some approaches (SFT). One aspect that could be improved here is some more intuition regarding why some setups work better than others. There is a growing body of literature along the lines of https://arxiv.org/abs/2501.17161 which explains memorization and generalization learning dynamics of post-training approaches that can be used to better contextualize this works' findings.

The hyperparameters and exact setups used can play a large factor here. The authors' present claims in quite a general manner, not sufficiently accounting for the limited setup used to justify them. For example, broad claims about SFT versus RL are supported only with a single method (GRPO) or limited hyperparameter selection choices for SFT. I'd also be curious to see whether the result in line 397 holds with LoRA, which as been shown to reduce overfitting.

The experiments regarding generalization are quite interesting. I imagine most common post-training setups will include other data for alignment. How would this skill of abstention interact with the standard post-training pipeline aimed at aligning models? 

While I believe temporal questions are certainly important, I believe the authors could do a better job setting up why it's worth focusing only on temporal questions. Adding more context about why temporal questions are particularly important or worth focusing on solely would help to better frame the contribution.

### Questions
- While temporal questions are certainly important, have the authors considered whether this approach would generalize to other types of unanswerable questions? It's not necessary to run these additional experiments for the scope of this paper, but it would be useful to discuss in the context of future work. 
- How are the hyperparameters in lines 244-246 selected? Is there precedent or justification in prior work or was a sweep conducted?

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
3

### Summary
This paper studies how RL can teach language models to abstain from answering when uncertain, particularly in temporal question answering. The authors introduce an RL framework with abstention-aware rewards where models are rewarded for saying "no answer" on unanswerable questions. Their method outperforms both GPT-4o and SFT baselines, showing higher accuracy and better handling of unanswerable questions. Results are supported by numerous ablations over prompt configurations, hyper-parameter tuning and generalization.

### Strengths
- **Relevance and Importance**: The paper addresses an important and timely problem. Current models struggle to abstain and the proposed RL technique is simple and effective. 

- **Strong Results**: The results, although surprising, are strong. RL significantly beats SFT, even for SFT models of larger sizes.

### Weaknesses
- **Poor Structure and Presentation**: The paper’s organization lacks coherence. Sections jump between unrelated topics (e.g., implicit reasoning, KG extraction, RL training) without clear motivation or integration into the main story. Figures and experiments are presented out of logical order, reducing readability.

- **Weak Experimental Design**: Some experiments feel arbitrary or poorly motivated. Dataset choices, baselines, and prompt configurations are insufficiently justified, and several results are unintuitive.

- **Limited Applicability**: This approach assumes access to datasets with unanswerable questions, which is generally not applicable. The proposed approach also seems very targeted to TemporalQA, which further limits applicability.

### Questions
- **Classifier Baseline:** Can the authors add a classifier baseline that predicts if a question is answerable? This could be combined with any model that always generates an answer, and is a simple post-hoc baseline that can compliment models that struggle to abstain. 

- **Existing Work:** There is some prior work [1] training models to abstain using RL on unanswerable questions. Can the authors discuss novelty compared to this (and possibly more) existing works. It is okay if they were concurrent works. 

- **SFT Data:** Does the SFT data also include unanswerable questions? If so, how are these distributed? They should ideally have the same proportion of unanswerable questions as the RL training (where data ratio was so critical). 

- **SFT performance:** Why is SFT performance so poor relative to RL? The SFT dataset appears to have only 1K examples compared to 20K for RL. Could this explain the gap? Note that I am not surprised by the fact that RL beats SFT, but rather by the margin of defeat. A 1.5B model beating a 8B model this significantly suggests that the SFT pipeline has issues.

- **Frontier Model evaluations:** For frontier model evaluations, what prompts were used? Were models explicitly told they could abstain? I think the best prompt to use for these evaluations is exactly the training prompt in Table 8 (except the Qwen/Alibaba text). 

- **Implicit Reasoning:** What is the importance of implicit reasoning, such as temporal reasoning or KG extraction, in the story of this paper? In particular, the best models seem to perform well even without these augmentations—what value do these methods add? If the focus of this paper is RL, then the implicit reasoning methods should be presented as baselines. 

- **Generalization**: The OOD generalization tasks differ substantially. In TimeQA, abstention is due to lack of information; in MCQ datasets, abstention reflects model inability. Are these two forms of abstention comparable? Why not evaluate on AbstentionBench, which contains ambiguous/unanswerable tasks and seems more aligned with the paper’s goals?

- **Focus on TemporalQA**: The emphasis on temporal QA is unclear. Why is this chosen as the focus, when the RL framework could generalize to other question-answering tasks which require abstention as well (multi-hop reasoning for example) ? 

- **Qwen 2.5-7B Results**: The results of this model on TimeQA-Hard are surprising. EM Accuracy is highest when only given the question. Context seems to reduce performance. Why is this happening?  

- Section 3.1 should clearly specify that some questions are explicitly unanswerable.

- The figures are misordered and disrupt reading flow. For instance, results for Figure 4 precede those for Figure 3, and Figure 5 appears beside unrelated text (Experiment 6.2).

[1]: Song, L., Shi, T., & Zhao, J. (2025). The hallucination tax of reinforcement finetuning. arXiv preprint arXiv:2505.13988.

### Soundness
2

### Presentation
1

### Contribution
2
