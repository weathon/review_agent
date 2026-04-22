# Iter-AHMCL: Alleviate Hallucination for Large Language Model via Iterative Model-level Contrastive Learning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
The development of Large Language Models (LLMs) has significantly advanced various AI applications in commercial and scientific research fields, such as scientific literature summarization, writing assistance, and knowledge graph construction. However, a significant challenge is the high risk of hallucination during LLM inference, which can lead to security concerns like factual inaccuracies, inconsistent information, and fabricated content. To tackle this issue, it is essential to develop effective methods for reducing hallucination while maintaining the original capabilities of the LLM. This paper introduces a novel approach called Iterative Model-level Contrastive Learning (Iter-AHMCL) to address hallucination. This method modifies the representation layers of pre-trained LLMs by using contrastive positive and negative models, trained on data with and without hallucinations. By leveraging the differences between these two models, we create a more straightforward pathway to eliminate hallucinations, and the iterative nature of contrastive learning further enhances performance. Experimental validation on four pre-trained foundation LLMs (LLaMA2, Alpaca, LLaMA3, and Qwen) finetuning with a specially designed dataset shows that our approach achieves an average improvement of 10.1 points on the TruthfulQA benchmark. Comprehensive experiments demonstrate the effectiveness of Iter-AHMCL in reducing hallucination while maintaining the general capabilities of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Iter-AHMCL, a novel method to mitigate hallucination in large language models (LLMs) while preserving their general capabilities. The approach leverages iterative model-level contrastive learning by training positive and negative guidance models on hallucination-prone and hallucination-free data. These models guide the fine-tuning of the base LLM through representation editing, with an asymmetric iterative strategy that updates the positive model while keeping the negative model fixed. Experiments on LLaMA2, Alpaca, and LLaMA3 show significant improvements on TruthfulQA and HaluEval benchmarks (average +10.1 points), demonstrating reduced hallucination without compromising performance on knowledge-intensive tasks like MMLU and C-Eval.

### Strengths
1. The integration of contrastive learning with iterative model-level guidance offers a fresh perspective on hallucination reduction. By dynamically updating positive guidance models, the method adaptively enhances truthfulness while avoiding catastrophic forgetting.
2. The paper rigorously validates Iter-AHMCL across multiple LLMs (e.g., LLaMA2, Alpaca) and diverse benchmarks (TruthfulQA, HaluEval, MMLU), ensuring robustness and generalizability. Results consistently show improved hallucination metrics without degrading general capabilities.
3. Despite increased computational overhead, the method achieves faster convergence (e.g., 50 steps for Iter-AHMCL vs. 250 for LoRRA) and reduced training time. The asymmetric iterative design optimizes resource usage while maintaining performance.

### Weaknesses
1. There is a lack of relevant and state-of-the-art baseline methods for comparison, such as [1]. Even these state-of-the-art methods are not discussed in the paper.
2. The different losses proposed in the paper seem to rely on different optimal values ​​for \alpha and \beta, as shown in Table 4 and Figure 6. This makes the method rather inelegant, and choosing the optimal hyperparameters can be time-consuming.
3. I look forward to seeing how different losses can be combined together to form an overall method. In the current experiments, I cannot see how different losses affect each other.

[1] Refine Knowledge of Large Language Models via Adaptive Contrastive Learning. ICLR 2025.

### Questions
See the aboved weaknesses.

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
2

### Summary
The paper sits in the area of LLM hallucination mitigation with a focus on editing internal representations instead of only output-time filtering. The core question is: can we reduce hallucinations while preserving general capabilities, by steering hidden representations using learned positive/negative guidance, and can an iterative procedure strengthen that guidance?

The authors propose Iter-AHMCL to answer this. They (1) build contrast triples by templating each instruction with “give a truthful answer” and “give an untruthful answer,” (2) pre-train positive (M⁺) and negative (M⁻) guidance models on PKU-SafeRLHF subsets, (3) use these guidance models to form model-level contrastive losses during editing (CL-MG), and (4) iteratively update only the positive guidance model (CL-IMG). The contrast triple construction and the use of explicit truthful/untruthful templates are stated in §3.2, and the model-level losses and their equations appear in §3.5–§3.6.

### Strengths
First, the model-level guidance is the central design change. Prior representation editing work tends to use sample-level vectors or discriminators; here the authors learn M⁺/M⁻ and plug them into the loss so the base model is pulled toward M⁺(T⁺) and pushed away from M⁻(T⁻). This idea is simple to apply with LoRA and aligns with the stated objective.  

Second, the asymmetric iteration is well motivated: update M⁺ as the model improves, keep M⁻ fixed to preserve a stable contrast; the ablation/round table shows consistent gains on LLaMA-2.  

Third, the implementation details (edited layers, α/β, step budgets) and the data splits are specified, which helps reproduction.  
Fourth, the paper provides multiple views of evaluation: TruthfulQA MC1–MC3, HaluEval, plus auxiliary capability plots for MMLU/C-Eval.

### Weaknesses
The data/label mapping from safety to truthfulness is the largest issue. Guidance models are trained on PKU-SafeRLHF with “response safe=true/false,” and the contrast triples use “give a truthful/untruthful answer” templates. Safety and factual truth are related but not the same; unsafe is not equivalent to factually wrong, and safe is not guaranteed to be factually correct. This mismatch may bias the guidance toward safety style rather than factual accuracy. The construction and splits confirm this setup.  

The negative template “give an untruthful answer” may create behavior that is unlike natural hallucination. The model could learn to avoid a style rather than to improve evidence use. The paper does not test open-ended factual generation with grounding; all core results are multiple-choice.  

The selection protocol risks optimism. The text states the TruthfulQA MC1 score is used as the criterion for updating the positive model across rounds. If this is measured on the same evaluation set each round, it becomes iterative selection on a test set, which can inflate gains. The paper should either hold out a development split or report a final score only once on a never-seen test set. 

The capability preservation claim is not backed by tables in the main text. The appendix shows radar plots but not numeric breakdowns for MMLU/C-Eval by subject or difficulty. This makes it hard to judge trade-offs and variance across domains. 

Baselines could be broader. LoRRA is the closest editing baseline, and DPO/SFT are alignment baselines, but other editing/contrastive preference methods are absent in the main table, and Qwen results are not fully integrated into the same table. The appendix does include an ablation that separates “pure-MG,” LoRRA, and the combined method, but a stronger comparison set would increase confidence. 
 
The generality is shown only on 7B/8B models. While compute limits are real, the paper’s claim of broad applicability would be stronger with at least one mid-size model beyond 8B. 

Finally, parts of the objective design need clearer motivation. The paper inherits LoRRA’s L2 geometry and adds terms to pull/push against M⁺/M⁻, but there is limited analysis of layer sensitivity, stability across α/β, or why the chosen layers are optimal beyond a citation and a grid search note.

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Iter-AHMCL (Iterative Adaptive Hallucination Mitigation via Contrastive Learning), a novel framework for reducing hallucinations in large language models (LLMs) through iterative model guidance and contrastive fine-tuning. The core idea is to use positive and negative guidance models—each pre-trained to represent truthful and hallucinated directions respectively—to steer the main model toward factual generation.
The approach involves three key stages:
1.	Contrast Triple Construction (CTC): Builds triplets of neutral, positive (“truthful”), and negative (“untruthful”) samples from the PKU-SafeRLHF dataset.
2.	Guidance Model Pre-training (GMP): Pre-trains two guidance models — M⁺ (truth-oriented) and M⁻(hallucination-prone) — using low-rank adaptation (LoRA) on positive and negative datasets respectively.
3.	Iterative Model Guidance (CL-IMG): Iteratively fine-tunes the base LLM using model-level contrastive loss, where only the positive guidance model is updated in each iteration to gradually steer representations away from hallucination.
Experiment results across TruthfulQA, HaluEval, MMLU, and C-Eval demonstrate significant and consistent improvements over LoRRA and baseline foundation models (e.g., up to +19.4 MC1 on LLaMA2). Overall, Iter-AHMCL provides a theoretically grounded and empirically effective approach for reducing hallucination while preserving LLM general capabilities.

### Strengths
High Novelty in Iteration: The iterative update mechanism of the positive guidance model (M^+ 〖<-M〗_best) is highly original, enabling the system to continuously raise the bar for anti-hallucination alignment.
Effective Contrastive Anchoring: The method effectively leverages dedicated positive (M^+) and negative (M^-) guidance models to define clear anchors in the representation space for "truthful" vs. "hallucinatory" features.
Empirical effectiveness：Experimental results show consistent improvements over strong baselines (Foundation, LoRRA, DPO, SFT), demonstrating that the proposed iterative guidance effectively reduces hallucination while preserving task performance and fluency.
Robustness and Generalization: The approach's effectiveness is validated across diverse foundation models (LLaMA2, Alpaca, LLaMA3), suggesting good generalization capability.

### Weaknesses
Computational Overhead of Iteration: While the main model uses LoRA, the iteration loop (Algorithm 1) requires repeated evaluation (Line 16) and model replacement (Line 17) to find and set Mbest. This process adds computational overhead (e.g., increased training time and model switching costs) compared to a single-pass method like DPO. This trade-off should be quantified.
Limited sensitivity analysis: No sensitivity analysis is provided for key hyperparameters (α, β, batch size, learning rate). As a result, it remains unclear whether the reported improvements stem from the iterative contrastive framework itself or from specific parameter settings, weakening the empirical rigor of the paper.
Clarity on data construction: The paper briefly mentions the construction of triple sets (T,T+,T−) but does not elaborate on how hallucinated samples are generated or validated. Providing more details or examples would improve reproducibility.

### Questions
Fixed M^- Analysis:  Could the authors provide an ablation study justifying the decision to keep the negative guidance model (M^-) fixed throughout the iterations? Would allowing  M^- to decay or be updated (perhaps to represent harder negative examples) further improve performance or efficiency?
Training Cost Quantification: Please provide a quantitative comparison of the total training time (including all iterative evaluation and update steps) of Iter-AHMCL versus the single training pass of LoRRA or DPO, especially for a high number of iterations (N).
On stability and convergence: Since both the main model and the guidance model are updated iteratively, how do the authors ensure stability and prevent parameter drift or overfitting to the guidance model’s bias?
Cross-Domain and Cross-Model Transferability:Table 3 shows that a positive guidance model trained on LLaMA-2 can transfer to Alpaca with comparable performance.Have the authors explored broader cross-architecture or cross-domain transferability (e.g., between LLaMA2 ↔ Qwen, Mistral, or domain-specific models such as medical/legal LLMs)?Does performance degrade when tokenizer or pre-training corpus differ?

### Soundness
3

### Presentation
3

### Contribution
4
