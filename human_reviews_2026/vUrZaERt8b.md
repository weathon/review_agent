# Resa: Efficient Reasoning Models via SAEs

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
How cost-effectively can we elicit strong reasoning abilities in language models by leveraging their underlying representations? We present Resa, a family of reasoning models trained via an efficient sparse autoencoder tuning (SAE‑Tuning) procedure. This method first trains an SAE to capture reasoning abilities from a source model, and then uses the trained SAE to guide a standard supervised fine-tuning process to elicit such abilities in a target model, all using verified question-answer data *without any reasoning traces*. When applied to certain Qwen-style models before further RL training, SAE‑Tuning retains 97\% of its RL‑trained counterpart’s performance while reducing training costs by 2000x to roughly \$1 and training time by 450x to around 20 minutes. Furthermore, even at the 1.5B model size, SAE-Tuning on lightly RL-trained models delivers strong reasoning results, reaching 43.33\% Pass@1 on AIME24 and 90\% Pass@1 on AMC23. We also show that SAE-Tuning works for Llama-style models, boosting their scores by over 10\% on tasks like AMC23 and MATH500. Surprisingly, the reasoning abilities extracted via SAEs are potentially both generalizable and modular. Generality means abilities extracted from one dataset still elevate performance on a larger and overlapping corpus. Modularity means abilities extracted from models like Qwen or Qwen‑Math can be attached to the R1‑Distilled Qwen model at test time, *without any retraining*, and yield comparable gains. Extensive ablations validate these findings and all artifacts are fully open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents SAE-Tuning, which first trains an SAE to capture reasoning abilities from a source model and then uses the trained SAE to guide a supervised finetuning process to elicit these abilities in a target model. The authors verify its effectiveness on two models and further show that the training can be simplified from both the source and SAE aspects. They also demonstrate that the reasoning abilities extracted via SAEs can be transferred across both data distributions and models.

### Strengths
1. The paper is well-written and easy to follow.
2. The SAE-Tuning method is novel and cost-effective compared to the RL method.

### Weaknesses
1. In the experiment, the target model is always an R1-distilled method. But from a feature extraction perspective, features should first be extracted from an R1-distilled model and further guide the tuning of a non-R1-like model. Therefore, the application of the method remains limited.
2. The experiments are limited to small models and math tasks. A larger scale and more tasks are required to verify its effectiveness.

### Questions
- How does the performance compare to the RL method when the target model is not R1-distilled?
- Have the authors tested on other tasks and model scales?
- How do the model's reasoning behaviours, such as response length, change when applying the SAE-Tuning?

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
5

### Summary
This paper introduces **Resa**, a family of reasoning models trained via **SAE-Tuning** (Sparse Autoencoder Tuning), a novel two-stage method that elicits reasoning abilities in language models without requiring reinforcement learning or chain-of-thought traces. The method first trains a sparse autoencoder (SAE) to extract latent reasoning features from a source model's activations, then inserts the frozen SAE into a target model to guide supervised fine-tuning with LoRA adapters using only verified question-answer pairs. The key contributions include: (1) achieving comparable performance to RL-trained models while reducing training costs by 2000× (to $1) and time by 450× (20 minutes), (2) demonstrating that reasoning abilities extracted via SAEs are generalizable across datasets and modular across models within the same family, working as portable "adapters" at inference time, and (3) showing effectiveness across model architectures (Qwen and Llama), with 1.5B models achieving 43.33% Pass@1 on AIME24 and 90% on AMC23, suggesting that reasoning abilities exist latently in base models and can be efficiently extracted and transferred.

### Strengths
### Originality
The paper presents a novel application of sparse autoencoders to reasoning ability transfer, which is creative and distinct from prior work. While SAEs have been used for interpretability and steering, using them as a bridge to extract and transfer reasoning capabilities between models is innovative. The concept of treating reasoning as a "modular adapter" that can be plugged into models at test time without retraining is particularly original and opens new research directions.

### Quality
The experimental work is comprehensive, with extensive ablations across multiple dimensions (SAE training modes, source models, layer selection, datasets). The paper evaluates on six diverse reasoning benchmarks and demonstrates consistency across different model families (Qwen and Llama). The inclusion of detailed cost and time analysis (Table 6) strengthens the practical claims. The layerwise analysis (Section C.1) with GMM fitting provides interesting insights into where reasoning features reside in the model.

### Weaknesses
## 1. Insufficient Evidence for Core Theoretical Claim

**Lines 46-48:** The paper claims "within this dictionary, certain features must correspond to the fundamental building blocks of reasoning" as a key insight. However, no empirical evidence or theoretical justification is provided to support this claim. This is a foundational assumption for the entire SAE-Tuning methodology, yet it remains unsubstantiated. The authors should provide either:
- Quantitative analysis showing which SAE features activate during reasoning tasks
- Ablation studies demonstrating that specific features are necessary for reasoning performance
- Theoretical arguments grounded in interpretability literature

## 2. Unfair Experimental Comparison in Ablation Study

**Lines 315-318, Table 1:** The comparison between Resa-STILL-v1 and STILL-CoT-free-SFT is fundamentally flawed:
- **Resa-STILL-v1** is trained using SAE-Tuning with Tina-STILL as the source model, where Tina-STILL has already acquired long CoT capabilities through RL training
- **STILL-CoT-free-SFT** appears to be trained directly from STILL-3-1.5B-preview (44.86% avg), which lacks long CoT capabilities
- This creates an unfair comparison: Resa inherits reasoning abilities from an RL-trained model, while the baseline starts from a weaker foundation
- More puzzlingly, STILL-CoT-free-SFT (39.00%) performs *worse* than its presumed base model STILL-3-1.5B-preview (44.86%), raising questions about training stability or base model identity

**Required clarification:**
- What is the exact base model for STILL-CoT-free-SFT?
- Why does training decrease performance?
- A fair comparison would use the same base model (STILL-3-1.5B-preview)

## 3. Conceptual Inconsistency in SAE Training Modes

**Lines 192-196:** The "Pre-trained" SAE mode directly contradicts the paper's core methodology:
- The paper emphasizes that Stage I is essential for SAEs to learn reasoning features from the source model on the trigger dataset
- However, the "Pre-trained" mode uses a generic SAE trained on R1-Distill and "bypasses Stage I entirely"
- This raises fundamental questions: How can a pre-trained SAE that has never seen the source model (e.g., Tina-STILL) or the trigger dataset capture source-specific reasoning features?
- Table 2 shows Pre-trained SAE performs worse (44.99%) than Fine-tuned (47.28%) or Trained-from-Scratch (47.36%), but the fact that it works at all contradicts the claimed necessity of Stage I

**This inconsistency undermines the paper's theoretical framework and requires clarification.**

## 4. Minor Typographical Error

**Line 216 (Equation 3):** "where Top-k means that we only the top k features in the vector" is missing a verb. Should read "we only **keep** the top k features" or similar.

## 5. Incomplete Evaluation Protocol Documentation

**Throughout the paper:** Critical evaluation details are missing or buried in the appendix:
- The evaluation metric (Pass@1) is only mentioned on line 755 in the appendix, not in the main text or table captions
- **Sampling details are completely absent:** How many times is each problem sampled? What are the sampling parameters (temperature, top-p)?
- For reasoning benchmarks, these details significantly impact reproducibility and fair comparison
- This information should be prominently displayed in Section 4 or in table captions

Although I am not an expert in this domain, I think the writing could be clearer to help readers unfamiliar with the topic better understand your paper. If your response is reasonable, I am happy to raise my score. 


## 6. Misleading Cost Claims Due to Upstream RL Dependency

**Core issue:** The paper's central claim of 2000× cost reduction ($2268 → $1) is misleading because it **does not account for the upstream RL training costs** required to create the source models.

**The dependency chain:**
- SAE-Tuning extracts reasoning abilities from source models like Tina-STILL or R1-Distill
- **Tina-STILL** was trained via expensive RL from R1-Distill
- **R1-Distill** itself was created by distilling reasoning traces from DeepSeek-R1, which required expensive RL training
- Therefore, the complete cost is: **Upstream RL cost (thousands of dollars) + SAE-Tuning ($1)**

**Why this matters:**
- SAE-Tuning does **not create reasoning abilities from scratch**—it only transfers/copies existing abilities from RL-trained models
- Without access to an RL-trained source model with long CoT capabilities, the method cannot work
- The claimed "$1 reasoning model" is actually "$1 to copy an existing reasoning model that cost thousands to create"
- This is fundamentally a knowledge distillation/transfer method, not a standalone reasoning elicitation method

**Fair framing:** The paper should position SAE-Tuning as an efficient method for **transferring** reasoning abilities between models (assuming one RL-trained model exists), rather than claiming it eliminates the need for RL training entirely. The cost comparison should either include amortized upstream costs or be clearly framed as "marginal cost per additional model."

### Questions
## Q1: Evidence for Reasoning Feature Identification
Can you provide concrete evidence that SAE features correspond to reasoning building blocks?

## Q2: Clarification on Ablation Baseline
For the STILL-CoT-free-SFT baseline in Table 1:
- What is the exact base model used for this baseline?
- If it's STILL-3-1.5B-preview, why does performance decrease from 44.86% to 39.00% after training?
- Can you provide a fair comparison where both Resa and the baseline start from the same base model?
- What was the training setup (learning rate, epochs, etc.) that led to this performance degradation?

## Q3: Pre-trained SAE Mode Mechanism
Regarding the "Pre-trained" SAE mode:
- How can a pre-trained SAE that bypasses Stage I and never sees the source model or trigger dataset still extract reasoning abilities?

## Q4: Detailed Evaluation Protocol
Can you provide complete evaluation details:
- How many samples per problem for Pass@1? (Is it truly 1 sample per problem, or multiple samples with pass@1 criterion?)
- What are the exact sampling parameters (temperature, top-p, etc.)?
- Are these consistent across all baselines and your method?
- Why was this information relegated to the appendix rather than included in the main experimental section?

### Soundness
2

### Presentation
2

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
This paper introduces RESA, a method for eliciting reasoning ability in language models using Sparse Autoencoder (SAE) tuning.
The key idea is to extract latent reasoning features from a teacher model (often an RL-trained reasoning model like Tina) via an SAE, and then inject those features into a target model using a lightweight SFT procedure.
The authors claim that this achieves near-RL-level reasoning performance at a fraction of the cost and without reinforcement learning or Chain-of-Thought (CoT) data.

### Strengths
1.Timely and interesting topic.

The paper targets an important direction: efficient reasoning ability transfer.
It goes beyond standard output-level distillation by exploring structural feature transfer through SAEs — a refreshing and potentially impactful angle.
Indeed, model capability acquisition is not limited to simple knowledge distillation, and studying low-cost, structural-level distillation is both novel and valuable.

2.Good integration of interpretability and model training.
Using SAEs to capture internal reasoning representations bridges interpretability and capability transfer.
This is a strong conceptual contribution that could inspire future work at this intersection.

3.Clear writing and presentation.
The paper is well organized and generally easy to follow.
Figures and tables clearly convey the main experimental findings.

### Weaknesses
1. Questionable claim of “efficiency.”

While the paper repeatedly claims RESA is efficient, the evidence is not fully convincing.
  - Section 4.1 only shows capability replication, not true efficiency.  RESA is demonstrated to replicate reasoning performance of an RL-trained model (Tina) using much cheaper fine-tuning,  but this assumes the existence of that RL teacher. Training Tina itself is extremely costly. Therefore, if we count the full pipeline cost (RL teacher + RESA transfer), RESA is not cheaper overall — it is conditionally efficient only when a high-quality teacher already exists. In contrast, Section 4.2 includes a self-distillation experiment, which also achieves reasonably strong performance, suggesting that RESA may not necessarily rely on an expensive RL teacher to obtain comparable results？
  - The “self-teacher” experiment (§4.2) partially addresses this but with misaligned settings. The Tina teacher and the R1-Distill teacher are trained on different data and objectives. Specifically, in the Tina pipeline, the teacher used for distillation is the same model trained on the corresponding dataset (e.g., Tina-STILL’s teacher is the model produce the STILL data). In contrast, RESA uses R1-Distill-Qwen-1.5B as its teacher across all experiments, which was not trained on the same dataset or objective as Tina’s teacher. Thus, their comparison does not isolate the gain from SAE-Tuning alone. A fairer efficiency claim should compare RESA vs. standard distillation.For example: given the same teacher, same base model, and same QA data, one could compare (a) traditional distillation (prompt–response SFT) vs. (b) RESA’s SAE-guided tuning. This would show whether RESA is not only cheaper than RL, but also more data-efficient or optimization-efficient than conventional post-training.
- Moreover, in practice, if a strong checkpoint already exists, directly using that model remains the most cost-effective option, making the claimed “efficiency” of RESA less meaningful in realistic scenarios.

2. Lack of discussion or evidence on scalability.
  - No evidence of efficiency in scaling direction. “Efficiency” would be more convincing if the method could transfer reasoning from a large teacher (e.g., 7B) to a smaller model (e.g., 1.5B) with reduced cost. Such large-to-small transfer is the true motivation behind efficiency, yet no such experiment is included. Overall, Section 4.1 validates feasibility (“it works”) but not efficiency (“it works better”).
  -  Although RESA’s optimization cost is small (only LoRA adapters are trained), the SAE module itself is extremely large: For a 1.5B base model (hidden size 4096), the SAE has roughly 2.1B parameters. Since SAE scales as O(d^2) with hidden size, applying it to 7B–70B models would require billions to tens of billions of parameters just for the SAE. This raises major scalability questions:
    - Can such an SAE be trained or even stored for larger models?
    - Does the efficiency still hold when the feature dimension grows?
    - Is there a way to share or compress SAEs across layers or models?
Currently, all experiments are limited to 1.5B models, and scalability is only discussed conceptually.
Hence, the method’s practicality for modern large-scale models remains unclear.

### Questions
1. On the Efficiency Claim

- How should the claimed “efficiency” be interpreted when training the RL teacher (Tina) is itself very costly?
- The self-distillation experiment (§4.2) performs well without an RL teacher. Does this imply that RESA’s efficiency mainly comes from self-distillation rather than teacher-based distillation? 
- Have you compared RESA with standard distillation or SFT under identical settings (same teacher, same base model, same QA data)? Such a comparison would help validate whether RESA is truly more efficient or simply an alternative distillation form.

2. Scalability of the SAE Module

- How does the parameter count of the SAE grow with model size? For example, with a hidden size of 4096 (1.5B model), the SAE already contains ~2.1B parameters. Would applying RESA to 7B or larger models require prohibitively large SAEs?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Resa, a family of reasoning models, and introduces the SAE-Tuning framework to efficiently elicit reasoning abilities in language models. SAE-Tuning first trains a sparse autoencoder (SAE) on a source model (e.g., Qwen-style Tina) using CoT-free verified question-answer data to extract latent reasoning features, then freezes the SAE and embeds it into a target model (e.g., R1-Distill) for guided supervised fine-tuning.

### Strengths
Experiments show the method retains ~97% performance of RL-trained models while cutting training costs by 2000x (to ~$1) and time by 450x, solving the high-cost issue of RL and CoT-based SFT while maintaining competitive performance
proving the generalizability and modularity of SAE-extracted reasoning abilities, breaking the limitation of task-specific or model-specific reasoning tuning

### Weaknesses
- most experiments focus on small models (1.5B/3B), leaving uncertainty about whether SAE-Tuning maintains efficiency and performance on larger models
- The method also relies on source and target models sharing the same architecture, which restricts its applicability across different model families
- the analysis of "reasoning features" remains superficial; while the paper identifies their layer-wise distribution, it lacks in-depth interpretation of what specific reasoning components these features correspond to (e.g., logical deduction vs. numerical calculation).

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
