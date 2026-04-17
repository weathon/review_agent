# Reinforcement Unlearning via Group Relative Policy Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
During pretraining, LLMs inadvertently memorize sensitive or copyrighted data, posing significant compliance challenges under legal frameworks like the GDPR and the EU AI Act. Fulfilling these mandates demands techniques that can remove information from a deployed model without retraining from scratch. Existing unlearning approaches attempt to address this need, but often leak the very data they aim to erase, sacrifice fluency and robustness, or depend on costly external reward models. We introduce PURGE (Policy Unlearning through Relative Group Erasure), a novel method grounded in the Group Relative Policy Optimization framework that formulates unlearning as a verifiable problem. PURGE uses an intrinsic reward signal that penalizes any mention of forbidden concepts, allowing safe and consistent unlearning. Our approach achieves up to $\times$46 lower token usage per target than state-of-the-art methods, while improving fluency by +5.48\% and adversarial robustness by +12.02\% over the base model. Extensive evaluation on the Real World Knowledge Unlearning (RWKU) benchmark shows that PURGE reaches 11\% unlearning effectiveness while preserving 98\% of original utility. PURGE shows that framing LLM unlearning as a verifiable task, enables more reliable, efficient, and scalable forgetting, suggesting a promising new direction for unlearning research that combines theoretical guarantees, improved safety, and practical deployment efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes PURGE, an unlearning RL framework for LLM. The method is built on the idea of Group Relative Policy Optimization. PURGE frames unlearning as a verifiable task with an intrinsic reward that penalizes output containing any tokens from a specified forget vocabular. It constructs this vocabulary by probing the base model with RWKU’s rejection-tuning questions and then running a conditioned NER to produce a set of terms; training then maximizes group-relative advantages with a KL regularizer to preserve the model performance. The method includes theoretical results such as leakage rates for unlearning guarantee and a utility bound that relates downstream performance to the KL-divergence. The proposed framework is evaluated on the RWKU benchmark across an array of metrics. Gains over baselines are competitive or stronger than SOTA on several metrics.

### Strengths
1. The paper is generally well written with self-contained and consistent use of notation. The idea is also clear and well-motivated, casting unlearning as a verifiable objective with a simple indicator reward, no expensive external reward model. The GRPO objective and vocabulary construction are described concretely.

2. The framework feels pipelined and easy to use, the idea of using probing to elicit model-specific answers and NER to produce forget target, and is practical and general, applicable to many unlearning methods.

3. Theoretical guarantees match the intended contribution, providing insights on both the forgetting capability (Theorem 1) and relating downstream performance with the KL-divergence, which can be explicitly controlled during the unlearning procedure (Theorem 2).

4. Solid empirical studies are provided using RWKU. The main table shows improved forgetting vs. base and competitive SOTA methods. The main PURGE benefits (efficiency per target) are also clearly shown.

5. Ablations study shows robustness against 8/9 adversarial attacks, and scaling curves across Qwen sizes demonstrating monotonic improvements in forgetting.

### Weaknesses
1. Keyword or vocab-based unlearning does not consider obfuscation and homonym issues. This suggests surface suppression rather than semantic forgetting.

2. Surface-level keyword suppression does not entirely solve the issue of memorizing sensitive information. There is a gap between the real intention of machine learning (to produce a model as if the model never sees targeted information) and the contribution of this paper (to suppress keywords associated with that information). 

3. The proposed method heavily relies on an additional large language model for forget corpus construction. However, there isn't a way to directly assess the quality of the targeted corpus itself or how well it is associated with the actual unlearning target.

4. The mixing rate alpha does not seem to be reflected in the algorithm itself, but is the key assumption for Theorem 1 on the suppression guarantee.

5. The scope and evaluation presented in this paper are somewhat limited, mainly focusing on RWKU. Its generalizability broader privacy/safety settings is deferred to future work.

### Questions
1. Theorem 1 assumes mixing of policies instead of a fixed policy. Please clarify whether the mixing rate α is used and how it is reflected in the algorithm. If not, how does it affect the guarantee?

2. How or could the proposed unlearning mechanism be transferred beyond keyword-forgetting? Do authors have a semantic or entailment-based criterion to prevent typos or obfuscation?

3. Why is GRPO preferable for unlearning tasks? Are there any alternatives to GRPO?

### Soundness
2

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
The paper introduces a reinforcement learning approach to achieve verifiable and efficient unlearning in large language models. It treats unlearning as a policy optimization problem that reduces the likelihood of generating forbidden content while preserving the model’s overall capabilities. PURGE employs a group-relative policy objective with a KL regularizer to ensure theoretical guarantees such as bounded utility loss and provable forgetting efficiency. Experiments on the RWKU benchmark demonstrate that PURGE performs unlearning much faster, using up to 46 times fewer tokens than existing methods, while maintaining fluency and robustness against adversarial inputs. This work presents a principled and scalable framework for safe and reliable unlearning in large language models.

### Strengths
1. The paper introduces a novel approach, PURGE, which reframes unlearning as a reinforcement learning problem. The use of group-relative policy optimization and the introduction of a verifiable unlearning process are both creative contributions, distinguishing this work from previous methods that often require full retraining or lack theoretical guarantees.
2. The method is rigorously designed with both theoretical guarantees and practical effectiveness. The authors provide a clear framework with a solid mathematical foundation, offering provable bounds on utility loss and forgetting efficiency. Empirical results on the RWKU benchmark demonstrate its superiority in terms of token efficiency and robustness.
3. The paper is well-structured and easy to follow, with clear explanations of the proposed method, the motivation behind it, and how it compares to existing approaches. The use of both theoretical analysis and empirical results strengthens the overall clarity and comprehensibility.

### Weaknesses
1. The work only compares performance on a single benchmark, RWKU. Other benchmarks, such as TOFU, should also be included in the experiments to better demonstrate the generalizability of the proposed method.
2. According to Table 1, PURGE lags behind the baseline on many metrics (including QA, FM, GA, etc.). Although Section 5.2 provides some explanations, the method's broad effectiveness remains questionable. Given the generally poor performance of the Utility Set, there may be a potential issue of overfitting after training, which could undermine other general capabilities.
3. The experiment in Section 5.1 is conducted only on the Phi-3-Mini-4K-Instruct model, and no results from other models are introduced to demonstrate that the method's effectiveness is not limited to a specific model.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces PURGE (Policy Unlearning through Relative Group Erasure), a novel reinforcement unlearning framework designed to remove sensitive or copyrighted data from Large Language Models (LLMs) to meet regulatory requirements like the GDPR. Unlike existing methods that suffer from data leakage, fluency degradation, or reliance on costly external reward models, PURGE formulates unlearning as a verifiable task grounded in the Group Relative Policy Optimization (GRPO) framework. The method utilizes a highly efficient, synthetically generated "forget corpus" and an intrinsic reward signal to penalize any mention of forbidden concepts without an external reward model. Key contributions include theoretical guarantees on information suppression (geometric decay of forbidden-token probabilities) and utility retention (a high-probability bound), alongside competitive empirical performance.

### Strengths
1. The paper is well-written and easy to follow. The logical flow from the proposed algorithm (PURGE), through its theoretical analysis, and into the experimental results is organic, making the core contributions clear and understandable.

2. A primary contribution is the novel formulation of LLM unlearning as a verifiable task, shifting the paradigm from standard preference-optimization or gradient-ascent methods. This re-framing is creatively combined with Group Relative Policy Optimization (GRPO), a technique that cleverly circumvents the need for a costly external reward model by using an intrinsic, computable reward signal.

3. The work demonstrates high quality through its rigorous theoretical backing. It provides formal guarantees for both information suppression (Theorem 1, proving geometric decay of leakage) and utility retention (Theorem 2, bounding performance drops via $KL$ divergence).

4. The empirical evaluation is comprehensive, conducting direct comparisons against strong baselines on the RWKU benchmark. The results validate that the proposed algorithm achieves competitive performance across most metrics. Furthermore, the ablation studies are particularly insightful for verifying the method's advantages from multiple perspectives.

### Weaknesses
1. Limited Discussion of Recent Literature: While the paper cites foundational unlearning works, it lacks engagement with the most recent literature, particularly the significant volume of unlearning papers from ICLR 2025 [1-5]. The authors should incorporate this discussion to more clearly differentiate their contributions from these recent papers.

2. Dependency on External Proprietary Models: The "Synthetic Forget Corpus Construction" (Sec 4.1) creates a significant external dependency on a powerful, proprietary model (GPT-4) for its core NER task. This introduces concerns regarding cost, reproducibility (as the external model may be updated), and potential biases. To validate the method's robustness against this dependency, the authors should demonstrate whether superior performance is maintained when using other (e.g., open-source) LLMs for the corpus construction step.

3. Questions Regarding Experimental Results on RWKU: The paper's core claim of achieving competitive performance with significantly fewer tokens is not sufficiently substantiated.

- Missing SOTA Baselines: As noted in point 1, the most recent unlearning algorithms [1-5] are not included in the experimental comparison. At a minimum, a few of these should be implemented as baselines. A potential comparison method could be to integrate the core unlearning loss functions from [1] or [4] into the PURGE framework to ensure a fair comparison.

- Unclear Superiority over NPO: It is questionable whether PURGE is demonstrably superior to NPO or DPO. The paper acknowledges PURGE has lower performance on most metrics (notably in the Forget and Utility sets) but justifies this with token efficiency. To make a valid efficiency claim, additional experiments are necessary: 1) What is the performance of NPO and DPO when restricted to the same small token budget as PURGE? 2) Conversely, what is PURGE's performance when trained on the full token budget used by NPO?

4. Potential Over-specialization to the RWKU Benchmark: The proposed algorithm appears highly tailored to the specific scenarios of the RWKU benchmark, which raises doubts about its generalizability. To address this, the authors must demonstrate the method's applicability and superior performance on at least one other widely-used unlearning dataset, such as TOFU [6] or MUSE [7].


[1] LLM Unlearning via Loss Adjustment with Only Forget Data, ICLR 2025

[2] Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond, ICLR 2025

[3] A Closer Look at Machine Unlearning for Large Language Models, ICLR 2025

[4] Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs, ICLR 2025

[5] Unified Parameter-Efficient Unlearning for LLMs, ICLR 2025

[6] TOFU: A Task of Fictitious Unlearning for LLMs

[7] MUSE: Machine Unlearning Six-Way Evaluation for Language Models

### Questions
Please refer to the main Weaknesses section for my major weakness/questions. My minor questions and suggestions are shown below.

1. The paper inconsistently mixes citation styles (e.g., \cite and \citep) throughout the manuscript. The authors need to standardize the citation style for clarity and consistency.

2. Acronyms such as "RT" and "NER" are used in Section 4.1 without prior definition. Please define these acronyms upon their first use to improve readability.

3. The notation $\pi_{\theta_r}$ in Equation 17 (Line 294) is potentially confusing, as 'r' could be mistaken for the "Retain" set $\emptyset_R$. Given its use as a reference policy, please consider changing this to a less ambiguous notation.

4. Figure 3, which illustrates performance scaling with model size, only shows results for PURGE. To substantiate the claim of superior scalability, this figure should ideally include comparative results for a key baseline, such as NPO.

5. Figure 4 presents results against 9 adversarial scenarios, but the caption lacks a description or reference for what these scenarios entail. 

I look forward to the authors' responses to the points raised in the Weaknesses and Questions sections. If there are any misunderstandings on my part, please correct them. I am open to changing our score based on the discussion.

### Soundness
1

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
This paper introduces PURGE (Policy Unlearning through Relative Group Erasure), a new method for machine unlearning in large language models that's built on the GRPO framework. The core idea is to treat unlearning as a "verifiable" task, where the model is fine-tuned to suppress mentions of forbidden concepts using an intrinsic binary reward signal (1 if no forbidden tokens appear, 0 otherwise). They construct a synthetic forget corpus by probing the model itself and extracting entities with GPT-4, then use GRPO to optimize the policy without needing external reward models. The authors provide theoretical guarantees like geometric decay in forbidden-token probabilities and KL-divergence bounds for utility retention. Empirically, they evaluate on the RWKU benchmark, claiming up to 46x fewer tokens per target than SOTA, +7.32 fluency improvement, +12.02 adversarial robustness, and 11% unlearning effectiveness while keeping 98% utility. Overall, it's positioned as a more efficient, safe, and scalable alternative to existing unlearning approaches like gradient ascent or preference optimization.

### Strengths
- The paper's structure is logical and accessible, with clear transitions from motivation to methods and results.
- Introducing reinforcement learning to the LLM unlearning domain is relatively novel, bringing fresh perspectives that could inspire future work and contributing to the paper's innovative quality.
- Empirical results are comprehensive and well-supported, including breakdowns across multiple RWKU sub-tasks with quantitative comparisons to baselines, enhancing the paper's credibility and showing rigorous validation.
- Theoretical contributions add depth, with proofs providing explicit bounds on key metrics like leakage and utility, which strengthens the paper's academic value beyond typical empirical-focused submissions.

### Weaknesses
- The effectiveness of the proposed reward model is questionable, as relying solely on extracted entities may not compactly represent the knowledge to be forgotten; in some cases, knowledge has many variants (e.g., complex concepts), while in others, like copyright protection, only specific text needs forgetting without erasing concepts, potentially leading to over-penalization based on entity presence alone.
- To maintain training stability, GRPO includes a clipping mechanism that limits policy changes during RL training, making it difficult to ensure complete erasure of forget content (e.g., driving prediction probabilities close to 0).

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
