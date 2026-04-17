# An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
In this paper, we introduce a systematic framework beyond conventional method to assess LLMs’ mathematical‑reasoning robustness by stress‑testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMS to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 17 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 49 % on the originals but drops by 4 percentage points on surface variants, and by 10.5 percentage points on core-step-based variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies the robustness of large language models (LLMs) in mathematical reasoning. The authors propose a Generalization–and–Perturbation (GAP) framework, which tests models on mathematically equivalent but linguistically or parametrically transformed problems. Based on this, they construct PutnamGAP, a large-scale benchmark built from 85 years of Putnam Competition problems, each augmented with five mathematically equivalent variants. Experiments on 18 commercial and open-source models show substantial accuracy drops under surface and parametric transformations, revealing that even advanced models rely heavily on surface cues rather than genuine mathematical understanding.

### Strengths
1. Robustness in mathematical reasoning is an increasingly important and underexplored direction. The paper tackles this with a clear motivation and a well-defined experimental setup. 

2. The authors introduce five transformation types (four surface-level renamings and one parametric rewrite), providing a systematic way to probe reasoning robustness. 

3. The evaluation spans 18 models and demonstrates consistent degradation under mathematically equivalent perturbations, validating the effectiveness of the proposed benchmark. 

4. The paper also categorizes error types (symbol confusion, step omission, logic hallucination), offering further insight into model failure patterns.

### Weaknesses
1. The experimental analysis is relatively limited and could be enriched by additional studies. (1) It would be useful to include math-specialized models in the evaluation to see how training objectives or dataset composition influence robustness, and to provide insights into how robustness might be improved. (2) The paper could explore whether specific prompting strategies (e.g., instructing models like O1 to pay attention to variable names or perform meta-reasoning) could help defend against these perturbations. (3) It would also be valuable to test whether training on the generated variants can actually improve model robustness. This could show whether the proposed dataset is not only diagnostic but also useful for enhancing reasoning stability. 

2. The paper lacks sufficient discussion and comparison with related robustness benchmarks such as GSM-Symbolic, GSM-Plus, and MathCheck, which employ similar perturbation-based methods to test reasoning robustness. A clearer comparison with these works would better situate the contribution of this paper.

### Questions
none

### Soundness
2

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
3

### Summary
This paper introduces a new framework, **Generalization-and-Perturbation (GAP)**, to systematically evaluate the mathematical reasoning robustness of Large Language Models (LLMs). The authors argue that existing benchmarks are compromised by data contamination and fail to test for true, generalizable reasoning beyond surface-level pattern matching.

The GAP framework addresses this by "stress-testing" models on **mathematically equivalent** variations of advanced math problems. The paper's contributions are twofold:

1.  **The GAP Methodology:** A novel procedure for generating two classes of problem variants: **surface renames** ($\mathcal{T}_{surf}$), which alter linguistic and symbolic representations (e.g., variable names) , and **parametric rewrites** ($\mathcal{T}_{para}$), which modify the problem's constants and parameters while preserving its core logical structure (termed "Kernel Variant").
2.  **The PutnamGAP Dataset:** A large-scale instantiation of this framework using 1,051 unique Putnam Competition problems from 1938-2024. The authors generate five variants for each original problem (four surface, one kernel), resulting in a 6,306-item benchmark.

Using this benchmark, the authors evaluate 18 prominent commercial and open-source LLMs. The key finding is that all models, including state-of-the-art systems, exhibit a **sharp degradation in performance** when tested on these equivalent variants. For example, the O3 model's accuracy drops by 4.7 percentage points (pp) on surface variants and 12.9 pp on parametric variants, demonstrating that their reasoning capabilities are brittle and sensitive to non-mathematical perturbations.

### Strengths
* **Originality:** The paper's originality is high. While robustness testing is not new, the GAP framework's focus on **mathematical equivalence**  is a crucial distinction from prior work on contrast sets or perturbations that change the problem's substance. The specific methodology, distinguishing between surface-level ($\mathcal{T}_{surf}$) and deep-structural ($\mathcal{T}_{para}$) perturbations, provides a novel and insightful way to disentangle different reasoning failures.
* **Quality:** The work is executed with exceptional rigor and quality.
    * **Methodology:** The variant-generation pipelines are sophisticated and well-designed. The use of LLMs to propose replacements for surface renames (Figure 2)  and the complex multi-stage process for "Kernel Variant" synthesis (Figure 3).
    * **Dataset:** The creation of **PutnamGAP** is a contribution on its own. By sourcing 85 years of challenging Putnam problems, the authors have created a high-difficulty, long-lasting benchmark that is unlikely to be "solved" or fully contaminated in the near future.
    * **Metrics:** The authors propose a new, principled **robustness metric $\hat{R}(e,h)$** . This is a strong theoretical contribution, moving beyond simple accuracy ratios to an item-aware, normalized score that better captures catastrophic failures.
    * **Experiments:** The evaluation is comprehensive, covering 18 models , and the auto-grading system is transparently described.
* **Clarity:** The paper is written with outstanding clarity. The motivation is clear , the methodology is explained precisely with helpful diagrams , and the results are presented and analyzed in a way that is easy to follow (Section 5) . The appendices are thorough and provide all necessary details for reproducibility.
* **Significance:** It provides a concrete, actionable methodology for addressing two of the most critical challenges in AI evaluation: **benchmark data leakage**  and **measuring true generalization** vs. "memorized textual surface forms". The GAP framework is general and could be applied to other reasoning domains

### Weaknesses
The paper's primary weakness is that it is more descriptive than diagnostic. It excels at *identifying* and *quantifying* the robustness failure but offers limited insight into *why* it occurs or how to fix it.

* **Analysis is Descriptive, Not Diagnostic:** The central finding—that LLM performance drops on perturbed inputs—is, while well-proven, not entirely surprising. The paper stops short of a deep analysis of these failures.
    * The error taxonomy (Section 5.3)  is a good start, but it's still a high-level description. The paper notes that "Logic Hallucination" dominates , but it doesn't explore *why* a simple variable rename (e.g., GS)  or a parameter change (KV)  would cause a *logical* failure, rather than just a "Symbol Confusion" error.
    * The paper misses an opportunity to correlate robustness with model architecture, scale, or training data. Why do some models (gemini-2.5-pro, grok4) suffer a ~15pp drop on KV, while others (gpt-4.1) see a "smaller" ~10pp drop?. A deeper dive into model-specific failures would be highly valuable.
* **Prescriptive Suggestions are Speculative:** The paper's implications section (6.2)  and Appendix F  rightly suggest using the GAP framework for data augmentation and curriculum fine-tuning. However, this is presented as a suggestion. The paper would be much stronger if it included even a small-scale experiment demonstrating this. Proving that fine-tuning on GAP-generated variants *improves* robustness on a held-out test set would "close the loop" and demonstrate the framework's utility for *solving* the problem, not just measuring it.
* **Minor: Potential Grader-Tester Contamination:** The auto-grader relies on an LLM (O3) , which is also used in the variant generation process. While the authors report high human-verified precision (>97%), there is a potential risk that the grader model shares the same sensitivities as the models being tested. For instance, could a "Descriptive_Long_Misleading" (DLM) variant confuse the grader itself? This is not fully addressed.

### Questions
1.  **On the Nature of the Findings:** The core finding that performance drops on perturbed data is expected. What do the authors consider the most *surprising* or *non-obvious* result from their experiments? Is it the sheer magnitude of the drop on Kernel Variants (KV)? Or is it the fact that some models *improve* on the "Descriptive Long" (DL) variant, suggesting the original problem phrasing was somehow suboptimal?

2.  **On Deeper Diagnosis:** The paper does a fantastic job quantifying the robustness gap. Can you provide more qualitative insight into *why* these failures occur? For instance, when a model fails on a KV problem, does it fail at the very first step (e.g., setup), or does it successfully follow the original reasoning structure for a while before derailing? A qualitative analysis of *where* in the solution a top model like gemini-2.5-pro (which drops ~15pp on KV ) goes wrong would be invaluable.

3.  **On Closing the Prescriptive Loop:** The paper suggests using GAP for curriculum fine-tuning. This is a key implication. Have the authors run any preliminary experiments to show that training on GAP-generated variants can improve robustness on a held-out set of variants?

4.  **On the Appendix F.5 Observation:** In Appendix F.5, you make a fascinating observation that "if prompted carefully... to rename perturbed variable names back into normal names," models showed "rebounds on performance." This seems to imply the failure is in correctly *encoding* the perturbed symbols, not in the *core reasoning* module. Could you elaborate on this? Does this suggest that the robustness gap for surface variants could be largely mitigated by a simple pre-processing/prompting layer?

5.  **On the Auto-Grader:** The auto-grader also uses an LLM. How can you be confident that the grader is not susceptible to the same perturbations as the models being tested? For example, in your 10% human audit, did you specifically check if the grader's accuracy was lower on the DLM or GS variants compared to the original problems?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors introduce  a methodology to evaluate the robustness of LLMs in mathematical reasoning through semantics-preserving perturbations. They call the framework GAP (generalize and perturb). On the surface, the idea is to generate 1) superficial variable renaming patterns and 2) scenario changes, without changing the core semantics or the final answer (/proof steps). Authors instantiate this framework as PutnamGAP, a large-scale benchmark (6.3k) built from 85 years of William Lowell Putnam Competition problems, each expanded into five mathematically equivalent variants. An interesting part is the way the accuracy drops are calculated using "robustness metric", which attempts to cater to the following statsitical issues: i) item-aware or large drops should hurt more than many tiny drops, ii)  scale-free across models and iii) differentiable. 
They evaluated 18 LLMs, which suffered significant accuracy drops. They show curriculum learning with randomized symbol identities and numeric parameters help to some extent. They also aim for GAP to be constant source of "leak-proof" test data generation framework.

### Strengths
1. The benchmark seems extensive, covering diverse categories, and based on high-level mathematical problems. The important contribution seems to be the evaluation metric. (some concerns below)

2. Evaluation is extensive. Each of the step seems to have been supported by many experiements.

3. Analysis gives us clear insights. The point with curriculum learning is important, but I did not find too much details in the main paper.

### Weaknesses
The main motivation is well-known. Other work has tried this in different ways: GSM8k_MORE, GSM-symbolic. Even PUTNAM-AXIOM does this, but not in a scalable way. 

The GAP framework seems innovative, though depends a lot of LLMs to do every step. I am unsure how does errors in generation taken care of.

Many important things are in Appendix, which makes the main contributions hard to follow -- like robustness metric details and motivation, curriculum learning training etc.

### Questions
L087: Why so many rounds of review? Are you changing some sort of self-verification paramteres? Or are you updating the original question while reviewing? 
Given so many rounds, wouldn't it be more feasible to do a manual check?

L105: How dependent are you on $ \pi_i $ ? I would have imagined you would have also used $ \pi_i $ during robustness metric calculation, by evaluating the "path". 

While Figures 2 and 3 looks good, it does not help much. I would have rather liked if the question transformations and the data were being shown clearly.

Robustness metric: Why do you need differentiability here? This seems an attempt to combine benchmarking and further training. Is there any reason to combine this? If we use these signals and bake it in the model, doesn't it fail the purpose of being an independent metric. Benchmarking is supposed to be completely independent of the process of model development.

### Soundness
3

### Presentation
3

### Contribution
2
