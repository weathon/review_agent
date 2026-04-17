# SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance–Diversity Data Selection

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Fine-tuning large language models often undermines their safety alignment, a problem further amplified by harmful fine-tuning attacks in which adversarial data removes safeguards and induces unsafe behaviors. 
We propose **SPARD**, a defense framework that integrates **S**afety-**P**rojected **A**lternating optimization with **R**elevance-**D**iversity aware data selection. 
SPARD employs SPAG, which optimizes alternatively between utility updates and explicit safety projections with a set of safe data to enforce safety constraints.
To curate safe data, we introduce a Relevance–Diversity Determinantal Point Process to select compact safe data, balancing task relevance and safety coverage. 
Experiments on GSM8K and OpenBookQA under four harmful fine-tuning attacks demonstrate that SPARD consistently achieves the lowest average attack success rates, substantially outperforming state-of-the-art defense methods, while maintaining high task accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
A safety-projected Alternating optimization with relevance-diversity aware data selection scheme is proposed to address harmful fine-tuning attacks.

### Strengths
1. Paper is extremely well written. I can understand the solution very easily. 

2. Two components, i.e., safety data selection and alternating safety projection is proposed. 

3. Compared to a similar work, Lisa, the optimization problem is transferred from a multi-task loss to a constraint problem, and this problem is solved by a projected gradient method. This transformation is crucial and might inspire newer ideas on design defense.  To my best knowledge, this is the first (among a few concurrent work) to explore such projection methods. It is a solid work in my understanding. 

4. The projection problem in Eq. (3) and the resulted projection step in (4) is elegant and makes perfect sense to me.

### Weaknesses
1. More discussion on penalty and constraint-based problem formulation.

* Line 3, Page 3, I think Lisa [1] and SafeGrad [2] both explore the penalty problem formulation. I suggest the authors to add them into the citation with (Bianchi et al., 2023). 

*  I can't agree the statement "Unlike penalty-based approaches, which require careful tuning of λ and offer no guarantee of feasibility, SPAG yields a closed-form projection step that enforces the constraint up to the first order."  The constraint problem and the penalty problem are sort of identical in some sense. For the constraint problem, you also need to carefully tune $\tau$, which is identical to tune $\lambda$ in the penalty problem. Please consider to remove such an unjustified statement, which is biased towards penalty methods. I tend to believe that the two methods are just two alternative way to express the same fundamental problem (trade-off between two losses), but we should not be biased towards one alternative only by simply looking at problem formulation. But I do feel that studying the constraint-based alternative is important as it gives alternative ways to design new methods and potentially these new methods can give better empirical performance.   

[1] Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack

2. A concurrent work should be discussed.  Stemming from the gradient conflict for the penalty problem, SafeGrad[2] derives a very similar projection method. However, because the two update rule are derived from different problems, it appears to me the fundamental projection rule seems to be not identical. I suggest the authors to:

* Discuss the similarity and difference between SafeGrad and SPAG. 
* Do an experiment to compare SafeGrad and SPAG empirically. I think this is interesting because the two projection rule are derived based on different problems (penalty/constraint) and let's see which method works better empirically.  

[2] SafeGrad: Gradient Surgery for Safe LLM Fine-Tuning

3. The finding of "Relevance between safety data and fine-tuning data Improves Safety" has already been covered by several existing literature but the authors did not cite and discuss them. I urge the authors to properly credit the following works, otherwise it will vitiate our research community.

* This finding is first concurrently covered by [3][4] in two ICLR2025 submissions.  Then it is also covered by [5].  Particularly, a highly relevant work [3] also discusses similarity and diversity metric. They use a similar similarity and diversity metric to measure the  a  subset of dataset.  However, this paper lacks proper credit to [3] given the relevance of these two papers. I strongly suggest the authors to properly credit [3]. Otherwise I can't recommend acceptance of this paper.   Also, could you discuss whether you have some novelty contribution over [3]? If you have, could you perform experiments to show that how your method compare against [3]? 

* With that said, it seems that  the finding "Notably, the curve also shows that ASR rises again (to 16.6%) when the selected samples are too similar to the fine-tuning data " is new to me. 

* A recent work [6] explores the safety sample curation problem. They explore an optimization-based solution for curate safety data. The solutions are not in the same direction with the  cosine similarity criterion explored in [3][4]. I suggest the authors compare with [6] to see which safety data curation method is better.   

[3] Your Task May Vary: A Systematic Understanding of Alignment and Safety Degradation when Fine-tuning LLMs  (ICLR2025 submission)

[4] Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models (ICLR2025 submission)

[5]  When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment

[6] Pharmacist: Safety Alignment Data Curation for Large Language Models against Harmful Fine-tuning

4. In addition to the above highly relevant papers, there are many more papers on harmful fine-tuning that are not discussed in this paper:

Scaling Trends for Data Poisoning in LLMs

Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models

Virus: Harmful Fine-tuning Attack for Large Language Models Bypassing Guardrail Moderation 

No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data

Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety 

Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents 

Eliciting Harmful Capabilities by Fine-Tuning on Safeguarded Outputs

Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs

Vaccine: Perturbation-aware alignment for large language model aginst harmful fine-tuning

Tamper-Resistant Safeguards for Open-Weight LLMs

Booster: Tackling harmful fine-tuning for large language models via attenuating harmful perturbation

Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation 

Self-Destructive Language Model

CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning

Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning

LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning

Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning

Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence

SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection

Safety alignment should be made more than just a few tokens deep

SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation

Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs 

Shape it Up! Restoring LLM Safety during Finetuning 

Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization 

Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation 

AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin

Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment

GradShield: Alignment Preserving Finetuning

A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space 

Detecting Instruction Fine-tuning Attack on Language Models with Influence Function

Antidote: Post-fine-tuning safety alignment for large language models against harmful fine-tuning 

Locking Down the Finetuned LLMs Safety 

Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation

Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets 

Navigating the safety landscape: Measuring risks in finetuning large language models

ESTIMATING WORST-CASE FRONTIER RISKS OF OPEN-WEIGHT LLMS

Detecting Adversarial Fine-tuning with Auditing Agents

Fundamental Safety-Capability Trade-offs in Fine-tuning Large Language Models

When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment 

There may be more relevant works (I just list above some more recent work), and I suggest the authors to read and discuss all of the relevant works when writing the paper.

### Questions
1. When limiting the projection step size $\alpha$ with  (Schulman et al., 2015), will the projection still make sure that constraint in (3) strictly holds?  Did you have some results for ablation when we does not limit the step size $\alpha$ with $\eta_{safe}$?

Please address the concerns and feel free to leave me a comment in the rebuttal phase. I enjoy reading this paper overall, although I have serious concern on your Section 3.2, which do not credit properly on [3]. I will consider adjusting my score based on the rebuttal.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a framework named SPARD to defend LLMs against harmful fine-tuning attacks. The proposed solution consists of two key components: (1) Safety-Projected Alternating Gradient (SPAG). This is a principled optimization strategy. Instead of using safety data as a soft penalty (which is common in other methods), SPAG formulates the problem as a safety-constrained optimization. It alternates between a standard utility update on the fine-tuning data and an explicit, closed-form projection step that pushes the model parameters back into a safe region defined by a curated set of safe data. (2) Relevance-Diversity Data Selection: the authors show that safety data must be both relevant to the downstream task and diverse. To achieve this balance, they propose a Relevance-Diversity Determinantal Point Process (DPP) to incorporate both a task-relevance quality score and an intrinsic diversity measure. Experiments conducted on two LLMs (Qwen-2.5-7B and LLaMA-3.2-3B) , two downstream tasks (GSM8K and OpenBookQA) , and four different harmful attack datasets demonstrate that SPARD consistently achieves the lowest ASR and HS. Notably, it does this while maintaining high task accuracy.

### Strengths
- The combination of SPAG and Relevance–Diversity DPP is a novel and principled solution to address harmful fine-tuning attacks. 
- SPARD consistently outperforms baseline methods across multiple datasets and model architectures.
- The paper provides extensive experiments, sensitivity analyses, and comparisons with baselines, demonstrating the effectiveness and generalizability of SPARD.

### Weaknesses
- [Minor] Details on generating the embeddings are not directly available until Section 4.1, making people slightly confused at the beginning. 
- Would the cost of applying DPP increase with the dataset size as it seems to compute pairwise similarity? how would it scale to larger datasets if that's the case? 
- Computational cost: if the threshold in Algorithm 1 is passed, then a second backward is required. This will impose greater computational cost and longer running time. It would be great if the authors can provide statistics on how frequent this will trigger. 
- The method's success appears to be highly dependent on several new hyperparameters that must be carefully tuned, but the paper provides limited guidance on how to set them universally.

### Questions
See above.

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
4

### Summary
This paper presents SPARD, a safety projection approach based on relevance-based diversity-aware data selection. The  relevance-based diversity-aware data selection is directly achieved by using the existing DDPs approach (Determinantal Point Processes) which ranks the top subset that contains both individually informative and mutually diverse data items/elements.

### Strengths
The idea of utilizing the existing DDPs approach to provide the relevance-based diversity-aware data selection is simple and interesting. 

The safety projected alternating gradient (SPAG) is an extension of Bianchi et al 2023's approach, aiming to improve the tuning efforts of setting the average safety loss parameter and the penalty parameter. 

The idea of extension is to perform fine-tuning on the fine-tuning dataset first (i.e., utility driven update) and then perform projection to encode the updated parameters into the half-space defined by C^+, which satisfies safety constraints. 

The paper provides experimental comparison using two LLMs: Qwen-2.5-7B-instruct and LLaMA-3.2-3B-instruct and compared to three existing safety guardrail methods in addition to the SFT baseline.

### Weaknesses
The paper can benefit from providing clear elaboration on the following aspects.

(1) Although the idea of utilizing the existing DDPs approach to provide the relevance-based diversity-aware data selection is simple and interesting, given that the quality of selected subsets by DPPs is based on the balancing of the two criteria: individually informative and mutually diverse, the paper should provide discussion to elaborate on how these two criteria are semantically measured since informative is context relevant and mutually diverse is also context driven (e.g., agreement diversity or disagreement diversity ... ) and why DDPs based  data selection will help mitigating harmful fine-tuning.

(2) The proposed approach relies on task-relevant safe samples to provide safety constraints. It is a very strong assumption. A discussion on how the proposed approach responds when the task relevant safe samples are of varying quality and volume. 

(3) The experimental results could benefit from more detailed discussion to elaborate on the boundary cases. For example, Table 1 has shown that there are two out of four datasets the proposed approach did not outperform existing safety guardrail methods, like LISA. The intuitions behind the proposed approach vs LISA (Huang et. al. 2024b) and vs. SafeInstr (Bianchi et.al 2023) should be provided. 

(4) Similar questions also apply to Table 2, Table 3 and Table 4. 

(5) Can you use Figure 6 style of comparison on GSM8K to show those cases where SPARD performance is weaker compared to existing methods and analyze why.

### Questions
See the weakness section

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
2

### Summary
This paper proposes SPARD, a defense framework against harmful fine-tuning attacks on aligned large language models (LLMs). The framework combines two complementary ideas:
(1) Safety-Projected Alternating Gradient (SPAG) — an optimization procedure that alternates between utility-driven fine-tuning steps and explicit safety projection steps to enforce safety constraints in closed form;
(2) Relevance–Diversity Determinantal Point Process (DPP) — a principled method to select a compact subset of “safe” data that balances task relevance and behavioral diversity.

Experiments on GSM8K and OpenBookQA under multiple harmful fine-tuning settings (BeaverTails, LatHarmful, etc.) demonstrate that SPARD significantly reduces attack success rate (ASR) while maintaining downstream utility. The method consistently outperforms strong baselines such as SafeInstr, SafeLoRA, and Lisa.

### Strengths
1. The combination of task relevance and diversity within a determinantal point process framework is well-motivated. It improves upon prior ad-hoc or random selection strategies, and empirical analysis (Figure 2) convincingly shows that both factors matter for robust defense.
2. The mathematical formulation of SPAG (Eq. 2–4) and its derivation (Appendix A) are precise and well-presented. The trust-region stabilization strategy further enhances practical robustness.

### Weaknesses
- The proposed Safety-Projected Alternating Gradient (SPAG) method is presented as a novel optimization framework. However, its core mechanism is well-established in classical constrained optimization literature (e.g., projected gradient descent, proximal updates, or trust-region projection). While the adaptation to LLM safety alignment is interesting and practically meaningful, the methodological novelty of SPAG itself appears limited. The paper would benefit from a clearer distinction between conceptual novelty (application to harmful fine-tuning) and methodological novelty (new optimization formulation).
- Although the DPP-based selection is implemented with a greedy approximation, the paper does not provide a clear discussion or analysis of its computational efficiency.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
