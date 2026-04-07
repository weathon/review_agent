## Summary
This paper introduces Forget-to-Focus (F2F), a two-stage protocol that first applies targeted unlearning on a “forget” set of general-domain data to remove irrelevant pretraining knowledge, then fine-tunes on a domain-specific dataset. The method consistently improves performance over standard fine-tuning across coding, medical, and mathematical tasks for models ranging from 0.6B to 72B parameters, accompanied by analysis showing representational shifts and better calibration.

## Strengths
- **Novel and impactful repurposing of unlearning**: The work reframes machine unlearning from a privacy tool to a deliberate intervention for domain specialization, demonstrating clear empirical gains across diverse domains (e.g., +32.5% pass@1 on HumanEval for Qwen-0.6B) and model scales.
- **Rigorous and multi-faceted evaluation**: Experiments span multiple model families, sizes, and domains, with in-depth mechanistic analysis via centered kernel alignment, SVCCA, Fisher information, PCA shifts, and calibration studies, providing convincing evidence that unlearning reshapes representations and reduces overconfidence.

## Weaknesses
- **Heuristic and under-specified forget set construction**: The forget sets (BC-Select, BC-Cosine) rely on manual curation or cosine similarity without clear criteria for domain-irrelevance, making the method difficult to reproduce and generalizing uncertainly to new domains.
- **Limited calibration evidence**: Improved calibration is shown only on medical QA (MedMCQA); without results from coding and mathematical tasks, the claim that F2F enhances reliability broadly is not fully supported.
- **Oversimplified theoretical analysis**: The convex linear model analysis (Proposition and Corollary) is a severe simplification of non-convex LLM optimization and does not meaningfully justify why the gradient ascent/descent procedure works in practice.
- **Inconsistent experimental setups**: Comparisons across model scales are confounded by varying fine-tuning strategies (e.g., Qwen-72B uses 50% of the dataset and QLoRA, while smaller models use full data and SFT), undermining fair assessment of F2F’s scalability.
- **Representation analysis lacks performance linkage**: While CKA and SVCCA show representational shifts after unlearning, the paper does not correlate these changes with accuracy gains, leaving it unclear which geometric alterations are beneficial for specialization.

## Nice-to-Haves
- Control experiments matching total optimization steps between F2F and standard fine-tuning to isolate the effect of unlearning from additional training.
- Broader evaluation on general-knowledge benchmarks (e.g., MMLU) to comprehensively verify that core capabilities are preserved.
- A more principled approach to constructing forget sets, such as using gradient-based influence scores to identify harmful knowledge automatically.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Formatting artifacts from PDF parsing**: These are extraction issues, not problems with the paper.
- **Demand for exhaustive baseline comparisons**: The paper includes standard fine-tuning, DAPT, LoRA, and CurlLoRA; requiring all possible parameter-efficient methods is scope creep.
- **Nitpicks about abstract omissions**: The abstract summarizes key contributions appropriately; missing caveats is not a substantive flaw.
- **Claim that theoretical section is an “afterthought”**: Subjective and not factually incorrect; the section is provided as intuitive motivation.

## Novel Insights
None beyond the paper’s own contributions.

## Suggestions
- Detail the forget set construction process, including the specific Transformer encoder used for cosine similarity, similarity thresholds, and manual curation criteria, to ensure reproducibility.
- Extend calibration analysis (reliability diagrams, ECE) to coding and mathematical benchmarks to substantiate claims about improved reliability across domains.
- Acknowledge the limitations of the convex theoretical analysis and supplement it with empirical observations on optimization dynamics (e.g., loss landscape or gradient conflict) to better ground the method’s intuition.