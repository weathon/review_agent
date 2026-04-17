# Review

## Summary
The paper explores how much chain-of-thought (CoT) prompting LLMs really need to solve physics problems accurately. The authors propose a deletion-based probing method to evaluate the faithfulness of CoT in physics reasoning tasks, aiming to distinguish between models that genuinely use CoT and those that merely use it as scaffolding. They find that LLMs can maintain high accuracy even with significant deletions of intermediate reasoning steps, suggesting a reliance on internalized physics knowledge rather than the explicit CoT provided. This behavior, termed “cramming,” highlights concerns about the faithfulness of CoT as a reflection of true reasoning processes in LLMs.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The paper addresses a critical gap in evaluating LLM reasoning by focusing on the faithfulness of CoT in physics problem solving, a domain where intermediate steps can be objectively evaluated.
- The deletion-based probing method is novel and allows for a direct assessment of whether LLMs genuinely rely on CoT or can reconstruct reasoning traces internally.
- The paper provides robust experimental results across multiple physics benchmarks and open-source models, revealing consistent patterns in CoT reliance and compensatory behaviors.

## Weaknesses
- The study is limited to physics reasoning tasks, which may not fully represent other complex reasoning domains (e.g., mathematics, commonsense reasoning). The generalizability of the findings remains uncertain.
- The paper does not examine the potential impact of model architecture, training data, and other factors on the observed CoT faithfulness, which could provide valuable insights into the conditions under which CoT reliance emerges.
- While the deletion experiments are systematic, they are restricted to three deletion strategies. Exploring a wider variety of deletion techniques could offer a more comprehensive understanding of how LLMs compensate for missing information.

## Questions
- Could the authors elaborate on how their findings might generalize to other reasoning domains beyond physics, such as mathematics or commonsense reasoning? Have any preliminary investigations been conducted in these areas?
- The paper notes that models often reconstruct deleted content in the final answer, which the authors term “cramming.” Could the authors provide more insight into the potential mechanisms underlying this behavior? Is it possible that the models are simply regurgitating memorized knowledge rather than genuinely reasoning through the problem?
- The study focuses on open-source models (Magistral, Phi-4, Qwen-A3B). Do the authors believe that similar findings would hold for closed-source models? Have any preliminary experiments been conducted in this direction?
- The authors use Bag-of-Words metrics (Jaccard similarity and Manhattan distance) to measure information overlap between original CoT traces and reconstructed answers. Have the authors considered using more sophisticated natural language processing techniques that might capture semantic similarity more effectively?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4