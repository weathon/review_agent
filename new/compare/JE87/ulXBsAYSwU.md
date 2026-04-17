# Review

## Summary
The authors propose MolMiner, a fragment-based generative model for molecular design that is capable of generating molecules conditioned on up to twelve different molecular properties. To achieve this, the authors introduce a novel, order-agnostic generation procedure that is able to sample molecules from the fragment vocabulary in any valid order. Additionally, the authors introduce a symmetry-aware protocol for fragment attachment that ensures that fragment symmetries are respected during generation. Finally, the authors incorporate spatial information into the attention mechanism via a global attention bias, which allows the model to be geometry aware. The authors demonstrate that MolMiner achieves calibrated conditional generation for most of the twelve properties.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper is well-written and easy to follow.
- The authors introduce a novel, order-agnostic generation procedure that allows the model to sample molecules in any valid order. This increases the diversity and flexibility of possible rollouts.
- The authors introduce a symmetry-aware protocol for fragment attachment that ensures that fragment symmetries are respected during generation.
- The authors incorporate spatial information into the attention mechanism via a global attention bias, which allows the model to be geometry aware.
- The authors demonstrate that MolMiner achieves calibrated conditional generation for most of the twelve properties.

## Weaknesses
- The authors do not compare MolMiner to any other conditional molecular generation models, which makes it difficult to assess how well MolMiner performs compared to other state-of-the-art models.
- The authors do not provide any examples of generated molecules or their corresponding properties, which would provide more insight into the model's performance and the types of molecules it can generate.
- The authors do not provide any information on the computational requirements or runtime of MolMiner, which would be useful for assessing its practicality and efficiency.

## Questions
- How does MolMiner compare to other state-of-the-art models for conditional molecular generation in terms of performance and capabilities?
- Can the authors provide examples of generated molecules and their corresponding properties to illustrate MolMiner's performance and capabilities?
- What are the computational requirements and runtime of MolMiner, and how do they compare to other models in the field?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4