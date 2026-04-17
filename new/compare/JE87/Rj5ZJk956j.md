# Review

## Summary
This paper introduces a new method for analyzing gated neurons in LLMs by computing cosine similarities between their weights, which reveals their read-write functionality. The authors find that early-middle layers are dominated by conditional strengthening neurons, while late layers contain weakening neurons, which have a surprisingly large impact on model behavior despite their small number. The paper also introduces a new analysis method, conditional ablation, to identify specific activations responsible for certain behaviors and demonstrates that part of the impact of weakening neurons is due to a mechanism involving negative gate values.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The paper introduces a novel method for analyzing gated neurons in LLMs by computing cosine similarities between their weights, providing new insights into the inner workings of these models.
2. The authors identify a class of neurons called weakening neurons that have a surprisingly large impact on model behavior despite their small number, and demonstrate that they activate often and influence various metrics.
3. The paper introduces a new analysis method, conditional ablation, which allows finding specific activations of a neuron responsible for certain behaviors, and applies it to weakening neurons to understand their impact.

## Weaknesses
1. The paper focuses on a specific type of activation function (gated activation function) and may not be applicable to other activation functions used in LLMs.
2. The authors rely on cosine similarities and activation frequencies to classify neurons, which may not fully capture their functionality and could lead to incorrect classifications.
3. The paper does not provide a detailed analysis of how the weakening neurons work together with other neurons to influence model behavior, and how they evolve during model training.

## Questions
1. How do the weakening neurons work together with other neurons to influence model behavior? Are there any specific patterns or synergies?
2. How do the weakening neurons evolve during model training? Are there any changes in their number, activation frequency, or functionality?
3. Can the proposed analysis method be applied to other types of neurons or activation functions used in LLMs? If so, what are the challenges and limitations?
4. How can the findings about weakening neurons be used to improve the interpretability or performance of LLMs? Are there any potential applications or implications?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4