# Review

## Summary
This paper compares the learning dynamics of state-space models (SSMs) and Transformers on associative recall and copying tasks, revealing optimization instabilities in SSMs that affect their performance and scalability. It highlights architectural differences, particularly in scaling behaviors, and shows that SSMs require specific tuning to match Transformer robustness.

## Soundness
3

## Presentation
2

## Contribution
2

## Strengths
1. The paper provides extensive experimental validation through over 3,000 runs and approximately 20,000 GPU hours, demonstrating a high level of experimental rigor and thoroughness. The authors conduct detailed ablation studies and explore various architectural variations, which adds to the robustness of the findings.

2. The authors identify critical optimization challenges faced by SSMs, particularly the narrow learning rate window, and its impact on performance. This finding has important practical implications for the training of SSMs and contributes to the understanding of their limitations.

3. The paper shows that SSMs benefit more from increased width compared to Transformers, which suggests architectural trade-offs for sequence models. This insight could inform the design of future models.

## Weaknesses
1. The paper's focus on synthetic benchmarks may limit the generalizability of the findings to real-world applications. While the authors acknowledge this limitation, it would be valuable to explore how these dynamics play out in more complex tasks or datasets. This is particularly important given that the paper aims to understand fundamental differences between SSMs and Transformers.

2. The paper does not provide a formal theoretical explanation for the observed optimization brittleness in SSMs. A deeper theoretical analysis could help in understanding the underlying mechanisms and potentially suggest solutions.

3. The paper's main finding about the narrow learning rate window for SSMs is an empirical observation. It would be stronger if accompanied by a theoretical analysis of why this phenomenon occurs.

## Questions
1. How do the observed optimization dynamics and scaling behaviors in SSMs and Transformers affect the architectural design choices for real-world applications, such as language modeling or other sequential tasks?

2. The paper mentions that DeltaNet exhibits more stable training dynamics. Can you provide more insights into the specific architectural features of DeltaNet that contribute to this stability, and how these insights could inform the design of future SSMs?

3. The authors note that Transformers show a loss drop resembling induction head formation during training, even in a single-layer configuration. Can you elaborate on the implications of this finding for our understanding of Transformer architecture and its inductive biases?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4