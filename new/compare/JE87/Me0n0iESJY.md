# Review

## Summary
The paper introduces a benchmark for merging multimodal large language models (MLLMs) across various tasks, including vision-language, audio-language, and video-language models. It proposes a new merging method called OptMerge, which removes noise from task vectors and optimizes the merged vector based on interactions between task vectors, resulting in a 2.48% average performance improvement. The study demonstrates that merging multiple modalities outperforms individual modalities, suggesting a promising approach for building enhanced MLLMs without additional training data.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper proposes a new model merging benchmark specifically designed for multimodal large language models (MLLMs), covering a wide range of tasks including VQA, Geometry, Chart, OCR, and Grounding. This benchmark provides a robust framework for evaluating and advancing MLLM merging techniques.
2. The authors introduce OptMerge, a novel merging method that effectively removes noise from task vectors and optimizes the merged vector based on interactions between task vectors. This approach demonstrates a significant average performance improvement of 2.48%, highlighting its effectiveness in merging MLLMs.
3. The paper provides a thorough analysis of various model merging algorithms on the proposed benchmark, offering a comprehensive understanding of the strengths and weaknesses of different approaches.

## Weaknesses
1. The paper does not explicitly mention the availability of the proposed benchmark for public use, which is crucial for the reproducibility and further development of the research. It is recommended that the authors clarify whether the benchmark will be released as an open-source resource and provide details on any associated licenses or access protocols.
2. The authors only evaluated the proposed OptMerge method on the MLLM merging task. It would be beneficial to assess whether this method is also effective for merging other types of models, such as LLMs, to demonstrate its broader applicability and robustness.
3. The authors evaluated the proposed OptMerge method on relatively small models, such as 7B and 13B. It would be interesting to investigate whether this method scales effectively for larger models, such as those with 30B or more parameters, to understand its potential impact on more powerful architectures.

## Questions
Please refer to the weakness section.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4