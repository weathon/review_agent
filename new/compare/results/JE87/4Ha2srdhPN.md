# Review

## Summary
This paper introduces GRAID, a framework for generating high-quality visual question-answering (VQA) datasets that improve Vision Language Models' (VLMs) spatial reasoning. GRAID creates VQA pairs from 2D bounding boxes alone, avoiding 3D reconstruction errors and generative hallucinations. The authors apply GRAID to generate over 8.5 million VQA pairs across three datasets, achieving a human-validated accuracy of 91.16%, significantly higher than previous methods. Models trained on GRAID data demonstrate improved spatial reasoning generalization, with fine-tuning on certain question types leading to gains on held-out types.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. GRAID is a novel framework that uses only 2D geometric primitives (bounding boxes) to generate spatial VQA data, avoiding errors from 3D reconstruction and generative hallucinations.
2. The paper presents a comprehensive evaluation, including human validation and fine-tuning experiments on multiple VLMs. The results are presented clearly with detailed quantitative analysis.
3. GRAID achieves high accuracy (91.16%) in human validation and demonstrates the generalization potential of models trained on its dataset.

## Weaknesses
1. While the paper acknowledges the limitations of current depth perception models, the depth-related questions in the dataset may still suffer from inaccuracies in real-world applications. The authors could consider introducing a tolerance threshold for depth-based questions to improve robustness.

2. The paper lacks a detailed analysis of the computational efficiency of the GRAID framework. While the authors mention that SPARQ provides speedups, a more comprehensive analysis - including runtime comparisons, scaling properties, and potential bottlenecks - would be valuable for practical applications.

## Questions
1. Could the authors provide more details on the human evaluation process, such as the number of evaluators involved, their qualifications, and the criteria used for assessing the validity of questions and answers?

2. How does the GRAID framework handle ambiguous cases where the spatial relationship between objects might not be clear-cut? For example, if two objects are partially occluded or situated in different planes, how would GRAID generate questions and provide answers in such scenarios?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4