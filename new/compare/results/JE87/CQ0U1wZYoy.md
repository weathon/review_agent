# Review

## Summary
This paper presents PRISM, a novel framework designed to address the complex issue of mixed noise resulting from both the sensor and the environment. PRISM employs a prompted conditional diffusion approach that integrates compound-aware supervision over mixed degradations with a weighted contrastive disentanglement objective. This enables high-fidelity joint removal of overlapping distortions while also allowing flexible, targeted fixes through natural language prompts. The effectiveness of PRISM is demonstrated across various datasets, including microscopy, wildlife monitoring, remote sensing, and urban weather imagery, where it outperforms state-of-the-art baselines on complex compound degradations, including zero-shot mixtures that were not seen during training.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow. The authors have provided clear explanations of the proposed method, making it accessible to a wide audience.

2. The authors have conducted extensive experiments across various datasets, including microscopy, wildlife monitoring, remote sensing, and urban weather imagery. This wide range of experiments demonstrates the robustness and generalizability of the proposed method.

3. The authors have also performed ablation studies to analyze the contribution of each component of the proposed method, providing valuable insights into the method's performance and the impact of individual components.

## Weaknesses
1. The paper does not discuss the limitations of the proposed method. It would be beneficial to include a section on the limitations, providing insights into the scenarios where the method may not perform optimally.

2. The paper does not mention the computational complexity of the proposed method. Providing information on the computational requirements and efficiency would be valuable for practical applications.

3. The paper could benefit from a more detailed comparison with existing methods. While the authors have provided comparisons with state-of-the-art baselines, a more in-depth analysis of how the proposed method differs from and improves upon existing approaches would strengthen the paper.

## Questions
1. What are the limitations of the proposed method, and in what scenarios does it not perform optimally?

2. What are the computational requirements and efficiency of the proposed method, and how do they compare to existing approaches?

3. Can the authors provide a more detailed comparison with existing methods, highlighting the key differences and improvements of the proposed method?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4