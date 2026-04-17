# Review

## Summary
This paper proposes a method for the task of few-shot novel view synthesis (NVS). Specifically, the authors propose a two-stage generative completion-based approach to enhance the scene completion in few-shot NVS. In the first stage, the authors introduce a generative point cloud completion-based strategy to generate complementary points for initializing the 3D Gaussian Splatting (3DGS). In the second stage, they propose a generative pseudo view completion-based strategy to synthesize pseudo views for optimizing the 3DGS. The proposed method is evaluated on the LLFF, DTU, and Shiny datasets, demonstrating superior performance compared to baseline methods.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The idea of using generative models for scene completion in few-shot NVS is interesting.
2. The proposed method achieves state-of-the-art performance on three benchmark datasets.

## Weaknesses
1. The proposed method is complicated and contains many hyperparameters, which makes it difficult to understand and follow. For example, there are five hyperparameters in the GCGI stage and six in the GCGO stage. How were these values determined? How sensitive is the proposed method to these hyperparameters? Providing ablation studies on these hyperparameters would help clarify their impact and justify their selection.
2. The authors propose many strategies to mitigate the hallucination of generative models. However, it is unclear whether these strategies are effective. For example, the authors propose a generative consistency loss to attenuate the impact of generative hallucination. However, it is unclear how this loss affects the final results. Providing qualitative and quantitative results demonstrating the effectiveness of these strategies would strengthen the paper.
3. The authors propose a generative point cloud completion-based strategy to generate complementary points for initializing the 3DGS. However, it is unclear whether the generated points are accurate and contain meaningful information. Providing visualizations or evaluations of the generated points would help validate their quality and usefulness.
4. The authors propose a generative pseudo view completion-based strategy to synthesize pseudo views for optimizing the 3DGS. However, it is unclear whether the synthesized pseudo views are accurate and contain meaningful information. Providing visualizations or evaluations of the synthesized pseudo views would help validate their quality and usefulness.
5. The proposed method contains many modules and stages, which makes it computationally expensive. Providing a comparison of the running time and memory usage of the proposed method with baseline methods would help clarify its efficiency.

## Questions
Please refer to the Weaknesses section.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4