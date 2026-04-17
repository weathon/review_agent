# Review

## Summary
This paper presents a novel approach to multi-identity (multi-ID) image generation, introducing a new large-scale dataset, MultiID-2M, and a corresponding benchmark, MultiID-Bench, for comprehensive evaluation. The authors propose a four-stage training pipeline and a set of tailored loss functions to address the prevalent "copy-paste" artifact issue in multi-ID generation. The paper includes extensive experiments and ablation studies to validate the effectiveness of their method.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The introduction of the MultiID-2M dataset, paired with the MultiID-Bench benchmark, provides a robust foundation for multi-ID image generation research. This dataset is a significant contribution, offering a large-scale, diverse set of images that include multiple identifiable celebrities across various expressions, poses, and hairstyles.

2. The paper proposes a novel training paradigm that effectively addresses the "copy-paste" artifact issue prevalent in existing multi-ID generation methods. By employing a four-stage training pipeline and a combination of tailored loss functions, the authors achieve significant improvements in identity consistency and controllability over the generated images.

3. The authors conduct thorough experiments and ablation studies to validate their approach. The quantitative results demonstrate the superiority of their method over state-of-the-art techniques, and the qualitative results showcase the high quality and controllability of the generated images.

## Weaknesses
1. While the paper presents a novel approach to multi-ID image generation, the technical contribution is relatively limited. The proposed method largely builds upon existing techniques, with the main innovation being the introduction of a new dataset and benchmark. The training pipeline and loss functions are adaptations of known methods. The paper would benefit from a more detailed explanation of how the proposed method significantly advances the state-of-the-art in terms of technical innovation.

2. The paper could provide a more in-depth discussion of the limitations of the proposed method. For instance, it would be valuable to explore scenarios where the method might fail or perform suboptimally. This includes a detailed analysis of the potential impact of the "copy-paste" artifact issue on different identity attributes (e.g., pose, expression, hairstyle) and how the method addresses (or does not address) these challenges.

3. The paper could benefit from a more comprehensive comparison with a broader range of existing methods. While it compares the proposed method with several state-of-the-art techniques, a more extensive comparison with a wider variety of approaches would provide a clearer picture of the method's strengths and weaknesses. This includes both quantitative and qualitative comparisons.

## Questions
1. How does the proposed method differ fundamentally from existing approaches in terms of technical innovation? While the introduction of a new dataset and benchmark is valuable, what specific aspects of the training pipeline or loss functions represent a significant departure from previous methods?

2. Can you elaborate on the scenarios where your method might fail or perform suboptimally? A detailed analysis of these limitations would provide a more balanced view of the method's capabilities and potential areas for improvement.

3. How does your method address (or not address) the "copy-paste" artifact issue across different identity attributes such as pose, expression, and hairstyle? A more in-depth exploration of how the method handles these specific aspects of identity variation would be valuable.

4. Can you provide more details about the user study conducted to evaluate the perceptual quality and identity preservation of the generated images? How many participants were involved, and what were the specific instructions given to them?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4