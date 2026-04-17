# Review

## Summary
The paper presents EmbodiedMAE, a unified 3D multi-modal representation learning framework for robot manipulation that integrates RGB, depth, and point cloud modalities. The authors first enhance the DROID dataset by extracting high-quality depth maps and point clouds, resulting in DROID-3D, a large-scale dataset containing 76K trajectories (350 hours) of robot interaction data with synchronized 3D information. They then develop a multi-modal masked autoencoder (MAE) model that learns representations across these modalities through stochastic masking and cross-modal fusion. The model is pre-trained on DROID-3D and subsequently distilled into smaller variants for computational efficiency.

The authors evaluate EmbodiedMAE on diverse tasks across multiple benchmarks, including the LIBERO and MetaWorld simulation suites, as well as real-world tasks on low-cost (SO100) and high-performance (xArm) robot platforms. They compare their model against various state-of-the-art vision foundation models (VFMs), demonstrating superior performance in both training efficiency and final success rates. The model exhibits strong scaling behavior with increasing model size and effectively promotes policy learning from 3D inputs, particularly for tasks requiring spatial understanding in tabletop manipulation.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. **Comprehensive Evaluation**: The paper provides extensive experimental results across both simulation and real-world tasks, demonstrating the model's effectiveness and generalization capabilities. The evaluation covers a wide range of scenarios, including different difficulty levels and task types, which strengthens the validity of the findings.

2. **Strong Performance**: EmbodiedMAE consistently outperforms state-of-the-art VFMs in both training efficiency and final performance, showcasing its potential as a reliable foundation model for embodied AI applications. The model's ability to leverage 3D information effectively enhances its spatial understanding and manipulation capabilities.

3. **Scalability and Model Variants**: The framework demonstrates strong scaling behavior with increasing model size, and the availability of smaller distilled models makes it practical for deployment in resource-constrained robotics applications. This scalability is a key advantage for real-world adoption.

4. **DROID-3D Dataset Contribution**: The enhancement of the DROID dataset with high-quality 3D annotations is a significant contribution to the robot learning community. DROID-3D provides a valuable resource for pre-training 3D vision-language-action (VLA) models and promotes further research in embodied AI.

5. **Open-Source Code**: The authors commit to releasing the code upon publication, which facilitates reproducibility and enables other researchers to build upon their work. This transparency is highly appreciated and aligns with the open-source principles of the robotics community.

## Weaknesses
1. **Limited Discussion on Failure Cases**: The paper lacks a thorough analysis of failure cases or scenarios where EmbodiedMAE underperforms. Understanding the limitations and failure modes of the model would provide valuable insights for future improvements and practical deployments.

2. **Lack of Comparative Analysis with Other 3D VLA Models**: While the paper compares EmbodiedMAE with several VFMs, it does not include a comprehensive comparison with other 3D vision-language-action (VLA) models that also integrate 3D information. Including such comparisons would strengthen the argument for EmbodiedMAE's superiority.

3. **Insufficient Ablation Studies**: The ablation studies presented in the paper are limited in scope and do not thoroughly investigate the impact of different components or design choices in the model architecture. More extensive ablation studies would help validate the effectiveness of the proposed approach.

4. **Potential Overfitting to DROID-3D Dataset**: Although the model demonstrates good performance on the DROID-3D dataset, there is a concern about its generalization capabilities to other datasets or domains. The paper does not address potential overfitting issues or provide evidence of the model's robustness to data shifts.

5. **Absence of Long-Term Deployment Results**: The paper does not report on the long-term deployment of the model in real-world settings, which is crucial for assessing its practical viability and potential issues that may arise in extended use cases.

## Questions
1. **Failure Case Analysis**: Can the authors provide a detailed analysis of the failure cases observed during the evaluation of EmbodiedMAE? Understanding the scenarios where the model underperforms would be valuable for identifying potential areas for improvement.

2. **Comparative Analysis with Other 3D VLA Models**: How does EmbodiedMAE compare with other 3D VLA models that integrate RGB, depth, and point cloud modalities? Including such comparisons would help establish the model's superiority and address any potential gaps in performance.

3. **Ablation Studies on Different Components**: Can the authors provide more extensive ablation studies to investigate the impact of different components and design choices in the model architecture? This would help validate the effectiveness of the proposed approach and provide insights into the model's functionality.

4. **Generalization to Other Datasets**: How well does EmbodiedMAE generalize to other datasets or domains beyond DROID-3D? Have the authors tested the model's performance on different types of robot manipulation tasks or real-world scenarios to assess its robustness?

5. **Long-Term Deployment Results**: Can the authors provide results on the long-term deployment of EmbodiedMAE in real-world settings? This would be crucial for evaluating the model's practical viability and identifying any potential issues that may arise in extended use cases.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4