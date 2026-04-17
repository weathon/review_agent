# Review

## Summary
This paper revisits the prevailing paradigm of using multi-component LoRA architectures in multi-task learning, where diverse components are often seen as necessary for task-specific adaptation. The authors propose a simplified approach that challenges this assumption: a multi-head LoRA without a dynamic router (M-LoRA), which performs better than more complex, diversity-focused variants. They further demonstrate that increasing the rank of a standard LoRA can match the performance of these complex multi-component models. Based on these findings, the authors introduce Align-LoRA, which aligns task representations within a shared adapter space to enhance multi-task learning. Their results show that Align-LoRA achieves superior performance while maintaining efficiency and simplicity, suggesting a new direction for multi-task PEFT that prioritizes task-shared representations over architectural isolation.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The paper presents a compelling challenge to the prevailing multi-component paradigm in multi-task learning, offering a fresh perspective on how to achieve effective task adaptation with simpler architectures.

- The introduction of Align-LoRA, which aligns task representations within a shared adapter space, is a novel approach that demonstrates strong performance while maintaining efficiency and simplicity.

- The authors provide thorough empirical validation across multiple benchmarks, showing that their proposed methods consistently outperform existing complex architectures.

- The paper offers a theoretical analysis of Align-LoRA's generalization capabilities, providing a solid foundation for understanding the benefits of their approach.

## Weaknesses
- The experiments primarily focus on natural language tasks. It would be beneficial to see how the proposed methods perform in other domains, such as computer vision tasks, to assess their generalizability.

- The paper mainly compares against a few multi-component architectures. Including a wider range of multi-task learning approaches would provide a more comprehensive evaluation.

- While the authors mention that Align-LoRA introduces no additional modules increasing overhead, a more detailed analysis of the computational efficiency compared to other methods would be valuable.

## Questions
- How does the proposed method perform on tasks from different domains, such as computer vision tasks?

- How does the proposed method compare with other multi-task learning approaches beyond multi-component LoRA architectures?

- Can the authors provide more detailed analysis or benchmarks on the computational efficiency of Align-LoRA compared to other methods?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4