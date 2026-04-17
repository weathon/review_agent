# Review

## Summary
This paper introduces a reinforcement learning-based framework for scheduling directed acyclic graphs (DAGs) in heterogeneous environments. The proposed framework, WeCAN, utilizes a weighted cross-attention network to capture environment information and task dependencies, enabling rapid, single-pass network inference. The approach incorporates a skip action mechanism to address optimality gaps in list-scheduling methods, improving computational efficiency and adaptability. Experimental results demonstrate WeCAN’s superior performance compared to existing methods on both synthetic and real-world datasets.

## Soundness
3

## Presentation
2

## Contribution
2

## Strengths
1. The paper presents a novel approach by integrating a weighted cross-attention mechanism and a skip action mechanism within a reinforcement learning framework for heterogeneous DAG scheduling. This combination addresses the optimality gap in list-scheduling methods, contributing a fresh perspective to the field.

2. The authors provide a theoretical analysis of the optimality gap in list-scheduling methods, highlighting the limitations of current approaches and the importance of skip actions in achieving optimal solutions.

3. The paper demonstrates the effectiveness of WeCAN through extensive experiments on both synthetic and real-world datasets, showcasing its superior performance over state-of-the-art methods in terms of makespan and computational efficiency.

## Weaknesses
1. The introduction of weighted cross-attention and skip actions increases the model’s complexity, which may lead to higher computational costs compared to simpler heuristic methods.

2. The reliance on a reinforcement learning framework with complex mechanisms may make the model challenging to implement and tune, especially for practitioners who may prefer simpler, more intuitive scheduling approaches.

3. The experiments are conducted on specific datasets (TPC-H and Computation Graphs), which may not fully represent the diversity of real-world heterogeneous environments. Broader testing across more varied datasets could strengthen the claims of general applicability.

## Questions
1. How does the weighted cross-attention network handle environments with dynamically changing resource capacities and task types? Is there a mechanism to quickly adapt to these changes?

2. The paper introduces a skip action to address the optimality gap. How does this action impact the overall makespan and computational efficiency in practice, particularly in environments with varying task characteristics?

3. Can the proposed framework be extended to handle more complex dependencies or resource constraints, such as those found in dynamic or multi-resource environments?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4