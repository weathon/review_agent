# Review

## Summary
This paper introduces On-Demand Communication (ODC), a novel approach that replaces collective communication with direct point-to-point communication in Fully Sharded Data Parallel (FSDP) training for LLMs. ODC addresses the inefficiencies caused by synchronization barriers in FSDP, particularly under imbalanced workloads. By relaxing synchronization from a layer-level to a minibatch-level, ODC improves device utilization and training throughput, achieving up to 36% speedup over standard FSDP. The paper demonstrates ODC's effectiveness across various LLM post-training tasks, including supervised fine-tuning and reinforcement learning.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. ODC's approach to decoupling device workloads and enabling finer-grained load balancing addresses a significant limitation of traditional FSDP, particularly under imbalanced workloads.
2. The empirical results are compelling, showing substantial speedups over standard FSDP across different tasks and settings.

## Weaknesses
1. The shift to a decentralized parameter server model may introduce complexity in implementation and management, particularly for large-scale systems.
2. ODC's performance may be sensitive to network conditions, potentially affecting its effectiveness in environments with varying or poor network quality.

## Questions
1. How does ODC compare to other emerging techniques for improving communication efficiency in distributed training, such as asynchronous communication or more advanced collective communication strategies?
2. What are the potential challenges or limitations of ODC's decentralized approach, particularly regarding fault tolerance, scalability, and security in large-scale systems?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4