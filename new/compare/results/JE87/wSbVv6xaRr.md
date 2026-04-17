# Review

## Summary
This paper proposes FedMPDD, a novel federated learning algorithm that optimizes bandwidth utilization and enhances privacy by using multi-projected directional derivatives to encode gradients. By projecting gradients onto multiple random vectors and transmitting only the directional derivatives and random seeds, FedMPDD significantly reduces uplink communication costs. The server then decodes the aggregated information through multiple projections, overcoming the limitations of single-projection methods. The authors provide theoretical analysis demonstrating that FedMPDD converges at a rate of O(1/K), comparable to FedSGD, and offers inherent privacy against gradient inversion attacks due to the geometric properties of low-rank projections. Extensive experiments on benchmark datasets validate the effectiveness of FedMPDD in reducing network congestion and providing strong privacy protection while maintaining high model performance, outperforming existing methods in resource-constrained scenarios.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-written, with clear and concise explanations of the proposed method and its theoretical foundations. The authors effectively communicate complex concepts, making the paper accessible to a broad audience.

2. The authors provide a thorough theoretical analysis of the FedMPDD algorithm, establishing its convergence rate and privacy properties. The mathematical derivations are rigorous, and the use of multiple projections to overcome the limitations of single-projection methods is well-motivated and theoretically sound.

3. The paper includes extensive experiments on benchmark datasets, demonstrating the practical effectiveness of FedMPDD. The results show that the method reduces network congestion, provides strong privacy protection, and maintains high model performance, outperforming existing methods in resource-constrained scenarios.

## Weaknesses
1. The paper lacks a detailed analysis of the computational complexity of FedMPDD on client devices. While the authors mention that the computational cost is negligible for deep neural network models, a more thorough investigation is needed, especially for resource-constrained devices. Providing a detailed analysis or empirical evaluation of the computational overhead on various device types would strengthen the paper.

2. The paper does not explore the potential for adaptive attacks that could potentially circumvent the privacy protections of FedMPDD. Gradient inversion attacks are a specific type of adversary, and evaluating the robustness of FedMPDD against more advanced or adaptive attacks would provide a more comprehensive assessment of its privacy guarantees. Conducting experiments or providing a theoretical analysis of FedMPDD's resilience to adaptive attacks would enhance the paper's contribution.

## Questions
1. The paper mentions that the computational cost of FedMPDD is negligible for deep neural network models. Could you provide more details on the computational overhead of FedMPDD, especially for resource-constrained devices or smaller models? A more thorough analysis or empirical evaluation of the computational cost on various device types would be helpful.

2. The paper evaluates the privacy protection of FedMPDD against gradient inversion attacks. Have you considered the robustness of FedMPDD against more advanced or adaptive attacks? Conducting experiments or providing a theoretical analysis of FedMPDD's resilience to adaptive attacks would provide a more comprehensive assessment of its privacy guarantees.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4