# Review

## Summary
The paper proposes a new framework for learning Boltzmann generators, which are variational inference models for sampling from unnormalized probability distributions. The authors introduce a constrained optimization approach that iteratively transports the variational distribution towards the target distribution, enforcing constraints on both the KL divergence and entropy change between successive steps. This method aims to address issues like mode collapse and mass teleportation that affect existing approaches. The framework is implemented using normalizing flows and tested on standard BG benchmarks, as well as a new system, ELIL tetrapeptide, demonstrating superior performance in terms of effective sample size and mode coverage compared to state-of-the-art methods.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The paper proposes a novel framework for learning Boltzmann generators through constrained mass transport. The combination of trust-region and entropy constraints in an annealing path is innovative and well-motivated.
- The empirical results are strong, with significant improvements in effective sample size and mode coverage compared to existing methods. The method is tested on multiple benchmarks, including a new large system (ELIL tetrapeptide), demonstrating its effectiveness.
- The paper provides a rigorous theoretical foundation for the proposed method, including proofs for the properties of the annealing paths. The connection to existing geometric annealing methods is well-explained.

## Weaknesses
- The paper could provide more discussion on the sensitivity of the method to the choice of hyperparameters, particularly the trust-region bound and entropy constraint.
- While the paper compares to several state-of-the-art methods, it would benefit from a more detailed comparison with other recent approaches, especially those using diffusion models for Boltzmann generator tasks.
- The practical implementation details could be expanded, particularly regarding the computational cost of the additional optimization steps and the practical guidelines for setting the constraints.

## Questions
- Can the authors provide more insight into the sensitivity of the method to the choice of trust-region bound and entropy constraint? Are there practical guidelines for setting these hyperparameters?
- How does the method perform compared to recent diffusion-based approaches for Boltzmann generators?
- Could the authors elaborate on the computational cost of the additional optimization steps required for CMT? How does this affect the scalability of the method?
- The paper mentions the potential for transferability across different molecular systems. Could the authors provide more details on this aspect and its practical implications?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4