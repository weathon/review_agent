# Review

## Summary
The paper proposes a method to fine-tune a pretrained flow matching model such that the generated samples obey specific constraints, in particular physical constraints given by a partial differential equation with coefficients that are not necessarily known in advance. The method relies on the Adjoint Matching framework of Domingo-Enrich et al. and extends it to the setting with unknown coefficients. The authors demonstrate the effectiveness of their method on a range of physical systems.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The paper is well written and easy to follow. The authors do a good job of explaining the background material and the proposed method.
- The proposed method is novel and extends the Adjoint Matching framework to the setting with unknown PDE coefficients. This is a significant contribution as it allows for the joint generation of PDE solutions and parameters.
- The authors demonstrate the effectiveness of their method on a range of physical systems, including Darcy flow, linear elasticity, Helmholtz equation, and Stokes flow. They also show an application to natural images, which is a nice demonstration of the generality of the method.

## Weaknesses
- The method assumes that the weak form of the PDE residual is available, which may not always be the case. In some cases, the strong form may need to be used instead, which can lead to numerical issues when computing the residual. The authors should discuss this limitation and potential ways to address it.
- The method assumes that the PDE coefficients are single-valued and spatially independent. In many real-world applications, the coefficients can be complex-valued and spatially dependent. The authors should discuss this limitation and potential ways to address it.
- The method assumes that the boundary conditions are known. In many real-world applications, the boundary conditions may not be known exactly and may need to be inferred from noisy measurements. The authors should discuss this limitation and potential ways to address it.
- The method assumes that the initial condition is available. In many real-world applications, the initial condition may not be known exactly and may need to be inferred from noisy measurements. The authors should discuss this limitation and potential ways to address it.
- The authors do not provide a detailed comparison with other methods for enforcing physical constraints on generative models. For example, it would be useful to compare with the method of Bastek et al. (2024) mentioned in the related work section. Such a comparison would help to better understand the advantages and disadvantages of the proposed method relative to existing approaches.

## Questions
- How does the method perform when the weak form of the PDE residual is not available? Are there any ways to approximate the weak form from the strong form?
- How does the method perform when the PDE coefficients are complex-valued or spatially dependent? Are there any ways to extend the method to handle these cases?
- How does the method perform when the boundary conditions are not known exactly and need to be inferred from noisy measurements? Are there any ways to extend the method to handle this case?
- How does the method perform when the initial condition is not known exactly and needs to be inferred from noisy measurements? Are there any ways to extend the method to handle this case?
- How does the method compare with other methods for enforcing physical constraints on generative models, such as the method of Bastek et al. (2024)?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4