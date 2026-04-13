## Human Reviewer 1

### Summary
This paper shows that it is possible to "learn" the solution to a particular PDE from the boundary/initial conditions as well as labeled data for the partial derivatives at collocations points in the space-time domain. In essence, this is an alternative approach to supervised learning of the solution, but using the gradients of the solution field rather than the solution itself.

### Strengths
The paper is well-presented and structured.

### Weaknesses
1. Although this paper is technically sound, my first major concern is that it is misleading in terms of what the proposed method achieves. For example, sentences such as "We claim that it is possible to learn physically consistent models without explicit knowledge about the underlying equations" or "DERL outperforms PINNs and other state-of-the-art approaches" make it sound like the paper proposes a novel method to solve PDEs, while that is NOT the case at all. What this paper proposes is a supervised learning approach for a particular PDE, where instead of training the network on the values of the solution, the network is instead trained on the gradients of the solution.
2. I fail to see the utility of this method in practice: it can't be used to calculate a PDE solution without a priori knowledge on the gradients of the solution. Thus, this paper's real contribution seems rather trivial to me, and it would be more suitable for a workshop rather than ICLR.

### Questions
It is surprising that the proposed method outperforms Sobolev learning, which trains the network using both the solution itself as well as its gradients. It seems that more data should make learning easier, especially when we have labeled data for the direct outputs of the neural network. Can you explain why you are getting such results?

### Soundness
3

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a method for learning solutions to ODEs/PDEs by training with information of the derivatives of the solution, called DERL. The method using a loss term consisting of the error in the derivatives and the error in the initial/boundary conditions. DERL is compared to PINN, OUTL, and SOB on several differential equation problems. Additional, the method is applied to learning the solution from a "teacher" model.

### Strengths
The theorems are sound, other than some notational issues. The experiments are extensive and emperically, the proposed model performs best in most cases.

### Weaknesses
The biggest concern is that the comparison between DERL and PINN is unfair because PINN does not have access to any information about the true solution within the domain. It only has access to the initial/boundary data and the PDE. DERL on the other hand has access to information about the true solution. It is access the derivative of the solution rather than the solution itself, but the PINN does not have any information about the true solution within the domain. So the results of DERL learning better than PINN are predictable. 
Another weakness is that all of the comparisons are against other methods that are not the most recent. Most of the other methods are from 2019 or later. It would be good to see comparisons to more recent methods.

### Questions
* In Table 3, the PINN loss is listed as 030950. Is this supposed to be 0.030950?
* For the PINN distillation, you list the model architecture for the teacher model. Is it the same architecture for the student model? Or is the student model a smaller model?
* You have figures comparing DERL to HNN and LNN. Do you have loss infomation for HNN and LNN similar to Table 2 to compare?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper primarily addresses a critical question in the training of neural PDE solvers: what should the learning objective be, or, in other words, how should the loss function be defined. Previous work has commonly utilized either the discrepancy between predicted and true state values or the residual form of the PDE itself as the loss. In contrast, this paper proposes a new loss function based on the derivatives of the state with respect to the time variable \(t\) and spatial variable \(x\) as supervisory signals. The authors also theoretically demonstrate that gradient-based supervision alone is sufficient and necessary for convergence toward the target solution. Extensive numerical experiments are conducted to validate the proposed method’s effectiveness.

**[-]** The main content of the paper centers around introducing the authors' perspective and framework—specifically, the IMPORTANCE OF DERIVATIVE LEARNING. However, it lacks a detailed discussion of challenges encountered during this process and the proposed solutions. Thus, the primary contribution lies in the perspective itself, asserting that derivative-based supervisory signals are both important and novel, which is the key selling point of the paper. However, from my viewpoint, this perspective is not entirely novel, as related approaches have been explored in previous work. Here are a few examples:

- [1] Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in neural information processing systems 33 (2020): 7462-7473.
    - Research has shown that first- or second-order gradients alone can be effective for image supervision and has examined the associated challenges, such as activation function selection.
- [2] Li, Chongchong, et al. "Gradient information matters in policy optimization by back-propagating through model." International Conference on Learning Representations. 2022.
  -  In reinforcement learning model training (closely related to the ODE scenarios in this paper), gradient information has been shown to be crucial for downstream control tasks, with solutions also proposed to address this.
- [3] D'Oro, Pierluca, and Wojciech Jaśkowski. "How to learn a useful critic? model-based action-gradient-estimator policy optimization." Advances in Neural Information Processing Systems 33 (2020): 313-324.
  -  In the learning of value functions in reinforcement learning, since downstream policy learning relies on the gradient of the value function (similar to how PDE or ODE gradients are essential for model evolution in this paper), gradient-based supervision is incorporated into value function learning.

These are merely indicative examples, suggesting that this topic has been explored in deep learning literature. The authors should demonstrate awareness of such research to avoid reinventing the wheel.

**[-]** Moving to the specific research content, Definition 2.1 attempts to prove the sufficiency of the proposed loss, but it does not clarify whether methods like PINN, OTL, or SOB are also sufficient, nor does it establish any clear superiority of the proposed loss over these methods.

**[-]** If we are given a PDE with an analytical form along with initial and boundary conditions, how would the proposed method obtain supervisory signals? Would it still require numerical methods to generate data \(u(x, y, t)\), followed by gradient estimation via interpolation?

**[-]** For a system with a 3D input (x, y, t) and a 2D output (u, v), first-order gradient information actually forms a Jacobian matrix (2x3). Would the second-order Hessian matrix then be a higher-dimensional matrix? What computational resources would be required to estimate such derivative information?

**[-]** Regarding the experiments, could the authors clarify why the metrics in Tables 2, 3, 4, 5, and 6 differ so greatly ( Perhaps  Table 3 includes the three more intuitively understandable metrics)?

**[-]** The proposed method does not perform well in Tables 4 and 5, and its performance in Table 7 is comparable to that of SOB (which aligns with my earlier concerns about the theoretical part). I reasonably suspect that, if given appropriate weight adjustments, the SOB method would be at least as effective as the proposed loss, given that the proposed loss is a subset of the SOB method’s loss function. The paper provides no evidence (or intuitive argument) to suggest that adding a reasonable constraint to a loss would have adverse effects.

**[-]** The paper's presentation could be improved. Too many important details are placed in the appendix, leaving the main text with almost no information about the algorithm and experimental specifics—such as how the loss function is implemented. There is nearly one page of unused space in the main text, which could be used to provide these crucial details.

### Strengths
see above

### Weaknesses
see above

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper proposes training neural networks by fitting its derivatives to the derivatives of a data set. The authors evaluate this method on physical systems, where the derivatives are obtained from the solution trajectories of differential equations. 

I see several fundamental issues with this work:
1) The method is presented as ‘physically consistent’, like physics informed neural networks (PINNs), because it uses partial derivatives. Later, it is argued that the authors’ method is easier to train because the network’s partial derivatives are compared to ‘individual targets instead of being entangled together’. The loss of PINNs incorporates the relationship between partial derivatives; this is what makes them physically consistent. To me, the proposed method appears to be a supervised method through and through.
2) The method is presented as a solution method to physical systems (see for instance in the conclusion: ‘We showed theoretically and experimentally that our method successfully learns the solution to a problem and remains consistent with its physical constraints, …’). What was actually done in the experiments is that the solution was known beforehand, either determined by classical numerical methods or it was a system with a known analytical solution. Then a neural network was fitted to that solution. While it is true that a solution was learned in the literal sense, the statement is misleading, as it indicates their method is a solution method for differential equations, just like PINNs. But more importantly, it raises the question what the method’s intended purpose then is.
3) The role of physics in this method is not really clear. As explained, the proposed technique is a method to fit a neural network to a given curve, not a solution technique. The authors argue that this is a consequence of the uniqueness theorem from the theory of ordinary differential equations, where dx/dt = f(x,t). What the method is actually based on is a simplified case, where the derivatives are known beforehand, corresponding to dx/dt = f(t) and within the scope of the well-known fundamental theorem of calculus. That a curve can be reconstructed from its derivatives is not a specific property of dynamical systems but only a property of a differentiable curve. For this reason, I would also recommend to present this work more related to Sobolev training than to physical systems and PINNs. The connection to Sobolev training should be clear from the beginning due to their strong similarities.
4) In section 2.1, theoretical analysis, the statements of the mathematical theorems are hard to follow as some of the quantities are not declared, or used in a way that does not make sense. For instance, it is not clear what the arrow in theorem 2.1 precisely means as there are no sequences that could somehow be connected to convergence of some sort. Consequentially, the proof also makes no sense to me.

### Strengths
See above.

### Weaknesses
See above

### Questions
See above.

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
1

### Confidence
3