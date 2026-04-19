# A Framework of SO(3)-equivariant Non-linear Representation Learning and its Application to Electronic-Structure Hamiltonian Prediction

- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
We propose both a theoretical and a methodological framework to address a critical challenge in applying deep learning to physical systems: the reconciliation of non-linear expressiveness with SO(3)-equivariance in predictions of SO(3)-equivariant quantities, such as the electronic-structure Hamiltonians. Inspired by covariant theory in physics, we present a solution by  exploring the mathematical relationships between SO(3)-invariant and SO(3)-equivariant quantities and their representations. We first construct theoretical SO(3)-invariant quantities derived from the SO(3)-equivariant regression targets, and use these invariant quantities as supervisory labels to guide the learning of high-quality SO(3)-invariant features. Given that SO(3)-invariance is preserved under non-linear operations, the encoding process for invariant features can extensively utilize non-linear mappings, thereby fully capturing the non-linear patterns inherent in physical systems. Building on this, we propose a gradient-based mechanism to induce SO(3)-equivariant encodings of various degrees from the learned SO(3)-invariant features. This mechanism can incorporate non-linear expressive capabilities into SO(3)-equivariant representations, while theoretically preserving their equivariant properties as we prove, establishing a strong foundation for regressing complex SO(3)-equivariant targets. We apply our theory and method to the electronic-structure Hamiltonian prediction tasks, experimental results on eight benchmark databases covering multiple types of systems and challenging scenarios show substantial improvements on the state-of-the-art prediction accuracy of deep learning paradigm. Our method boosts Hamiltonian prediction accuracy by up to 40\% and enhances downstream physical quantities, such as occupied orbital energy, by a maximum of 76\%. Our method also significantly promotes the acceleration performance for the convergence of traditional Density Functional Theory methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a method combining SO(3)-equivariance with non-linear expressiveness, and applied it for Hamiltonian prediction. This method has a theoretical foundation, and experiments demonstrate the effectiveness of the approach.

### Strengths
- Theoretical analysis is provided to ensure equivariance is maintained alongside non-linear expressiveness.
- Experimental results indicate significant improvement in Hamiltonian prediction accuracy, outperforming QHNet and DeepH-E3.
- Detailed explanations are provided for the experimental setup, showcasing a comprehensive evaluation.
- The proposed method is simple and can be easily combined with existing approaches.

### Weaknesses
- To show whether the acceleration in DFT outweighs the additional computation, it would be helpful if the paper provided a comparison of the wall time cost for accelerating traditional DFT algorithms. 
- Since SO(3)-equivariance is important in many contexts, could the authors provide an example of how this model might be applied to another prediction task?

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present a novel architecture and loss function designed for learning targets in SO(3) symmetry. Their focus is on the challenge of predicting the Hamiltonian within Kohn-Sham Density Functional Theory (DFT). Their contribution is twofold: First, they suggest fitting the Forbenius norm of the target alongside the direct fitting of the equivariant targets. Second, they propose a method for differentiating the predicted Forbenius norm of the Hamiltonian, utilizing invariant representations to produce equivariant embeddings that capture non-linear dependencies within SO(3)-equivariant embeddings.

### Strengths
* The proposed TraceGrad method shows significant improvements over previous approaches.
* The use of differentiation to achieve higher-order SO(3)-equivariant embeddings is a novel concept, as it has not been extensively explored before, apart from its application to atomic forces.

### Weaknesses
* The paper suffers from unclear writing, particularly in the abstract and introduction, which fail to clearly convey the contributions and include overly verbose sections (e.g., lines 66-72).
* Although the paper claims to present a general framework for SO(3) equivariance, it is heavily focused on Hamiltonian prediction, which limits its broader applicability.
* The method is inherently designed to learn conservative vector fields on the embeddings, a point that is not addressed but could significantly influence its empirical performance.
* l.57-61, SO(3)-equivariant models include non-linearities through gating or activating the norm of a vector.
* l.129,130, "Nevertheless, multiplying SO(3)-equivariant features with linear coefficients may not fundamentally improve their non-linear expressiveness." is a claim without evidence.
* l.43, In DFT, the Hamiltonian is not predicted but computed or solved for.
* l.46, Hamiltonian prediction cannot work in O(N) given that the matrix itself is O(N²).

### Questions
* Are the models in the experiments adjusted for size and compute time? The extra computations through TraceGrad could equally be spent on larger networks, how does this compare?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the challenge of incorporating non-linear expressiveness into SO(3)-equivariant deep learning frameworks for physical systems. The authors propose a theoretical framework that constructs SO(3)-invariant quantities tr(Q·Q†) from SO(3)-equivariant targets and utilizes gradient operations to induce SO(3)-equivariant features while maintaining non-linear expressiveness. The method is applied to electronic-structure Hamiltonian prediction tasks and evaluated on eight benchmark databases, showing improvements in prediction accuracy and downstream physical quantities. The method demonstrates generalization ability across various scenarios including thermal motions and bilayer twists, and aids in accelerating traditional DFT calculations through better initialization. The main contribution is providing a framework that attempts to combine strict SO(3)-equivariance with non-linear expressiveness in deep learning models for physical systems.

### Strengths
Originality:
- Proposes a novel perspective on bridging SO(3)-equivariance and non-linear expressiveness through invariant quantities and gradient operations
- Creates a theoretical connection between invariant and equivariant representations for l>1.
- Introduces a gradient-based mechanism for inducing equivariant features while preserving non-linear information

Quality:
- Provides rigorous mathematical proofs for the theoretical framework
- Conducts comprehensive experiments across several different benchmark databases

Clarity:
- Well-structured presentation of the theoretical framework
- Clear mathematical derivations with detailed proofs in appendices
- Thorough experimental analysis with multiple evaluation metrics

### Weaknesses
While proposing a new method, it fails to provide detailed comparisons of parameter counts, memory consumption, training time, and convergence behavior between the baseline and proposed models. The claimed superiority of gradient-based transfer over traditional gating activation mechanisms is not convincingly demonstrated through empirical evidence. Moreover, the practical significance of the accuracy improvements is not well justified - the paper does not discuss what constitutes a meaningful accuracy threshold for Hamiltonian prediction tasks, nor does it adequately analyze the trade-offs between these moderate accuracy improvements and their associated computational costs. These omissions make it difficult to fully evaluate the practical value and efficiency of the proposed method relative to existing approaches. A comprehensive analysis of these aspects would significantly strengthen the paper's contributions and help readers better understand the method's practical advantages and limitations.

### Questions
1. The paper proposes using gradient operation (∂z/∂f) to transfer nonlinearity from l=0 features to equivariant features, where z = S_nonlin(u) and u = CGDecomp(f⊗f, 0). However, similar to the gated activation mechanisms, this method still essentially applies nonlinearity only to l=0 components. The fundamental advantage of gradient-based transfer over the gated activation mechanism remains unclear - could the authors provide theoretical insights on why gradient operations might be more effective than gated activation for transferring nonlinear information from l=0 to higher-degree features?

2. The paper lacks crucial implementation details comparing the baseline and proposed method, such as parameter counts, training epochs, convergence curves, and computational overhead. More importantly, the reported improvements are moderate (e.g., MAE reduction from 0.274 to 0.226 on BB dataset) - such accuracy gains could potentially be achieved through increasing training epoch, learning rate adjustment, or other hyperparameter tuning of the baseline models. To demonstrate the superiority of the proposed method, one needs to compare the upper limits of model expressiveness after controlling for various training factors (number of epochs, hyperparameters, optimization strategies, random seeds, etc.). The current empirical evidence does not adequately address this comparison.

3. For electronic-structure Hamiltonian prediction tasks, what is the practically meaningful accuracy threshold? If, for instance, 1 meV accuracy is sufficient for most practical applications, how do we justify the value of further accuracy improvements (e.g., from 0.274 to 0.226 meV on BB dataset)? Could the authors discuss the practical significance of such fine-grained improvements in the context of real-world applications, such as materials design and DFT calculations?

4. In line 1116, The paper compares the O(N) complexity of Hamiltonian prediction with the O(N³) complexity of traditional DFT methods. However, this comparison seems unfair: the O(N³) complexity in traditional DFT includes solving the Kohn-Sham equations, while the O(N) complexity here only covers Hamiltonian prediction. To obtain physical properties from the predicted Hamiltonian, one still needs O(N³) operations for diagonalization. Could the authors clarify this comparison and provide a more comprehensive complexity analysis that includes all steps necessary for obtaining the same physical properties in both approaches?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a new framework that strictly satisfies the SO(3) equivalence and has the potential to enhance expressive power with nonlinearity. The effectiveness of the proposed ansatz is demonstrated with experiments on Hamiltonian prediction tasks and DFT.

### Strengths
-	Balancing between physics constraints (i.e. SO3-equivalence in this task) and the expressive power of the network is highly non-trivial. Therefore, an ansatz that both strictly satisfies the equivariance constraint and has the potential to leverage nonlinearity for expressiveness improvement is a noticeable contribution. 
-	The authors test their new framework on various tasks

### Weaknesses
The main weakness of this paper is lack of readability. Many important background and detailed introduction to the problem, the task, and technical details are missing, making it hard to follow.

-	L156-160, it is very abrupt when concepts like Q and direct-product state are mentioned without any context. The *problem formulation* section should provide details (with clearly explained equations) on how the related concepts are defined, along with their physical meaning related to the task, which is also not clearly delivered.

-	In appendix A, it is hard to understand why the authors only include well-known concepts like definitions of group and equivariance but omit those more problem-specific and highly relevant ones, e.g. irreducible representation, properties of Wigner-D-matrix, Clebsch-Gordan decomposition, etc. These are crucial for readers to understand their methods. Hiding all these related contexts is not appreciated.

-	The learning task considered in the experiment section is not explicitly described. If the authors find themselves running out of pages, a detailed description on electronic-structure Hamiltonian prediction task should then be added in the appendix, including the physics background, the mapping to be learned, and methods adopted as well as model architecture. Same comments on the DFT experiment.

Also, I didn’t find a comparison on computation cost, e.g. GPU hours / FLOPS for the Hamiltonian prediction task.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
3
