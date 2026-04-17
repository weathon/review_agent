# Learning activation functions with PCA on a set of diverse piecewise-linear self-trained mappings

- Decision: Reject
- Scores: 2, 8, 2, 2

## Abstract
This work explores a novel approach to learning activation functions, moving beyond the current reliance on human-engineered designs like the ReLU. Activation functions are crucial for the performance of deep neural networks, yet selecting an optimal one remains challenging. While recent efforts have focused on automatically searching for these functions using a parametric approach, our research does not assume any predefined functional form and lets the activation function be approximated by a subnetwork within a larger network, following the Network in Network (NIN) paradigm. We propose to train several networks on a range of problems to generate a diverse set of effective activation functions, and subsequently apply Principal Component Analysis (PCA) to this collection of functions to uncover their underlying structure. Our experiments show that only a few principal components are enough to explain most of the variance in the learned functions, and that these components have in general a simple, identifiable analytical form. Experiments using the analytical function form achieve state of the art performance, highlighting the potential of this data-driven approach to activation function design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This submission proposes a method to find new formulations for activation functions. Motivated by previous work, that proposes parameterized activation functions, they propose a two step approach. In the first step, they express activation functions as MLPs - based on the universal function approximation theorem. In a second step, they identify the eigenfunction within the learned activation functions, and find a compressed symbolic form with few learned parameters. Empirically, they demonstrate that their activation function performs well on small computer vision datasets.

### Strengths
- The topic of the submission is very interesting: many aspects of deep learning architectures are iteratively designed to address specific shortcomings. The idea of learning activation functions and then condensing them into efficient and powerful versions is appealing and fits well into the general learning regime of DL.  
- The idea of representing a large class of activation functions with an MLP is neat. I’d be very curious to compare the performance and behavior of models with such learned activation functions to regular models and encourage the authors to add such an analysis.  
- The identification of eigenfunctions seems a powerful idea in this application. Given that the learned activations are piecewise-linear, the identification appears tractable. While this is a nice idea, the lack of details in this part makes it challenging to evaluate.

### Weaknesses
While I appreciate the general idea of the submission, there are several weaknesses.  
- **W1 - experimental evaluation:** The experimental identification of activation functions is based on small and simple image classification problems with 2-layer MLPs. While the evaluation adds a small CNN, both the basis to activation function search as well as the evaluation are very limited. I understand that the method requires repeated training of SLAF models to identify eigenfunction with a corresponding computational burden. However, if the authors are convinced of the merit of their method, they need to evaluate on larger and different domains, different architectures and different tasks in order to claim any general benefit.  
- **W2 - eigenfunction identification:** The method of eigenfunction identification remains unclear. Since that’s at the core of the proposed method and there is ample space within the page limit, the gap is problematic. I strongly encourage the authors to include further details on how they identify eigenfunctions and what function space they consider.  
- **W3 - inductive biases of SLAF:** The expression of the learned activation functions as MLPs is elegant and based in the universal function approximation theorem. That said, related work has shown that NNs generally and MLPs specifically have a bias towards specific parts of the signal, e.g., https://arxiv.org/abs/2403.02241. By induction, the MLP that the authors used for their search inherits that bias, and so it’s not entirely surprising that the activation function that is found is not dissimilar to existing functions. I encourage the authors to discuss these biases, whether or not they are desirable, and how that affects the overall search space.

### Questions
How are the eigenfunctions identified? Are you searching over a specific function space? If not, how did you to identify this particular formulation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel, data-driven methodology for discovering new activation functions. Instead of manual design or simple parametric search, the authors first define a "Self-Learning Activation Function" (SLAF) as a small, one-hidden-layer MLP, which is effectively a flexible piecewise-linear function. They then train thousands of networks (4608 in total) embedded with these SLAFs on a range of problems (MNIST, Fashion-MNIST, CIFAR-10) and SLAF network sizes to generate a diverse collection of effective activation functions.

The core contribution is the analysis of this collection. The authors apply Principal Component Analysis (PCA) and find that the first two principal components (PCs) explain over 99.5% of the variance in the learned functions. These two PCs are well-approximated by simple analytical functions. The first component is described as 'x * tanh(beta * x)' (a soft absolute value). The second component is a simple linear function, described as 'gamma * x'.

By combining these two components, the authors derive a new, two-parameter learnable activation function, which they term twish. It is defined by the expression 'x * tanh(beta * x) + gamma * x'. The authors note that twish is a generalization of the Swish activation function. In validation experiments on simple CNNs, twish is shown to consistently outperform ReLU, pReLU, and Swish in terms of both final test accuracy (particularly on the more complex CIFAR-10 dataset) and convergence speed. The work serves as a strong proof-of-concept for this new PCA-based discovery method.

### Strengths
Novelty of the Discovery Method: The core strength is the PCA-based methodology. Using PCA on a large ensemble of learned functions (the SLAFs) to distill their essential components is a highly original and intelligent approach to functional discovery.

Strong Empirical Finding: The discovery that just two principal components explain >99.5% of the variance is a powerful and elegant result, strongly suggesting a simple, low-dimensional underlying structure for effective activation functions.

### Weaknesses
Limited Scale of Validation: This is the primary weakness, which the authors rightly acknowledge. The validation experiments (Section 4) use small datasets (MNIST, CIFAR-10) and very simple CNNs. The true test of a new activation function is its performance and stability in deep, complex models (e.g., ResNets, ViTs) on large-scale tasks (e.g., ImageNet, large NLP corpora). Without this, it's hard to judge if "twish" will be broadly useful.

Limited Diversity of SLAF Training: The initial 4608 SLAFs were all trained on simple FFNs for small-scale image classification. It's an open question whether this set is "diverse" enough. The discovered PCs might be biased towards this specific task and architecture family.

### Questions
1. How were the beta and gamma parameters of the twish function initialized in the Section 4 experiments? Did you observe any sensitivity to this initialization during training?

2. Your analysis was based on SLAFs trained for classification tasks. Did you also conduct similar experiments for regression tasks?
If not, would you expect the PCA analysis on SLAFs trained for regression (e.g., optimizing MSE) to reveal the same principal components and, consequently, the same "twish" function, or do you think different functional forms might emerge?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a data-driven method to discover new neural network activation functions. It defines a small neural subnetwork (SLAF) that learns its own activation mapping during training, and then applies Principal Component Analysis (PCA) to thousands of these learned functions trained on small, classical datasets (MNIST, FashionMNIST, and CIFAR-10). The claim is that the first two principal components explain nearly all the variation and lead to a new function, they call twish, defined as $f(x; \beta, \gamma) = x \tanh(\beta x) + \gamma x$, which is a slight generalization of the Swish activation. Experiments are conducted on simple convolutional neural networks and results illustrating marginally better accuracy and faster convergence compared with ReLU, PReLU, and Swish are presented.

### Strengths
The paper is fairly clearly written.

### Weaknesses
This work presents a slight generalization of a popular activation function as the key contribution. Despite the work being experimental in nature, the conclusions as to the value of this new activation function are drawn based on toy datasets (MNIST, FashionMNIST and CIFAR10). In my view the experiments are not adequate to make any substantive claims and for this paper to be interesting I think you would need very strong evidence. I also don't see why the ideas behind the derivation of this activation function are particularly new, there are many techniques and approaches for deriving activation functions. In addition, despite the supposed benefit being that we derive activations from data, the key finding appears to be something close to what we already use.

### Questions
- In what sense is applying PCA to a collection of learned functions conceptually new, compared to previous meta-learning or search-based activation discovery frameworks, e.g., the one from Ramachandran et al?

- What theoretical advantage does twish offer over existing parameterized activations (e.g., Swish, Mish, GELU), for instance from an edge of chaos perspective / dynamical isometry / vanishing and exploding gradients perspective?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a data-driven method for discovering activation functions. 
A simple ReLU-based MLP is trained as a Self-Learned Activation Function (SLAF), and the shapes of the resulting functions — obtained from various datasets and network widths — are analyzed using PCA.
The top two principal components explain ~99.5% of the variance, corresponding respectively to $x\tanh(\beta x)$ and $\gamma x$. 
Building on this observation, the authors define a new activation function Twish, 
$f(x; \beta, \gamma) = x\tanh(\beta x) + \gamma x$.
Experimental results show that Twish achieves faster convergence and higher accuracy than ReLU, pReLU, and Swish across benchmark datasets such as MNIST, FashionMNIST, and CIFAR-10.

### Strengths
- Data-driven discovery: Uses PCA over a large pool of learned SLAFs to quantify shape diversity and reveal a compact, interpretable 2-D structure of activation functions. 
-Clear, reproducible pipeline: SLAF training → uniform sampling on a fixed grid → PCA → analytic fitting of PCs → definition of Twish. simple and easy to replicate.
- Strong quantitative support: The first two principal components explain ~99.5% of the variance, providing a solid justification for dimensionality reduction. 
- Empirical signal: On small CNNs (e.g., CIFAR-10), Twish shows faster convergence and consistently higher accuracy that ReLU, pReLU, and Swish.
- Practical value: Offers a lightweight alternative to large parameter searches for activation design, turning observed functional modes into a compact parametric family.

### Weaknesses
- Constrained search space: SLAFs are ReLU-based and thus piecewise-linear; the approach may bias discoveries toward piecewise linear shapes and under-explore smooth families.
- Preprocessing dependence: PCA is performed on BN-normalized pre-activations sampled only on [−4,4];  stability to the choice of range/resolution/normalization is not established.
- Limited scale: Experiments focus on small datasets and shallow models; generalization to ResNet/ViT/ImageNet or transformer LMs remains untested.
- Baseline coverage: Direct, controlled comparisons against modern smooth activations (e.g., GLEU, Mish, ELU, SELU) under the same setup are missing.
- Implementation specifics: The sharing/initialization/constraints of the learnable parameters $(\beta, \gamma)$ (layer, channel, or unit-level) are insufficiently detailed for full reproducibility.

### Questions
1. Where are $(\beta, \gamma)$ shared - at the layer, channel, or neuron level? Do you impose any constraints (e.g., $\beta \geq 0$)?
2. How do the PCA results and Twish’s performance change if BatchNorm is removed or if a different input range is used for sampling?
3. If the SLAF is trained as a non–piecewise-linear network (e.g., using tanh or RBF bases), does the same principal-component structure persist?
4. Do you have results applying Twish to larger models (e.g., ResNet, ViT, Transformers) or large-scale datasets (e.g., ImageNet)?
5. You exclude ELU/SELU/GELU citing Ramachandran et al. (2017) showing Swish's superiority. However, to rule out setting dependence, shouldn't you still report direct, controlled comparisons under your exact setup or at least provide their PCA coordinates relative to Twish?

### Soundness
3

### Presentation
3

### Contribution
2
