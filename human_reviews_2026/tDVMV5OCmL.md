# IPPRO: Importance-based Pruning with PRojective Offset for Magnitude-indifferent Structural Pruning

- Decision: Reject
- Scores: 2, 6, 4, 0

## Abstract
Not only the classical methods of neural network pruning but also most importance-based pruning methods rely too much on parameter magnitudes to prune effectively. We propose a novel pruning strategy, named IPPRO, using projective space to alleviate the unfair advantage given to parameter magnitudes. We use gradient of loss in the projective space to construct PROscore, which is a magnitude-indifferent score that is in turn used by IPPRO, our novel importance-based structured pruning algorithm. Extensive experiments on Convolutional Neural Networks (CNNs), Vision Transformers (ViT), and Large Language Models (LLMs) demonstrate that IPPRO consistently outperforms, especially in high compression scenarios. Our results establish IPPRO as a task-agnostic and architecture-agnostic pruning paradigm, offering both a new theoretical foundation and a practical tool for magnitude-indifferent structured pruning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes IPPRO (Importance-based Pruning with PRojective Offset), a magnitude-indifferent structured pruning method for neural networks. The key innovation lies in using projective geometry to evaluate filter importance through angular displacement under gradient descent, rather than relying on filter magnitudes. 

The authors introduce PROscore, computed as the tangent of angular distance in projective space after one gradient descent step, as the importance criterion. Extensive experiments demonstrate IPPRO's effectiveness across CNNs, Vision Transformers, and LLMs, showing consistent performance improvements especially in high compression scenarios and without fine-tuning.

### Strengths
1. The projective geometry framework provides a principled alternative to magnitude-based heuristics. 

2. Testing on 7+ architectures (ResNet, MobileNet, DeiT, EfficientFormer, DeepLabV3, LLaMA).

3. Robust performance without fine-tuning.

### Weaknesses
1. Why should angular distance in projective space correspond to filter importance? Is there a theoretical justification that angular distance is better than geometric distance or norm? A formal theorem would strengthen the contribution. It would also be important to theoretically justify why angular distance is superior, rather than merely demonstrating that it performs better.

2. No wall-clock time comparisons with baselines. This is important for pruning papers.

3. The performance improvement is marginal. For example, in Table 2 (Cityscapes – DeepLabV3-ResNet50), when comparing IPPRO (ours) with SIRFP (Wu et al., 2025), the improvement in mIoU is very small. Under the base pruning setting, SIRFP achieves 81.3 mIoU, while IPPRO reaches 81.5 mIoU, giving only a +0.2 gain. In terms of FLOPs reduction, IPPRO obtains 61.8 %, just 0.5 % higher than SIRFP’s 61.3 %. This slight edge shows that the performance improvement is marginal, i.e., the proposed method performs almost on par with the previous state-of-the-art under the same compression ratio.

4. No analysis on modern architectures (e.g., Swin Transformer). 

5. Experiments on Imagenet should be put in the main text, not supplementary.

### Questions
see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes IPPRO (Importance-based Pruning with PRojective Offset), a structured pruning framework that overcomes the limitations of magnitude-based pruning by focusing on directional importance. The method embeds each filter into a projective space, where importance depends on direction rather than scale, ensuring scale-invariant comparison across layers. A new PROscore measures how much each filter’s direction changes under a gradient step, and filters that move closer to the origin are pruned. A simple parameter injection trick allows this computation without affecting model outputs. Experiments on CNNs, Vision Transformers, and LLMs show that IPPRO achieves slightly better accuracy and greater stability than previous direction-based pruning methods even without fine-tuning.

### Strengths
**1. Clear and principled formulation of scale invariance through projective geometry**

The paper formalizes magnitude-independent pruning not as heuristic normalization but as a property of the underlying space. By embedding filters into a projective space, the method guarantees scale invariance at the definition level, providing a clean and mathematically grounded justification for direction-based importance.

**2. Unified pruning framework applicable across architectures**

The same projective formulation applies consistently to CNNs, Vision Transformers, and large language models. This unification makes the approach architecture-agnostic and highlights that the proposed importance measure is not tied to a specific model design or normalization scheme.

**3. Comprehensive empirical validation across diverse tasks**

Experiments cover image classification, semantic segmentation, and language modeling, using models such as ResNet-50, DeepLabV3, DeiT, and LLaMA-2-7B. The breadth of evaluation supports the claim that IPPRO is a general framework rather than a task-specific trick.

**4. Robustness and fine-tuning-free performance**

IPPRO maintains accuracy even when computed with limited data sampling and performs competitively without any fine-tuning after pruning. This property makes the method practical for large-scale models where retraining is expensive or infeasible.

**5. High clarity and reproducibility**

The paper is well-organized, with precise notation, clear pseudo-code, and detailed ablations that enhance transparency. The paper’s presentation quality and completeness make reproduction straightforward and strengthen the empirical credibility of the results.

### Weaknesses
**1. Direction-based pruning has been extensively explored in prior work**

Several recent methods such as Torque, Catalyst, and geometric pruning already focus on gradient direction rather than magnitude. IPPRO provides a cleaner mathematical reformulation but does not introduce a fundamentally new optimization insight.

**2. Limited theoretical gain from adopting projective geometry**

Although the paper frames pruning in the language of projective geometry, the practical effect largely reduces to normalizing vectors and measuring angular displacement. The framework adds elegant terminology but yields little new theoretical understanding beyond existing direction-based normalization schemes.

**3. Increased computational overhead with limited performance improvement**

Computing PROscores requires additional gradient accumulation and parameter injection, which significantly increases pruning cost. However, the resulting accuracy gain over previous direction-based methods is minimal, raising concerns about the efficiency–benefit trade-off.

**4. Unclear behavior in hybrid architectures without manual layer-wise control**

While the method is claimed to be unified across CNNs and Transformers, it is unclear how well the approach generalizes to mixed architectures such as ConViT or hybrid CNN–ViT models when global pruning is applied without manually setting layer-wise ratios. This raises questions about the true level of architectural unification achieved by the framework.

### Questions
See the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors develop an architecture-agnostic pruning algorithm (although it is not entirely clear how it would work on a simple MLP) that is based on ideas from projective geometry. Concretely, the method uses a score called PROscore (defined in Eq (4)) involving a modified version of the model under consideration that includes some fresh parameters (parameter injection, the meaning of which isn’t entirely clear) and the gradient of the loss function of this modified model.

The authors then explain how to use this score function for pruning in the case of CNNs, vision transformers and LLMs and proceed to show that the method works well in a very wide range of experiments.

### Strengths
- Based on the empirical section of the paper, the method clearly works very well.
- The experiments and results are convincing and well put together.
- The scoring function (4) is interesting.

### Weaknesses
- The paper starts from the following axiom: when it comes to pruning a NN, the idea that the magnitude of the parameters of the NN is important is a myth (143-145), and “magnitude-invariant” methods must be developed. There is a mathematical and a motivational problem with this perspective:
	- Mathematical: What operation are involved in a forward pass through a NN? Lots of additions and multiplications, ReLU and softmax to name the most important. Addition, multiplication by positive numbers, ReLU and softmax are isotone in their arguments, multiplication by negative numbers is antitone; in particular they are all monotone. Therefore, inputs with bigger magnitudes (i.e. absolute values) produce outputs with bigger magnitudes. Thus magnitude matters at a very fundamental level in any NN. The onus is therefore on the authors to substantiate their claim that the importance of magnitude is a myth on the face of this very basic mathematical fact. Which brings me to the second point,
	- Motivational: one cannot do science by simply stating that a certain way of doing things is a myth and taking another approach. If magnitude is not, or less, important than one might think, this must be documented and explained. This paper doesn't do either.
- Despite the claim that the proposed method is “magnitude-invariant” (a term that is never defined precisely), the proposed solution is very much magnitude dependent. The expression (4) -- which has a typo, it should $\lVert F_i\rVert$ in the denominator, not $D_i$ -- is large when the numerator $\lVert F_i-\lambda\nabla_{F_i}\mathcal{L} \rVert$ is large, i.e. the updated $F_i$ with learning rate $\lambda$ has a large magnitude, and the denominator $\lvert \lVert F_i\rVert-\lambda\frac{\partial\mathcal{L}}{\partial D_i} \rvert$ is small (the interpretation of this term is more complicated, see below). In which sense is this magnitude-invariant?
- The expression (4) could have been written without any reference to projective space and projective geometry. It really plays no role in this story. Eq (4) provides a gradient-based method with an unusual denominator, and this denominator is the heart of the story. 
- I wish more time and care had been spent on this denominator $\lvert \lVert F_i\rVert-\lambda\frac{\partial\mathcal{L}}{\partial D_i} \rvert$. To start to understand it one must jump to the next section to find the definition of $\frac{\partial\mathcal{L}}{\partial D_i}$ (this is not a great way to present things…). But this is not enough to understand the meaning of this term. What exactly is $\psi$ in (5)? What does “modifying element-wise computation layer $\sigma$” mean? If $\sigma(x)$ returns a tuple of dimension N and $x$ is a tuple of dimension $M\neq N$ how are we supposed to understand (5)? Since $\frac{\partial\psi}{\partial D_i}=x$ (and the second derivative will therefore vanish), what can we say about the general shape of $\frac{\partial\mathcal{L}}{\partial D_i}$ from the chain rule? And what does this mean for the proposed method and the meaning of the denominator of (4)?
- The paper is littered with grammatical errors, making it quite hard to read in some places.

### Questions
If the proposed method does work better than other pruning strategies, it looks like it is due to the denominator in (4) which penalises certain behaviours. Which behaviours and how? What is the role of $\lambda$ (and how is it chosen)? What is the intuition behind $\frac{\partial\mathcal{L}}{\partial D_i}$? What rate of change does it measure?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The work introduces a structured pruning approach for neural nets, that uses projective space. This attempts to address some of the shortcomings of magnitude based pruning. It projects the filters on latent space and then notices its movement during gradient descent. An importance score is then computed leveraging the movement information, which in turn forms the basis for pruning the filters.

### Strengths
The only strength of the article, in my opinion, is that it attempts to address a very timely problem is neural net.

### Weaknesses
The papers has several major weaknesses. 

1. First and foremost, the description of the proposed method is very poor. It is very difficult to understand what is going on. I do not think a reader would be able to implement the algorithm from the description given in the paper. Specifically, I do not find the following critical information. 
  A) When does it stop taking the filters out from the net? 
  B) What happens if all filters are removed from a layer and how does the approach handle layer collapse? 
  C) Does it need pre-trained model always? 

2. Projective geometry seems the key idea of the paper. Yet, there is hardly any clarity in the paper why and how does it help. The description seems too superficial and cursory. 

3. Algorithm 1 does not add any value, in my opinion. Rather the authors should use the space to better justify why and how of projective geometry. 

4. Results: Finally, the results are nowhere close to the state of the art. For example, on CIFAR 10, CURL (Neural network pruning with residual-connections and limited-data, 2020), SPvR (SPvR: Structured Pruning via Ranking , 2025), Hrank ( Hrank:Filter pruning using high-rank feature map, 2020), OTOv2 achieves the same performance as the proposed method with 40% (absolute) lesser parameters. Overall, I think the proposed approach uses much more parameters than the state of the art models to achieve the same performance.

### Questions
I have no additional question. In weakness section, I have detailed the issues.

### Soundness
1

### Presentation
1

### Contribution
1
