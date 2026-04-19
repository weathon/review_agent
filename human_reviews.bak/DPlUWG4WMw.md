# Operator Deep Smoothing for Implied Volatility

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
We devise a novel method for nowcasting implied volatility based on neural operators.
Better known as implied volatility smoothing in the financial industry, nowcasting of implied volatility means constructing a smooth surface that is consistent with the prices presently observed on a given option market.
Option price data arises highly dynamically in ever-changing spatial configurations, which poses a major limitation to foundational machine learning approaches using classical neural networks.
While large models in language and image processing deliver breakthrough results on vast corpora of raw data, in financial engineering the generalization from big historical datasets has been hindered by the need for considerable data pre-processing.
In particular, implied volatility smoothing has remained an instance-by-instance, hands-on process both for neural network-based and traditional parametric strategies.
Our general *operator deep smoothing* approach, instead, directly maps observed data to smoothed surfaces.
We adapt the graph neural operator architecture to do so with high accuracy on ten years of raw intraday S&P 500 options data, using a single model instance.
The trained operator adheres to critical no-arbitrage constraints and is robust with respect to subsampling of inputs (occurring in practice in the context of outlier removal).
We provide extensive historical benchmarks and showcase the generalization capability of our approach in a comparison with classical neural networks and SVI, an industry standard parametrization for implied volatility. 
The operator deep smoothing approach thus opens up the use of neural networks on large historical datasets in financial engineering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
**Disclaimer**: I have no background on financial engineering so I cannot comment on the originality/significance aspect of this work. I might have missed essential points and only base my review on technical soundness and empirical validation

This paper introduces a method to nowcast implied volatility based on neural operators. This is achieved by a graph neural operator with MLPs as the lifting/projection layers and the kernel functions. The model uses a composite loss function that includes the fitting loss, no-arbitrage constraints, enforcing monotonicity and regularization terms to ensure smoothness. The method improves over the baseline SVI model.

### Strengths
The proposed method appears to be an interesting approach to model implied volatility surfaces for option pricing. The method through composite loss functions that include constraints such as no-arbitrage allows for direct training of a neural network architecture that directly provides smooth surfaces that directly satisfy these constraints. While I cannot comment directly how difficult/hands-on other alternative approaches are in this domain, the proposed model appears to have the flexibility to introduce other loss terms or vary the graph structure to incorporate other structural properties directly into the model. The model leads to lower errors compared to the SVI method.

### Weaknesses
The paper gives a good detail on the exact instantiation and implementation of the proposed method. I understand that there might be little prior work to base this paper on. However, it is unclear how the exact choices have been made and what would be their impact if changed. Note that the choices here might be purely made by domain knowledge, which I cannot judge and therefore might miss the specific considerations made here. However, I think the reader would benefit from more detail on the following: 

**In-Neighborhood sets**: The choice of in-heighborhood sets is arguably key model choice in this work. However, this work lacks detail on how the choice has been made and the subsampling heuristic has been devised. The authors argue that the choice is made because volatility smoothing requires limited global information exchange. I would then expect that the error of this method would go up if larger neighborhood sets are used. This paper would benefit from an additional analysis on changing the $\bar{p}$ and $K$ parameters. 

**Other model choices**: Again, there is little information about all the other model choices and/or instantiations. In particular, the choice of the kernel functions as MLP (as opposed to other choices) would be interesting for the reader. I'm also wondering whether the MLP can output negative values? I understand that there is a softplus operation in the architecture and ensures positive values but again the choice of a MLP would benefit from further explanation/ablation. 

**Loss function weighting coefficients:** I would kindly ask the authors to include more detail how the weighting of the loss function has been performed. As it is stated right now, there is no information on how to select/adjust the weightings on a new pr.oblem. It would help the dissemination the method if the authors would share their methodology.

**Ablations**: The claims in the paper could be strengthened by providing ablations In particular, different (wider) choices of the graph neighborhood would justify the choice of the authors. Additionally, omitting the no-arbitrage term and any of the other auxiliary terms would give more insight on the effect of the individual chosen components. 

I would reconsider my score if additional justification on the specific choices are included in the paper and the additional ablations on in-neighborhood and loss function weighting are performed.

### Questions
See Weaknesses section.

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
2

### Summary
This paper proposes a novel approach to nowcast implied volatility using neural operators, advancing the technique of implied volatility smoothing by constructing consistent, smooth surfaces that reflect current option market prices. Unlike traditional machine learning models, which struggle with dynamically changing spatial configurations in option price data, this work leverages a graph neural operator architecture to achieve robust and accurate predictions. Extensive benchmarks showcase that this work achieves superior generalization and accuracy over conventional neural networks and the SVI parametrization.

### Strengths
1. This paper is well-written and easy to follow.

2. This proposed method is resilient to input subsampling, aligns with no-arbitrage conditions, and eliminates the need for extensive data pre-processing.

### Weaknesses
1. The contribution is limited. It is focused on applying the Graph Neural Operator (GNO) architecture to the specific task of nowcasting implied volatility, without modifications to the operator itself.

2. It's important to discuss various neural operators and clarify why the Graph Neural Operator (GNO) was chosen for this task.

### Questions
please refer to the weaknesses part

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel approach to smoothing implied volatility using neural operators, focusing on Graph Neural Operator (GNO) architectures. Unlike traditional parametric models or classical neural networks, the proposed operator deep smoothing method directly maps observed implied volatilities to smooth, arbitrage-free surfaces, eliminating the need for instance-by-instance recalibration. By harnessing the unique capability of neural operators to handle data of varying sizes and spatial arrangements, the model effectively addresses the challenges posed by the dynamic and irregular nature of real-world financial data. Validated on a decade of intraday S&P 500 options data, the method outperforms the SVI benchmark and other machine learning techniques while generalizing effectively to unseen datasets, including end-of-day options for other indices, without retraining.

### Strengths
- **Novelty:** This paper proposes a novel application of GNO architectures to implied volatility smoothing, a task traditionally reliant on parametric models like SVI. By leveraging the discretization-invariance of neural operators, the method effectively addresses the challenges of dynamic and irregular financial data, marking a significant step forward in financial engineering.

- **Practical Significance:** The operator deep smoothing approach eliminates the need for instance-by-instance recalibration, streamlining the online calibration process. This drastically reduces computational overhead, making the method highly practical for real-time applications in trading, risk management, and other financial operations.

- **Robust Validation:** The method is extensively validated on a decade of intraday S&P 500 options data, demonstrating superior accuracy compared to the SVI benchmark and other machine learning techniques. Its strong generalization performance on unseen datasets, including end-of-day options for other indices, further highlights its robustness and adaptability.

 - **Reproducibility:** The authors provide open-source code, model weights, and detailed implementation details, including architecture, loss functions, and training setups. These resources ensure the experiments are easy to replicate and extend.

### Weaknesses
- **Limited Benchmark Comparisons:** The paper benchmarks its approach against SVI [1] and Ackerer et al. [2] but does not include comparisons with other key methods, such as SSVI [3] and VAE-based approaches [4]. Incorporating these would provide a more comprehensive evaluation. Additionally, using synthetic data, as in [2], could further strengthen the experimental validation.

 - **Insufficient Analysis of Computational Efficiency:** While the paper highlights the elimination of instance-by-instance recalibration, it lacks a detailed discussion on computational complexity, including runtime and memory requirements. These metrics are crucial for assessing the method’s practicality in high-frequency financial contexts.

 - **Abstract Treatment of Discretization-Invariance:** The explanation of discretization-invariance is largely theoretical. Providing concrete financial scenarios, such as handling abrupt market shifts, would better illustrate its practical significance and strengthen the narrative.

- **Clarity of Experimental Figures**: Figures 3 and 4 have unclear legends and insufficient annotations, which reduce their interpretability. Improving their clarity and labeling would make the results more accessible and impactful.

[1] A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives.

[2] Deep Smoothing of the Implied Volatility Surface.

[3] Arbitrage-free SVI volatility surfaces.

[4] Variational Autoencoders: A Hands-Off Approach to Volatility.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
