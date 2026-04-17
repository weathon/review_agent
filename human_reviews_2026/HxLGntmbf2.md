# The Centroid Affinity Hypothesis: How Deep Network Represent Features

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Understanding and identifying the features of the input space a deep network (DN) extracts to form its outputs is a focal point of interpretability research, as it enables the reliable deployment of DNs. The current prevailing strategy of operating under the linear representation hypothesis (LRH) -- where features are characterised by directions in a DN's latent space -- is limited in its capacity to identify features relevant to the behaviour of components of the DN (e.g. a neuron or a layer). In this paper, we introduce the centroid affinity hypothesis (CAH) as a strategy through which to identify these features grounded in the behaviour of the DN’s components. We theoretically develop the CAH by exploring how continuous piecewise affine DNs -- such as those using the ReLU activation function -- influence the geometry of regions of the input space. In particular, we show that the centroids of a DN -- which are vector summarisations of the DN's Jacobians -- form affine subspaces to extract features of the input space. Importantly, we can continue to utilise LRH-derived tools, such as sparse autoencoders, to study features through the CAH, along with novel CAH-derived tools. We perform an array of experiments demonstrating how interpretability under the CAH compares to interpretability under the LRH: We can obtain sparser feature dictionaries from the DINO vision transformers that perform better on downstream tasks. We can directly identify neurons in circuits of GPT2-Large. We can train probes on Llama-3.1-8B that better capture the action of generating truthful statements.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes the Centroid Affinity Hypothesis (CAH) as an alternative framework to the Linear Representation Hypothesis (LRH) for identifying features in deep neural networks. The core claim is that functionally relevant features, which are actually used by network components rather than merely represented, can be identified through centroids (sums of Jacobian rows) that form affine subspaces. The theoretical foundation is based on spline theory and the functional geometry of continuous piecewise affine (CPA) networks, suggesting that aligned linear regions correspond to features that the network actively utilizes. The authors validate CAH through experiments on toy problems, DINO vision transformers, GPT-2, and Llama-3.1-8B, claiming improvements in feature sparsity, generalization, and circuit discovery over LRH-based methods.

### Strengths
Strengths:

1. The limitation that LRH abstracts away from individual network components is legitimate and widely recognized in the interpretability community. The distinction between "extracted" versus "utilized" features is conceptually valuable.

2.  The grounding in spline theory and functional geometry is rigorous.

### Weaknesses
Weaknesses:

1.  Figure 1 Lacks Critical Methodological Details: The "Functional Geometry" visualization using SplineCam doesn't specify which layer, which hyperplanes are shown. While Section 3.2 suggests these are real experimental results (not illustrations), the artistic rendering and lack of detail make reproducibility and understanding difficult. 

2. Section 2 is notation-heavy, which makes it difficult to follow. Motivation for functional geometry is lacking and requires detailed discussion. 

3. The main paper provides an informal theorem to show that the functionally relevant features of a deep network sub-component are represented by the affine structures in its corresponding centroids and defers the formal theorem and proof to the appendix. 

4. Proposition 3.2 (Formal Proof for Theorem 3.3) shows three specific centroids $(\mu_1, \mu_2, \mu_3)$ have a geometric relationship where $\mu_3$ deviates from the line through $\mu_1,\mu_2$ by an amount proportional to $\sqrt{2\epsilon}$. To say centroids form "an affine subspace," authors need to show that there exists a consistent affine coordinate system where ALL centroids in the feature can be expressed. The last equation in the formal proof has a sign issue. Using the provided information, we get
\begin{equation} \mu^{(l)}_3=  \mu^{(l)}_2   + || n_2||_2 \cos(\theta)d + ||n_2||_2 \sin(\theta) \hat{u}, \end{equation}
But the paper has the opposite sign for the second and third terms. Please clarify.

### Questions
Questions to Authors:
1. In Figure 1, for the "Functional Geometry" plot, which layer's geometry is visualized? What are the SplineCam visualization parameters? Are all linear regions shown or a subset?

2. Power diagram formulation lacks motivation and needs discussion on why this parameterization is required. 

3. What happens with different neighborhood sizes ($\epsilon$ in $B_{\epsilon(x)}$)?

4. Please answer the questions in the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a new method to investigate how deep nets represent features. The key contribution is Centroid affinity hypothesis (CAH), which states that the features of the input space that a DN uses are characterised by affine subspaces of centroids. This is different than the traditional, and widely-used Linear representation hypothesis (LRH) which focuses on latent activations. 

The paper demonstrates how this CAH makes sense by using an example of polygon interior classification. The paper also demonstrates that using CAH as feature representations instead of latent activations can achieve higher probing accuracy in some experiments. 

CAH can also be used to estimate the attribution of a neuron to the features of a collection of samples. To that end, the paper demonstrates how using CAH to identify a neuron for article 'an' in GPT-2.

### Strengths
* This paper introduces a novel and interesting hypothesis (CAH) for how features are represented in deep networks. The paper points out the differences between CAH and the traditional LRH. 

* The experimental results are good to show that centroids are more useful than latent attentions for probing. Especially, the experiment that CAH can be used to identify the neuron for article 'an' in GPT2 is interesting and seems helpful for circuit discovery on LLMs. 

* Although the technical details are difficult to follow, the implementation is simple and and thus it's quite straightforward to replicate the paper.

### Weaknesses
* The content can be challenging to understand. Some places can be made clearer. For instance, it would be helpful to state clearly that the example in section 3.2 (fig 2) is about classifying whether a pixel is in-or-outside the polygon. Also, it's unclear why the centroids in fig2 (right most) is different than the centroids in fig 1 (right most)

* The paper focuses only on ReLU. How about other activation functions? 

* The experiment results on probing accuracy may be not very convincing for why one should use CAH. Could simple noise reduction (e.g. applying PCA to latent activations) also work?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Centroid Affinity Hypothesis (CAH), a theoretical framework towards the analysis of the representation of features in deep networks. The core idea revolves around "centroids", i.e., vector summarisations of the Jacobians of the network. Unlike the Linear Representation Hypothesis (LRH), which assumes that features correspond to directions in latent space, CAH aims to connect features directly to the network’s functional geometry and its nonlinear components.

### Strengths
The paper offers a compelling geometric reinterpretation of feature representation in neural networks and provides a well motivated framework that connects spline-based geometry with interpretability.

### Weaknesses
My main concerns revolve around the experimental/empirical evaluation of the approach and any potential insights that might stem from it. 

*The empirical evaluation lacks depth*: The experiments primarily illustrate the CAH qualitatively and focus on reporting differences (e.g., higher sparsity, slightly better probe performance) without offering substantial interpretive insight. There is no rigorous statistical or diagnostic analysis to confirm that the CAH leads to more faithful or causally relevant features. Can the authors provide clearer empirical validation or quantitative tests of faithfulness and functional relevance?

Despite strong theoretical framing, the experiments do not yield new mechanistic understanding of models. The DINO and GPT2 results largely restate known patterns or replicate existing findings under the new representation. What unique interpretability or mechanistic insight does CAH enable that previous frameworks could not?

The linear probe and sparse autoencoder tests rely on standard benchmarks but lack baselines, error bars, or ablation results. There is no clarity on how results generalize across runs or hyperparameter choices. Could the authors clarify whether CAH performance differences persist under varying conditions? What happens when the full ImageNet is considered instead of the small Imagenette?

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2
