# I Predict Therefore I Am: Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data?

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Recent empirical evidence shows that LLM representations encode human-interpretable concepts. Nevertheless, the mechanisms by which these representations emerge remain largely unexplored. To shed further light on this, we introduce a novel generative model that generates tokens on the basis of such concepts formulated as latent discrete variables. Under mild conditions, even when the mapping from the latent space to the observed space is non-invertible, we establish rigorous identifiability result: the representations learned by LLMs through next-token prediction can be approximately modeled as the logarithm of the posterior probabilities of these latent discrete concepts given input context, up to an invertible linear transformation. This theoretical finding: 1) provides evidence that LLMs capture essential underlying generative factors, 2) offers a unified and principled perspective for understanding the linear representation hypothesis, and 3) motivates a theoretically grounded approach for evaluating sparse autoencoders. Empirically, we validate our theoretical results through evaluations on both simulation data and the Pythia, Llama, and DeepSeek model families.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors study why next-token prediction (NTP) models appear to learn human‑interpretable concepts. The authors posit a discrete latent-variable generative model in which concepts c (discrete, possibly with arbitrary causal structure) generate input x and target y. Under a diversity condition and without requiring invertibility of the mapping from c to observations, they prove an identifiability result: the model’s representation f_x(x) is (approximately) an invertible linear transform of \log p(c \mid x) up to additive terms. This provides a unified account of “linear representation” phenomena (steering vectors, linear probing) and motivates a principled evaluation of SAEs by comparing their features to linear-probe estimates of concept posteriors. The empirical section includes synthetic validations and tests across Pythia, Llama, and DeepSeek families.

### Strengths
* Clear theoretical link from NTP to concept posteriors via a linear transform (Theorem 3.1), with corollaries explaining steering/probing phenomena.
* Actionable implication for evaluating SAEs by benchmarking against linear-probe posterior estimates.
* Good validation practices: synthetic and multiple model families (Pythia, Llama, DeepSeek).

### Weaknesses
* Assumptions strength: The diversity condition is nontrivial and may not hold; more intuition or empirical checks would help.
* Residual terms: The correction term in Theorem 3.1 (involving h_y) complicates the neat linear story; Can you quantify when is it negligible?
* Empirical depth: SAE evaluation is promising but would benefit from a broader battery (multiple SAE variants, structured sparsity, stability across seeds).

### Questions
* Can you provide sufficient conditions (in practice) under which the diversity condition holds for token distributions in large corpora?  
* How large is the h_y correction term in typical NTP models? Can you report empirical magnitudes and the impact on linearity fits?
* In the SAE evaluation recipe, how sensitive are conclusions to the quality of linear probes (regularization, class imbalance, definition of concept labels)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a theoretical framework connecting next-token prediction with latent variable identifiability. It argues that the representation in large language models approximates a linear transformation of the log posterior of latent concepts. The authors derive this relationship analytically and validate it through small-scale experiments showing linear alignment between counterfactual representation differences and classifier weights.

### Strengths
* The paper provides an interesting conceptual link between next-token prediction, linear probing, and identifiability theory.
* It provides a single, unified mathematical explanation for a diverse set of empirical phenomena (linear probing, concept steering, and vector arithmetic).
* They test their theory on synthetic data where the ground truth $c$ is known, and then successfully test its corollaries (e.g., $A_s W_s \approx I$) across multiple LLM families and sizes.

### Weaknesses
The paper presents an interesting finding that LLM representations could be linear latent concept posteriors. However, some assumptions may be too strong and make the analysis flawed.
*  The main theorem (Theorem 3.1) critically depends on the invertibility of the matrix $\hat{L}=[f_{y}(y_{1})-f_{y}(y_{0}), ..., f_{y}(y_{l})-f_{y}(y_{0})]$, which implies that the representation dimension d must equal the total number of latent configurations $l$. In practice, $l$ grows combinatorially explosive (e.g., $2^m$ for m binary concepts), while d (e.g., 4096) is a fixed hyperparameter. This d<<l seems to invalidate the proof. 
* Also, the diversity condition, requires the existence of $l+1$ distinct values of y for the matrices L and $\hat{L}$ to be constructed.  $l$ is combinatorially explosive (e.g., $2^m$ for m binary concepts) and will be much larger than the vocabulary size |V| for any non-trivial concept space. And it is impossible to find $l+1$ if $l$ > |V|.
* The latent variable model is  $p(x,y)=\sum_{c}p(x|c)p(y|c)p(c)$. This assumes that the context $x$ and the next token $y$ are conditionally independent given the latent concept $c$. This is a simplification that breaks the autoregressive nature of language, where $x$ (past tokens) is the direct causal parent of $y$ (the next token).

### Questions
* Could the authors please clarify how the theory (theorem 3.1) can hold in the realistic setting where $d \ll l$?
* The theory requires $l+1$ distinct $y$ tokens. How can this condition possibly be satisfied in practice, where $l$ is combinatorially large and $l \gg |V|$?
* The latent variable model assumes $x$ and $y$ are conditionally independent given $c$. How does this simplification, which breaks the autoregressive nature of language, where $x$ (past tokens) is the direct causal parent of $y$ (the next token), affect the main claim? Would the linear identifiability result still hold under a more standard model?

### Soundness
2

### Presentation
3

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
This paper theoretically shows that next-token prediction leads to linear identifiability of data-generating latents under a fairly general data model. In particular, such identifiability could be obtained when the generation function is non-linear and non-invertible, and when the latents are discrete and non-Boolean. Motivated by this result, the paper proposes improved techniques for evaluating and training sparse autoencoders and empirically evaluates the proposed method both in synthetic settings and on practical LLMs.

### Strengths
- Compared to existing analysis, the identifiability result proved by this paper is more general; the introduced diversity condition makes sense to me and is likely to hold in practical pre-training of LLMs. Showing approximated identifiability in the non-invertible case is also a decent complement to the results.
- The paper translates the theoretical results to practical algorithms on SAEs.
- The paper is well-written and easy to follow.

### Weaknesses
- A major drawback of the results, like existing ones, is that linearity is defined only for the output activations (i.e., in the unembedding space) of LLMs. Yet, prior empirical work on the linear representation hypothesis often shows that linearity is usually formed in the _intermediate_ layers [1, 2]. For some concepts, such intermediate-layer linearity could even fade away at the output layer [3], which cannot be captured by the proved results. Although I agree with the authors that rigorously analyzing linearity for intermediate layers is (much) more difficult, I feel it is still necessary to mention it here since it may be of more interest for both theorists and practitioners (e.g., SAEs are often trained on intermediate-layer outputs rather than last-layer outputs), in my opinion.
- The relation between theoretical results and the proposed structured SAE is rather loose (see Questions for more details).
- (Minor) Missing related work: [4] also proves linear identifiability of latent variables under a general setting and discusses the implications for the linear representation hypothesis.

---

[1] Language models represent space and time. ICLR, 2024.

[2] Emergent linear representations in world models of self-supervised sequence models. arXiv preprint arXiv:2309.00941, 2023.

[3] The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. COLM, 2024.

[4] When do neural networks learn world models? ICML, 2025.

### Questions
- While discrete concepts are beneficial for modeling text semantics such as topics, there may also be continuous concepts in the text, such as those relevant to numbers. Can the identifiability result be extended to modeling both discrete and continuous latents?
- What is the difference between the proposed SAE evaluation approach and existing probing approaches, e.g., the one used in [5]?
- Why does low-rank regularization help model the interdependence between latent variables/concepts in structured SAE?

---

[5] Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093, 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper models text generation via discrete, human-interpretable concepts and shows, under a diversity/invertibility condition, that accurate next-token predictors' internal features are (up to an invertible linear map and bias) equal to the log posterior over those concepts. This yields a unified explanation for linear probing, steering, and "concept directions," and motivates a practical test (alignment between probe weights and concept directions). The authors add small-scale simulations, apply the alignment test to several LLMs with hand-crafted counterfactual pairs, and propose a structured sparse-autoencoder (SAE) plus a probe-based evaluation that outperform common SAE baselines on correlation/MSE.

### Strengths
**Unifying lens on ``linear representations.”:** The same matrix $A$ explains concept directions, steering, and probe linearity, which clarifies why these tricks often work together.

**Simple checks in evals:** The $A_s W_s \approx I$ is a neat diagnostic. I haven’t seen exactly this identity‑product visualization before. 

**Practical tie‑in to SAEs:** The theory suggests a supervised reference for evaluating SAE features and motivates a structured (sparse+low‑rank) variant that shows consistent gains

**Cross‑model qualitative replication of concept–probe alignment**

### Weaknesses
**The diversity condition seems very strong and under-discussed**
From what I understand, in Appendix D, the paper implicitly assumes $f_x(x), f_y(y) \in \mathbb{R}^\ell$ so that $\hat{L} \in \mathbb{R}^{\ell \times \ell}$ exists with columns  $f_y(y_j) - f_y(y_0).$  

That means the representation dimension must be at least  $\ell = \prod_k n_k$  (the number of joint latent configurations). This seems like a very strong and unrealistic assumption and the manuscript never states this dimensionality constraint explicitly.  

To illustrate, even for a toy model with just 20 binary concepts ($k=20, n_k=2$), $\ell = 2^{20} \approx 1 \text{ million}$. The representation dimension $d$ of a real LLM is large, but fixed (e.g., $d=4096$ or $d=8192$). In any realistic scenario, the representation dimension $d$ is *far smaller* than the number of possible joint concepts $\ell$ ($d \ll \ell$).

If $d = \mathrm{dim}\, f_x \neq \ell,$ you need to formulate everything on the span of  $\{ f_y(y_j) - f_y(y_0) \}$ and use for example the Moore–Penrose pseudoinverse on a $d \times \ell$ matrix; otherwise the shapes don’t line up (you would have a non-square matrix). As written, the proof seems to treat $\hat{L}$ as square without justifying $d = \ell.$ (Same comment for $L$ as well.) The authors state in Appendix C that this is a ``weak assumption" so please correct me if I'm not understanding something here. I think the proof should go through with a pseudo-inverse but I haven't worked out the details. 

**Again in appendix D, on the approximately invertible part:**
I am not sure if the argument that the remainder term $\mathbf{h}_y$ vanishes (Pg 20) is correct.

The term is $h_{y_j} = \mathbb{E}_{p(\mathbf{c} \mid y_j)} [\log p(\mathbf{c} \mid \mathbf{x}, y_j)]$. The proof argues this vanishes because $p(\mathbf{c} \mid \mathbf{x}, y_j)$ concentrates at a single $\mathbf{c}^*$ (the $\epsilon$-invertibility assumption).

However, the expectation is taken with respect to $p(\mathbf{c} \mid y_j)$, not $p(\mathbf{c} \mid \mathbf{x}, y_j)$. If I split the expectation:
$$
h_{y_j} = p(\mathbf{c}^* \mid y_j) \log p(\mathbf{c}^* \mid \mathbf{x}, y_j) +  \sum^{\mathbf{c} \neq \mathbf{c}^*} p(\mathbf{c} \mid y_j) \log p(\mathbf{c} \mid \mathbf{x}, y_j)
$$

As $\epsilon \to 0$, we have $\log p(\mathbf{c}^* \mid \mathbf{x}, y_j) \approx \log(1-\epsilon) \approx 0$, but $\log p(\mathbf{c} \neq \mathbf{c}^* \mid \mathbf{x}, y_j) \approx \log \epsilon$.

This seems to imply that the expression for $h_{y_j}$ becomes:
$$
h_{y_j} \approx \sum^{\mathbf{c} \neq \mathbf{c}^*} p(\mathbf{c} \mid y_j) (\log \epsilon) 
$$

$$ = (1 - p(\mathbf{c}^* \mid y_j)) \log \epsilon $$

This term clearly depends on $y_j$ (via $p(\mathbf{c}^* \mid y_j)$) and diverges to $-\infty$ as $\epsilon \to 0$.

Therefore, the difference $h_{y_j} - h_{y_0}$ seems to be:
$$
h_{y_j} - h_{y_0} \approx \big( p(\mathbf{c}^* \mid y_0) - p(\mathbf{c}^* \mid y_j) \big) \cdot \log \epsilon
$$

Instead of vanishing, this term seems to diverge, given that the diversity condition requires $p(\mathbf{c} \mid y_0) \neq p(\mathbf{c} \mid y_j)$. The argument on Pg 20 that $\mathbb{E}[\log p(\mathbf{c} \mid \mathbf{x}, y)] \approx \log \epsilon$ "not depending on $y$" appears to miss the $(1 - p(\mathbf{c}^* \mid y_j))$ factor.

This feels like a significant error in the proof, so I wanted to check if I'm missing a different assumption. To make this term vanish, it seems you would need to assume $p(\mathbf{c}^* \mid y_j)$ is constant for all $j=0...\ell$, which seems like a very strong constraint that might conflict with the diversity condition.


**Some concerns about experimental evals**
Using the same pairs to build both $A_s$ (directions) and $W_s$ (probes) can inflate diagonal alignment. There should at least be some reporting over held-out concepts or perform cross-validation, and include null baselines (say, randomly permuted labels) to show that the effect isn’t a trivial consequence of using the same supervision twice.  

Also, quantifying ``approximate identity” by reporting, layer-wise and across seeds, something like diagonal/off-diagonal gap might help (just an idea).

**Contextualize Structured SAEs**
I like the Structured SAEs idea but low‑rank‑plus‑sparse decompositions with a nuclear‑norm regularizer have a deep literature (e.g., Candes et al., ``Robust PCA”; many variants since). Bringing L+S to SAEs is reasonable but should be contextualized and credited; please cite this prior line of work explicitly (no relation to me) and clarify what is new here (architecture? training recipe? results on LM activations?)

### Questions
For the "Structured SAE," the authors chose nuclear norm regularization to model concept dependencies. What was the intuition behind this specific choice (implying a low-rank structure on concept activations) versus other forms of structured regularization, such as imposing a graphical model prior on the features $\mathbf{z}$ or using a group-lasso penalty? Just a curiosity rather than a weakness of the paper.

### Soundness
2

### Presentation
3

### Contribution
3
