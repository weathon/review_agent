# Geometry-aware Euclidean Diffusion Language Generation

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 4, 0

## Abstract
We formulate a powerful generative framework for language, premised upon modeling discrete token generation as continuous trajectories of a Gaussian stochastic process in a Euclidean space. Specifically, to address the challenge of the high-dimensional discrete nature inherent in language data, we devise two core components: a projection function to embed discrete tokens into a continuous domain and a metric function to infer the conditional probability distribution of subsequent tokens from continuous embeddings.  Subsequently, we employ a forward diffusion process that incrementally perturbs the data distribution towards a tractable standard Gaussian prior. To learn the generative reverse process, we formulate a novel \textit{data geometry-aware score} that explicitly exploits the inherent manifold structure of the discrete language data to refine the fidelity of the score approximation. Since direct optimization of the score function is intractable, we propose optimizing a tractable surrogate objective, the Relaxed Evidence Lower Bound (RELBO), which ensures a bounded approximation error via continuous relaxation. Finally, we critically reassess conventional evaluation protocols and introduce a novel comprehension score, designed to enable a more robust and equitable performance comparison against competing architectures. Empirical validation on the LM1B and OpenWebText benchmarks corroborates the effectiveness of our proposed framework. Our results significantly outperform SOTA discrete diffusion and autoregressive (AR) schemes, indicating a very promising direction for language modeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a variation of diffusion modeling for natural language.  
To leverage the strengths of continuous diffusion models, the authors propose techniques to apply these models to inherently discrete language data.  
A continuous projection operator, defined by an embedding dictionary, is used to map discrete data into a continuous space, and this projection is incorporated into the formulation of the diffusion ELBO loss.  
Additionally, a relaxed version of the ELBO is introduced to enable optimization via standard gradient backpropagation.  
Furthermore, to complement the conventional Perplexity metric, the authors introduce the Perceptual Score---a metric that uses an LLM as a proxy for human evaluation.  
Experiments are conducted on the LM1B and OpenWebText datasets, and the proposed methods are compared to existing approaches.

### Strengths
1. The paper tackles the important topic of designing diffusion models for discrete data such as natural language, and explores new approaches to data geometry and modeling in both discrete and continuous domains.
2. The proposed method, being continuous, has the potential to be combined with consistency modeling and discriminative score distillation, which is a clear advantage as mentioned in Section 4.

### Weaknesses
1. Key mathematical symbols and terms are not properly defined, and several equations lack sufficient explanation or correct notation. Examples of unclear variables are as follows:
    * The driving noises B of SDE in Eq. 1, 2, around line 122, and in Table 1.
    * $\mathbf{e}^g$ around line 145.
    * $\tilde{p}$ around line 160.
    * $\mathbf{n}$ in Eq. 14.

2. The derivation and presentation of important equations (e.g., Eq. 4, 5, 13–14) are incomplete or confusing. Providing derivations or relevant references would enhance the clarity of the paper.
    * Regarding the RELBO in Eq. 10, it should be mentioned that the derivation is described in the appendix.

3. The novelty of the proposed method compared to existing discretization approaches is not clearly established.
   In particular, the discussion of data geometry is unconvincing, and the claimed use of Riemannian components, in Table 1, is not clearly demonstrated.

4. The explanation of the ranking metric in Appendix E is insufficient, lacking a clear natural language description and details about LLM prompts.
  Although python scripts for the metric are provided, basic information such as which LLM is used and what prompts are used should, in principle, be described in the paper.
  In addition, it seems necessary to provide evidence for the effectiveness of the proposed metric based on LLMs, such as its correlation with human evaluation metrics.

5. Although perplexity is not a perfect metric, the results for the proposed method in Table 3 are notably poor. It would be helpful to include some discussion on the possible reasons for this and the limitations of perplexity as an evaluation metric regarding this point.

6. In Figure 2, it is unclear why the red dashed line is ideal.

7. There are some typos:
    * in line 173, "can be achieve" should be "can be achieved."
    * unnecessary parentheses around line 100.
    * The explanation of $\epsilon_{\perp}$ and $\epsilon_{\parallel}$ around line 265 is confusing and contains typos.

### Questions
1. Regarding Weaknesses 1 and 2, could you elaborate unclear notations and derivations of equations?
2. Regarding Weakness 3: Could you provide further explanation on the difference of the proposed method from previous methods? What is the Riemannian metric considered in the paper?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a generative framework that models discrete token generation as continuous trajectories of a Gaussian stochastic process within a Euclidean space. To address the challenges posed by high-dimensional discrete data, the authors introduce two core components: a projection function to embed discrete tokens into a continuous domain, and a metric function to infer the conditional probability distribution of subsequent tokens from continuous embeddings.

Furthermore, the paper formulates a novel geometry-aware score that explicitly exploits the inherent manifold structure of discrete language data to refine the fidelity of score approximation. Since direct optimization of the score function is intractable, the authors propose optimizing a tractable surrogate objective, the Relaxed Evidence Lower Bound (RELBO), which ensures a bounded approximation error via continuous relaxation.

Section 3.3 discusses specific model specifications, including the design of the continuous projection function $P^{(c)}(\cdot)$, SDE customization, and the reverse integral form. Empirical validation on the LM1B and OpenWebText benchmarks demonstrates the effectiveness of the proposed framework, showing performance competitive with state-of-the-art models.

### Strengths
- **Bridging discrete and continuous domains**: The paper proposes geometry-aware score matching techniques for handling discrete token generation via Euclidean diffusion, achieving Gaussian diffusion language generation that recognizes the inherent structure of discrete language data and provides a theoretically consistent framework.

- **Compatibility with editing and non-monotonic generation**: Unlike auto-regressive models that commit to a fixed prefix, diffusion models can refine the entire sequence representation simultaneously, enabling non-monotonic generation and providing unprecedented control over the generative process, including insertion, editing, and constrained generation.

- **Concrete guidelines for model specifications**: Section 3.3 provides practical implementation guidelines, including theoretical analysis of projection density (Eq. 12), embedding constraints via hypersphere regularization, and SNR curve design.

- **Competitive performance in initial experiments**: On the LM1B dataset, the proposed method achieves a 100% win rate against RDLM and a Ranking score of 0.51. On OpenWebText, with only 17% of training steps, the method achieves comparable results to MDLM and SEDD, and outperforms some auto-regressive Transformer outputs.

- **Potential for future improvements**: Being implemented as a continuous model, the framework can directly leverage mature image diffusion techniques (e.g., consistency training, discriminative score distillation) for further performance enhancements.

### Weaknesses
### 1. Concerns regarding the reliability of evaluation metrics

The proposed Perceptual Score (Ranking metric) uses LLMs as proxies for human experts, but lacks sufficient validation of reliability in the following aspects:

- Automatic evaluation by LLMs can be influenced by presentation order and writing style, potentially biasing results in favor of the proposed method
- Details about the LLM model name, prompt design, and evaluation criteria are unclear, making reproducibility verification difficult

**Suggestion**: The robustness of the evaluation should be reinforced through correlation analysis with human evaluation and the use of additional automatic evaluation metrics.

### 2. Lack of empirical evidence for scaling laws

Due to computational constraints, scaling laws with varying model sizes and training data amounts have not been empirically validated. The OpenWebText experiment was completed with only 0.17M steps, leaving the long-term training behavior unclear.

**Suggestion**: Even at small scales, performance trends across different parameter counts should be demonstrated to provide initial evidence of scaling characteristics.

### 3. Absence of ablation studies

While geometry-aware score matching techniques and RELBO are highlighted as main contributions, the individual contributions of each component have not been validated. The loss function consists of cross-entropy and $L_2$ distance terms, but the impact of weighting these terms is unclear.

**Suggestion**: At minimum, comparative experiments should be conducted on: the presence/absence of geometry-aware score, weighting of each loss term, and comparison between RELBO and standard ELBO.

### 4. Insufficient experimental validation of model specifications

Section 3.3 presents detailed theoretical analyses including embedding density analysis, hypersphere regularization, and SNR curve design, but lacks experimental validation of how these design choices contribute to final performance.

**Suggestion**: Experiments are needed comparing: with/without hypersphere regularization, performance across different embedding dimensions, and variations in SNR design.

### Questions
### Q1. Response to weaknesses

Could the authors provide their perspectives on the four weaknesses mentioned above? Specifically:

1. **Evaluation metrics**: Validation of bias mitigation strategies for the Ranking metric and plans for using additional evaluation metrics
2. **Scaling**: Plans for conducting small-scale scaling experiments
3. **Ablation study**: Plans for additional experiments to quantify the individual contributions of each component
4. **Model specifications**: Experimental evidence demonstrating that the design choices in Section 3.3 contribute to performance improvements

### Q2. Applicability of image diffusion techniques

The paper states that as a continuous model, the proposed method can directly apply mature image diffusion techniques. However, the data estimator is $P_{\theta}\^{(c)}(\hat{z}_{\theta})$, which appears structurally different from conventional image diffusion models.

**Question**: Despite this architectural difference, can consistency training and distillation techniques be applied without issues? If modifications are necessary, what kinds of adaptations would be required?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a diffusion language model that performs denoising in a continuous embedding space for discrete text tokens. During training, it uses two losses. First, a cross-entropy loss on the logits ($p_\theta(z \mid x_t)$), which are the model outputs before discretization. Second, a MSE loss after snapping (discretization): take the logits, discretize to a one-hot token, then project back to the embedding space with $P^{(c)}$; the MSE matches this re-embedded token to the ground-truth embedding. These losses encourage cycle consistency between the discrete and continuous spaces, i.e., ($z \approx P^{(r)}(P^{(c)}(z))$) and ($x \approx P^{(c)}(P^{(r)}(x))$), where $P^{(c)}$ maps tokens -> embeddings and $P^{(r)}$ maps embeddings -> token distributions.

During inference, it starts from a prior noise state in the embedding space. At each step, the model predicts logits, then discretizes them to a one-hot token. This is either probabilistically (sampling from softmax) or deterministically (argmax). The chosen token is then projected back to the continuous embedding space with $P^{(c)}$ to form the geometry-aware clean estimate used in the reverse diffusion update. After the final step, the method outputs the discrete token sequence.

The method shows competitive performance to the diffusion language model baselines with much fewer training iterations, and surpassing both perceptual quality using the llm-as-judge. However, they have shown that perplexity does not show the good results, as the authors has posit that the confidence of the model is not aligned well with the quality of the prediction.

### Strengths
1. The paper explores the possibility of Diffusion LLM that works not in the discrete space but in the continuous space, which better aligns with the original Diffusion setups.

### Weaknesses
1. **Relation to DDIM is under-discussed.** Both LanGeo and DDIM (Song et al., ICLR 2021) convert the intermediate latent into a "clean" representation and then re-inject noise to continue the trajectory. The high-level procedure appears conceptually similar, but the paper does not position LanGeo clearly with respect to DDIM or explain the differences.
2. **perplexity vs. perceptual quality claim is weakly supported.** The paper asserts that perplexity is not well-aligned with perceptual quality, but provides limited citations and concrete examples. In practice, if the reference AR model used to compute perplexity is very strong (e.g., Qwen3), the misalignment may be smaller. Moreover, the proposed LLM-as-Judge ranking metric is hard to interpret: a score of 1.0 vs. 0.51 does not imply "2× better", and the comparison is further limited by evaluating against only two models in Table 2, which reduces statistical strength and external validity. 
3. The paper does not explicitly specify which model was used as the LLM-as-Judge in the text.

### Questions
I am not deeply familiar with diffusion or language modeling, so I will carefully consider other reviewers’ comments before making a final decision. I do not have sufficient understanding to fully evaluate the significance of the proposed method. However, at present, I lean toward rejection, primarily because the evaluation protocol (the perplexity discussion and the ranking metric) does not convincingly reflect the actual performance of the method.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper studies diffusion language models, with continuous-state diffusion modeling on latent space. The proposed RELBO (relaxed ELBO) with triangle inequality realizes a differentiable loss on continuous-to-discrete transition at the end of inference. The authors also propose a way to determine the hyperparameters of continuous diffusion through the lens of signal-to-noise ratio. The proposed method is tested against LM1B and OpenWebText datasets, and shows a competitive performance among the continuous-state discrete diffusion models.

### Strengths
The authors point out that the current diffusion language models ignore the geometry of discrete data, which is an important point. To bridge this gap, the authors introduce a (perturbed) embedding and run a diffusion model over the embedding space, together with a relaxed objective (RELBO) with better regularity. This idea, with additional polishing, may lead to stronger diffusion language modeling.

### Weaknesses
My impression is that the draft is still not completed. There are so many typos and undedined math symbols, and even the main experiment is not finished. I suggest a thorough revision and submission to another venue. Let me mention some points that need improvement and/or clarification.
- [W1] **Incomplete experiment:** First of all, the main experiment on OpenWebText is not finished. I understand that the training requires heavy computation, but "we have only trained the model with only 0.17M steps on OpenWebText by the submission deadline. We will report a fully trained results after finishing the training." in L345 is simply an unfair use of the rebuttal revision. If the authors are going to update the "main result" during rebuttals, it is essentially an extention of submission deadline.
- [W2] **Inaccurate/incomplete presentations:** There are many expressions that are incorrect, ambiguous, or undefined.
    - [W2-1] The two equations in L91 for the solution of SDE should be valid only for a specific class of $f, g$.
    - [W2-2] In Equation 4, $L_t$ is comparing $\tilde{p}$ and $p$, which are probability distributions on different spaces (as they specify $D$ as KL in L160). I believe it requires correction.
    - [W2-3] For the derivation of Equation 11, while I checked the Appendix A.2, apprently it is not completed and L781-L787 does not make sense to me. For instance, what is $Y_t$ in L782? The sentence in L784 seems incomplete, and I do not see how Equation 32 is derived.
    - [W2-4] There are many math symbols without clear definitions in the paper, (at least) as listed below. I found the definitions of some of them in the appendix, but they are not readily available in the main body. (I might have just missed some definitions in the main body though.)
        - L146: What is $e^g$?
        - L153: What is $\tilde{p}$? Is this "$\to$" a definition of $\tilde{p}$?
        - L246: In which sense is "$n$" the diffusion dimension (single token or whole input or..)? What is its relation to $g$, $s$, $L$, which were introduced in Section 3.1?
        - L269: What is $n$ here?
        - L277: What is $O(\cdot)$ in Algorithm 2?
        - L280: What is $\hat{z}_\theta$ in Algorithm 1? What is $E$?
        - L307: What is $\tilde{n}$?
        - L312: What is $\omega$?
        - L782: What is $Y_t$?
    - [W2-5] There are also many typos or English errors, or even incomplete sentences. I strongly recommend some form of proofreading.
        - L54: some some -> some
        - L55: remanian -> Riemannian
        - L80: we extensive experiments -> we conduct extensive experiments
        - L89: $dB$ -> $dB_t$? Many of the time variables in SDE-like equations are missing; is there a reason?
        - L100: "masked diffusion models ()" what's inside the parentheses?
        - L101: transfer -> transfers
        - L120: "are designed based to directly synthesis the , e.g.,": This sentence looks broken.
        - L136: "the another" ??
        - L149: Is it $R^s$ or $R^L$? $P_r$ or $P^{(r)}$?
        - L168: What do the authors want to mean by $\{P(z)\}$? The set as it is a singleton set. I guess something like $\{P(z)\mid z\in Z\}$ was intended?
        - L174: achieve -> achieved
        - L175: determinists -> deterministic
        - L184: boundary -> bound
        - L192: A period is missing.
        - L197, L206: $dt$ is missing.
        - L218: before sections -> previous sections
        - L252: "Due to we regularize the $\hat{x}$ onto a hyper-sphere.": I don't think this is complete as a sentence..
        - L256: $\varphi$ indicate -> $\varphi$ indicates
        - L265: fix "$bm\epsilon$"
        - L265: sensible -> sensitive?
        - L288: exist -> existence
        - L312: $logp$ -> $\log p$

While the proposed idea of relaxed loss computation for discrete-continuous diffusion language modeling is interesting, I believe the current paper is not ready for a review process.

### Questions
I have the following questions.
- [Q1] L148: How can it be an invertible function with different dimensions? Do the authors assume a subset of Euclidean spaces in the input? If so, specify the set.
- [Q2] L235: What is "the domain size in a single dimension"? Are the authors talking about hypercube-like objects?
- [Q3] L239: What is "the occpancy range for a single class"? How is it defined mathematically?
- [Q4] L357: I have briefly read the ranking script in the supplementary material, but I wonder why the ranking takes values lower than zero, because in the script the ranking is defined as one-based indexing (so positive integers?). Or are they projected to the $[0, 1]$ interval afterwards?

The following are minor questions. I do not need answers for them but please incorporate them into future revision:
- [Q5] L172: What is "Kronecker delta transition function": it seems the authors itemize some choices of $K$ in L174, none of which is familiar with what I know as "Kronecker delta". Does it just mean a one-hot vector?
- [Q6] L187 etc: Can the authors add parentheses for the expectation operation for readability?
- [Q7] L219: "infinite classes"? "$N$ classes with arbitrarily large $N$" and "$\infty$ classes" are different; I think the authors consider the former?

### Soundness
1

### Presentation
1

### Contribution
2
