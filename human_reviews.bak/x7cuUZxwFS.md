# Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models

- Decision: Reject
- Scores: 3, 6, 5, 6

## Abstract
In spite of their huge success, transformer models remain difficult to scale in depth. In this work, we provide formulae that govern the moments of the forward and backward signal through all transformer components, and develop a unified signal propagation theory for transformers. Our framework can be used to understand and mitigate vanishing/exploding gradients, rank collapse, and instability associated with high attention scores. We also propose DeepScaleLM, an initialization and scaling scheme that conserves unit output/gradient moments throughout the model, enabling the training of very deep models with $100$s of layers. We find that transformer models could be much deeper -- our deep models improve $1.0$ points in perplexity, and $2.2$ points in downstream tasks compared to shallow models across multiple model sizes, without any extra parameters, and even outperform larger shallow models using only half the number of parameters.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Despite their success, scaling transformer models in depth remains challenging. This work introduces formulas governing signal moments in transformers, offering a unified signal propagation theory. In this paper, the proposed framework aids in addressing issues like vanishing/exploding gradients, rank collapse, and instability from high attention scores. We also propose DeepScaleLM, an initialization and scaling method conserving output/gradient moments, enabling deep model training. The proposed method improve deep narrow Bert's perplexity by 1.0 point and downstream task performance by 2.2 points compared to shallow models across various sizes, even outperforming larger shallow models with half the parameters.

### Strengths
1. Simple idea and it makes intuitive sense.
2. Creative combinations of various existing techniques.

### Weaknesses
1. Lack of novelty: scaling Residual/Skip-Connection is a well-known trick to stabilize training of deep neural networks, very similar ideas can be found at [1, 2]
2. Similar signal propagation idea is presented in [3]
3. The results are not surprising [4] shows exactly the same conclusion with similar model configs. 
4. The idea of preventing rank collapse has been thoroughly explored in [5,6]
5. Experiments on downstream tasks seem lack of diversity. More downstream tasks with different characteristics should be included.
6. Results on more modern architectures beyond Bert should be included to present a convincing argument.

Overall, most of the tricks are already well-studied and published, the paper generally feels incremental and results are not surprising. In combination with weak empirical studies on models with trivial sizes, this paper doesn't seem to be significant enough to be presented at the ICLR venue.

[1] Kai, Hu, et al. "Is normalization indispensable for training deep neural networks?." (Neurips 2020)
[2] Bachlechner, Thomas, et al. "Rezero is all you need: Fast convergence at large depth." Uncertainty in Artificial Intelligence. PMLR, 2021.
[3] He, Bobby, et al. "Deep transformers without shortcuts: Modifying self-attention for faithful signal propagation." arXiv preprint arXiv:2302.10322 (2023).
[4] Xue, Fuzhao, et al. "A Study on Transformer Configuration and Training Objective." (ICML 2023).
[5] Zhai, Shuangfei, et al. "Stabilizing transformer training by preventing attention entropy collapse." International Conference on Machine Learning. PMLR, 2023.
[6] Zhou, Daquan, et al. "Deepvit: Towards deeper vision transformer." arXiv preprint arXiv:2103.11886 (2021).

### Questions
1. What happens when the same method is applied on decoder only architecture?
2. Does the shallow network with same parameters run faster or slower in terms of wall time? What is the benefit of using deeper and narrow config beyond marginal improvement in perplexity?
3. The results of the pretraining experiments seem off, can you please cite credible sources on numbers with similar config pretrained on Pile-CC?
4. On the parameter counts, do you count the embedding parameter size when reshaping the networks in the experiments, which could potentially be an unfair comparison?

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The author introduces a theory to understand unstable issues in deep transformers and suggests a solution called DeepScaleLM. Their experiments show the effectiveness and superior performance of this method.

### Strengths
1. The authors present a novel theoretical analysis concerning the moments of transformer models.
3. Their development of an effective, theory-driven approach is both sound and provides valuable insights.
3. They conducted comprehensive experimental exploration to support their theory.

### Weaknesses
While I am not well-versed in the experimental section, I would like to point out certain aspects that I found challenging or unclear during my reading.

1. In Figure 2, there seems to be a discrepancy. The author mentions that the backward gradient variance rises hyperbolically with N, but the depicted curve suggests a decline as N grows. This is somewhat perplexing.
2. The representation and caption for Figure 5 lack clarity, making it challenging to decipher the conveyed information.
3. For Table 4 and Figure 7, it would be beneficial if the author could elucidate why the thinnest and deepest transformers utilizing DSLM yield the most optimal results.

### Questions
See weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This manuscript studies signal propagation in transformer networks
to study difficulties in training deep transformer networks.
From the analysis, an initialisation method is proposed to facilitate learning.
The theoretical results are verified empirically on language modelling tasks.

### Strengths
- (significance) Enabling the training of deeper architectures typically leads to improved performance on a variety of tasks.
   At least historically, this kind of result has proven extremely impactful.
 - (clarity) The paper is well written and easy to follow.
   I especially liked how Table&nbsp;1 provides an overview over signal propagation through the different building blocks.
 - (originality) This is the first work that performs such a thorough signal propagation analysis for transformer models.
 - (quality) The derivations in the appenix provide evidence for the theoretical claims.

### Weaknesses
- (originality) A very similar idea for regular networks has been presented in (Arpit et al., 2016).
   The connections with this work should definitely be discussed.
 - (originality) By moving all related work to the appendix and not citing much in the main text it becomes unclear which concepts are new and which already exist.
   I did notice the references to the appendix section, but I would argue that a citation puts more emphasis on the fact that something already exists and is not a contribution of this work.
   Citing one (seminal) paper for each aspect should suffice to counter this issue.
 - (clarity) In Section&nbsp;4.4 an initialisation that renders a model more linear is claimed to be bad for performance.
   However, there is quite a bit of work where this kind of linearity is considered desirable (e.g. Hardt &amp; Moritz, 2017; Zhang et al., 2019)
 - (clarity) I am not sure if it makes sense to report perplexity in the context of masked language modelling.
   I am no expert in NLP, but I thought perplexity is only meaningful for autoregressive language models.
 - (quality) The baseline performances seem to be remarkably weak.
   Table&nbsp;6 in (Delvin et al., 2019) reports perplexities in the range 3-5.
   Results for GPT models on the pile go below 1.
   Also, Yang et al. (2019) report results in the range of 70-76% accuracy for BERT on RACE.
 - (significance) The results in this paper can not be directly applied to vision transformers.
   It could be emphasised a little stronger in the main paper that the analysis focuses on text inputs.
 - (significance) There are too little direct comparisons with competing methods (e.g.&nbsp;Zhang et al., 2019; Noci et al., 2022; Wang et al., 2022a; He et al., 2023).
   Apart from Table&nbsp;5, these alternative methods seem to be ignored completely.
   Also, it would be easier to compare if some of the experiments from these papers would have been adopted.
 - (significance) The experiments all seem to use the same architecture.
   Ideally, an initialisation method works for a variety of architectures.
   This is never properly tested.
 - (quality) Experiments do not have error bars.
   Especially for a random initialisation strategy, error bars would be helpful to assess how consistent the improvements are.
   If error bars would make the experiments prohibitively expensive, it would be nice to include at least one small-scale experiment to provide some insights on the variability of the proposed method.

### Appendix

 - (quality) The propagation of correlation between samples was introduced by (Poole et al., 2016), not by (Schoenholz et al., 2017).
 - (quality) It seems like two fundamental signal propagation papers are missing in the related work.
   The foundations for signal propagation analysis can be found in (Neal, 1996) and a popular reference is (LeCun et al., 1998).
 - (quality) In Section&nbsp;B.2§1, it is claimed that this work considers expectations over inputs, but the expectations are over inputs AND weights (cf.&nbsp;Glorot &amp; Bengio).
 - (clarity) Section&nbsp;B.2§4 states that non-IID inputs are "also" accounted for, but I would argue that this is the only case that is accounted for.
 - (clarity) It is unclear what computation is contained in the embedding component.
   Originally, I suspected this to be only about the token embeddings, but it seems to include other computations as well.
 - (quality) The approximation for the correlation in the embedding layer seems to be missing the Euler constant:
   $$\sum_{i=1}^{|V|} p_i^2 = \frac{\sum_{i=1}^{|V|} 1 / i^2}{\Big(\sum_{i=1}^{|V|} 1 / i\Big)^2} \approx \frac{\zeta(2)}{(\ln |V| + \gamma)^2}.$$
   Without the constant, the approximation is pretty bad.
   Also, it could be made clearer in the derivation where this approximation comes from and that the final simplification step assumes large $L$.
 - (originality) It should be more clearly stated for each derivation where it can be found in literature.
   In its current form, it is hard to distinguish which derivations are really new.
 - (clarity) Instead of using the identity $\pi - \arccos(x) = \frac{\pi}{2} + \arcsin(x)$, it would be better to use $\pi - \arccos(x) = \arccos(-x)$ to stay closer to the formulation from (Cho &nbsp; Saul, 2009; Daniely et al., 2016).
   Similarly, the GELU variance can be further reduced to $$\frac{\sigma^2}{2 \pi} \bigg(\arccos\Bigl(\frac{-\sigma^2}{1 + \sigma^2}\Bigr) + \frac{2 \sigma^2}{(1 + \sigma^2) \sqrt{2 \sigma^2 + 1}} - \frac{\sigma^2}{1 + \sigma^2}\bigg).$$
 - (clarity) The derivation of LayerNorm could use some more explanation.
   Also, the dependencies between samples, sample mean and sample variance should be discussed a bit more.
   Finally, the affine transformation that is typically included after each normalisation layer is not addressed at all.
 - (clarity) It should be more clear what the limitations of the different approximations for the softmax derivation are.
   A quick numerical check verifies that the approximation breaks down quite quickly for larger variances.
   Although this is probably not practically relevant, it would be nice to provide some insights when the analysis breaks down.
 - I noticed that the derivation for scaled dot-product attention starts with some rough shortcuts.
   Also, a completely different approach is used to handle the softmax.
   This seems suspicious.
   I do not have time to check the math any further, but the numerical results should provide enough proof that these results, if not correct, are at least useful.

### Minor Comments

 - Citations could be polished a bit more (e.g. "Deep Information Propagation" has been published at ICLR, "ReZero is all you need" has been published at ICML, ...).
   I also noticed not some references have a link while others do not have an URL.
 - possible typos in Section&nbsp;1§5: "issues with very deep transformerS", 
   in Section&nbsp;1§6: "ensures the moments of outputs and gradients (to) remain fully conserved"

### References

 - Neal, R. M. (1996). 
   Bayesian Learning for Neural Networks.
 - LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). 
   Efficient BackProp.
   Neural Networks: Tricks of the Trade (1st ed., pp. 9–50).
 - Cho, Y. &amp; Saul, L. (2009). 
   Kernel methods for deep learning. 
   Advances in neural information processing systems, 22.
 - Arpit, D., Zhou, Y., Kota, B., & Govindaraju, V. (2016). 
   Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks. 
   Proceedings of The 33rd International Conference on Machine Learning, 48.
 - Daniely, A., Frostig, R., &amp; Singer, Y. (2016). 
   Toward Deeper Understanding of Neural Networks: The Power of Initialization and a Dual View on Expressivity. 
   Advances in Neural Information Processing Systems, 29.
  - Poole, B., Lahiri, S., Raghu, M., Sohl-Dickstein, J., &amp; Ganguli, S. (2016). 
   Exponential expressivity in deep neural networks through transient chaos. 
   Advances in Neural Information Processing Systems, 29. 
 - Hardt, M., &amp; Ma, T. (2017). 
   Identity Matters in Deep Learning. 
   International Conference on Learning Representations, 5.
 - Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., &amp; Le, Q. V. (2019). 
   Xlnet: Generalized autoregressive pretraining for language understanding. 
   Advances in neural information processing systems, 32.

### Questions
1. Please, add citations to make clear which parts of this work are not part of the contribution.
 2. Please, connect the proposed initialisation to the work from (Arpit et al., 2016).
 3. How is perplexity computed for masked language modelling and why does it make sense as a metric?
 4. Do you have any references or other explanation for the seamingly poor baseline results?
 5. How easy or difficult would it be to apply the analysis to vision transformers or transformers that do not work with language/embeddings?
 6. How does deepScaleLM compare to other (non-standard) initialisation strategies and deep transformers?
 7. How does deepScaleLM perform on different architectures (e.g. GPT, long-context transformers, ...)?
 8. Would it be feasible to include error bars for some of the results?
 9. How can the numerical results be so good if the approximation for the Embedding layer correlations is significantly off? 
    Is correlation not that important after all?
 10. Why are the insights from Section&nbsp;C.7 not reused for the analysis in Section&nbsp;C.8?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper derives a closed form signal propagation formula, i.e., mean, variance, and input-output correlation, for each component in Transformer, including both forward and backward passes. It helps explain gradient vanishing or explosion, rank collapse, and training instability of Transformer. Leveraging these formulas, the paper then proposes a new initialization scheme, DeepScaleLM, that stabilizes the signal propagation across training. Experimental results show that with the proposed initialization scheme, Transformer with 100s of layers can be trained and obtains better results than a shallow counterpart.

### Strengths
The paper studies a fundamental problem of scaling neural nets in depth. The study follows the direction of signal propagation, which is promising. The derived mean and variance formulas are shown to be matching the empirical simulation. Experimental results are also encouraging, showing depth is indeed helpful for Transformer as well.

### Weaknesses
1. The main issue probably comes from the model selection in the experiment section. Although BERT experiments are good, an additional evaluation using an encoder-decoder Transformer or decoder-only Transformer would provide more interesting spikes and make the justification more compelling.

2. The concrete picture of the adjusted model architecture is missing to the reader. The proposed DeepScaleLM scheme scales residual connections, and seems like it adds extra dropout layers. In which component are these adjustments applied? Part of a layer or every component in a layer? Partial layers or every layer?

### Questions
1. Plotting log-scale in Figure 3 does not help the reader understand deeply about the gap between empirical simulation and theoretical prediction. Providing statistics such as min, max, mean, variance of the gap would be better.

2. In Section 2, the paper assumes normal distribution of inputs, weights, and gradients while deriving the formulas in Table 1. To what extent this assumption holds since in Section 1 paragraph 2 the paper mentioned that some of the assumptions in prior works broke down on real world data?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
