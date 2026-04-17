# RaBEL: Scale-Aware Radial-Basis Embeddings for Tabular Foundation Models

- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Recent tabular foundation models routinely match or surpass strong tree ensembles and specialized deep architectures, yet their numeric embeddings remain a bottleneck. We diagnose a low-rank collapse induced by the prevalent linear+ID scheme and introduce RaBEL, a compact Radial Basis Embedding Layer that front-loads nonlinearity via localized RBF features. RaBEL increases shallow-layer effective rank and improves conditioning without deeper stacks; it is complementary to periodic mappings. We further identify a permutation-order pathology in bidirectional attention (feature$\rightarrow$sample) and propose a reordered stack: sample-attention $\rightarrow$ FFN $\rightarrow$ feature-attention, ensuring column-level context precedes feature mixing and that all attention computations influence the readout. Combining both ideas yields MiniX, a 2M-parameter model that surpasses 7M-parameter TabPFN-v2 and 27M-parameter TabICL baselines on popular benchmarks while reducing training and inference cost. Our results highlight principled nonlinear embeddings and attention-order redesign as key enablers of accuracy and efficiency gains in tabular foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new embedding scheme for tabular foundation models, such as TabPFN, that addresses the low-rank problem encountered with linear embeddings. Moreover, the authors change the order of blocks in transformer layers, which improves the model. All these improvements result in MiniX.

### Strengths
Overall, the idea of improving tabular foundation models and making them smaller yet effective is a great outcome of the research. 

1. I think the low-rank problem is crucial, and this finding is the most valuable result of the paper.
2. Figure 1 is also valuable.
3. A small tabular foundation model with comparable performance is a strong result.
4. The ablation study in Table 4 is important for the paper.

### Weaknesses
See the questions for details.

Weaknesses summary:
1. Section 4.2 is poorly written.
2. The toy experiments are currently misleading.
3. There is a lack of details about the experimental setup.
4. I think the title can be improved.
5. There is a lack of important ablations on embedding.

### Questions
I will divide my questions into two parts: technical and stylistic/paper-related. You can find a summary at the end.

**Technical:**

1. I believe TabPFNv2 layer has the order `feats_attn -> sample_attn -> ffn`. Do you retain the final FFN layer, resulting in `sample_attn -> ffn -> feats_attn -> ffn`? If so, can you confirm that the improvement comes from the reordering rather than from adding an additional FFN? This should be clarified in the main text.

2. L231-238 contain the statement "RaBEL increases $r_{eff}$ in shallow layers by providing diverse..." without any experimental evidence. Either include supporting artifacts (tables/figures) or remove this paragraph, please.

3. What is a "bounded set of exponent bins"? What are the specific values for $b_{min}$ and $b_{max}$? From my understanding, this represents a method to partition inputs into bins like $[2^{b}, 2^{b + 1})$. I believe this explanation could be made clearer.

4. **Toy experiments:** Periodic embedding actually can approximate the functions shown in Figure 2b, though the results heavily depend on the initialization of $c_i$ (which I view as a significant limitation of periodic embedding and would emphasize). The simplest demonstration involves initializing $c_i$ with $N(0, 20^2)$ for equation (5) and increasing $k = 64$. I can provide code if needed. This section must include experimental details—currently, there's no information about the setup whatsoever (learning rate, number of epochs, num train points, etc.).

5. RaBEL is compared against periodic embeddings in toy examples, but there's no comparison including MLP-RaBEL or MiniX-PLR, making this toy example comparison appear redundant. While I understand that training MiniX-PLR is costly, I believe `MLP-RaBEL vs MLP-PLR` would provide more valuable insights than the current toy examples section. Additionally, could periodic embedding potentially resolve the rank problem? Overall, in this and previous question I mean that the question "why RaBEL is better than other embeddings?" is not answered clearly.

6. **Main experiments:** As I understand, you completely followed LimiX. However, there's currently no information about the experimental setup. What normalization was applied across all datasets? How was HPO performed? How were ranks calculated? How do you establish statistical significance for the reported results? Why do the TabArena results diverge from the official benchmark (e.g., RealMLP performs significantly worse in your setup)? Is there any reason TabM was not included?

7. Are all RaBEL embedding components necessary? Why not simply make $c$ and $\sigma$ trainable and omit the exponent-gating and shared-gates? Ablations using MLP could help address this question.

8. (minor) The paper would benefit significantly from experiments on large datasets. Due to the lower hidden dimension, you can evaluate MiniX on substantial datasets.

9. Could you provide intuition on why flattening the sample+feature dimension is preferable to flattening the feature+D dimension?


**Paper-related:**

1. Section 4.2 requires substantial revision: it contains duplicate text, redundant discussion of $r_{eff}$, lacks hyperparameter values ($M, b_{min}, b_{max}, \tau, h_s$), and presents unclear motivation. Motivations (ii) and (iii) on lines 293-294 remain unexplained. The third motivation should be validated experimentally. I recommend the authors invest time in rewriting this section more clearly. Consider starting with the initial RBF kernel, then introducing enhancements sequentially (without text duplication) and experimentally prove their significance. Alternatively, retain only the final "full" version of the embedding explanation.

2. **Minor:** Please define "energy" on line 163. I assume full energy represents the sum of squared singular values.

3. **Title suggest:** I find the title and abstract potentially misleading (w.r.t. paper conent) and potentially harmful to the paper. The title suggests the embedding as the primary contribution, but I believe the paper mostly about MiniX and making tabular foundation models better by fixing low-rank problem. I recommend revising the title to something like (this is just a rough suggestion) "MiniX: Fixing the low-rank problem in tabular foundation models with RBF-based embedding" to better reflect the paper's results and impact on the field.

4. **Note for multiple sections:** Please include experimental setup and implementation details throughout.

Overall, I believe these findings hold significant value for the community and the field, but the current paper quality is low. My current score is 2, though I'm willing to increase it toward acceptance if most technical questions are addressed during the rebuttal. Questions 4-6 are particularly crucial for raising my score to 6.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The submission proposes two architectural design choices for tabular foundational models (TFMs) that allow to achieve the lightweight model which is comparable to the state-of-the-art in terms of predictive accuracy.

### Strengths
The paper investigates the design choices for tabular foundation models, which is an active research direction. In my subjective opinion, the field is currently at the point where we should firstly focus on improving predictive accuracy rather than efficiency, since for modern hardware the existing TFMs are quite lightweight. But the paper findings indeed can be interesting to the subset of community working on the TFMs architectures.

### Weaknesses
(1) I am not fully satisfied with the positioning of the paper. Only one of two equally important contributions is presented in the title, why? For instance, Table 4 shows that on TabZilla RBA is more important than Rabel.

(2) The submission makes several claims that are not supported:

  (2.1) line 43 - "low rank of activations impedes gradient flow" - I did not find the proof in the paper.

   (2.2) line 73 - "strengthening gradient signals throughout the stack and improving parameter utilization" - I did not find the proof in the paper.

Without empirical confirmation each of these point is only a guess, intuition.

(3) The recent TabM model is not included in the comparison.

(4) From the description of new embeddings, it is not clear what are the main hyperparameters of the scheme and the sensitivity of Minix to them. I assume that the important hyperparameter is M (the number of centers) but I did not find any instructions how to choose its value and what value was used in the experiments.

(5) The authors do not report the advantage of Minix in terms of runtime efficiency (wall-clock time). The advantage in terms of memory in my opinion is less interesting since tabular foundational models are quite compact compared to models from other domains (CV, NLP).

(6) From the current set of experiments I do not buy the claim "The hidden states output by transformer layers tend to be low-rank, especially early in the network. This could severely decrease the expressivity of the network, leading to potential performance degradation." I appreciate the results from Table 1 but it demonstrates only the redundancy of TabPFNv2, which could be exploited to achieve higher efficiency. From the submission, I do not see any evidence that low-rank activations can be harmful for performance in terms of predictive accuracy.

### Questions
(1) Please, address my concerns in the Weaknesses section.
(2) Why the authors did not investigate the advantage of their embeddings beyond the context of tabular foundational models, for instance with the simplest MLP? Would they work better than MLP-PLR? In my opinion, such experiments would highlight the advantage of RaBEL better. Or is there some special chemistry between the RaBEL and TabPFN-like models? I appreciate the experiments from section 5.1 but it would be more interesting to extend them to real problems.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces RaBEL, a radial-basis embedding layer to address the low-rank bottleneck in early layers of tabular foundation models, and proposes a reordered bidirectional attention stack (sample-attention → FFN → feature-attention); it further presents MiniX, a 2M-parameter TFM.

### Strengths
1. The paper is relatively clear and easy to understand.
2. Analyzing the model from the rank perspective is interesting.
3. The proposed method achieves comparable performance on datasets (≤50,000 training samples, ≤10 target classes) as defined in the paper.

### Weaknesses
1. Since MiniX and LimiX share the same training setup, including their experimental protocols and optimization configurations, incorporating LimiX into comparative analyses—such as the experiment on "rank evolution across layers"—would strengthen the validity of conclusions. This addition would help clarify whether observed differences in performance or rank dynamics stem from architectural modifications (e.g., RaBEL vs. LimiX’s embedding) rather than training discrepancies, enhancing the rigor of the comparative framework.  

2. The improvement of the model performance mainly comes from both RaBEL and RBA, yet the ablation experiment shows that the performance gain brought by RaBEL is not significantly higher than that of RBA. However, the paper focuses primarily on RaBEL. This mismatch between the core focus and experimental results may require a revision of the paper’s structure—either to adjust the emphasis on RaBEL’s differentiated value or to clarify why RaBEL is prioritized as the core contribution.  

3. Regarding RBA: Compared with the module stacking order of TabPFN-v2, RBA not only rearranges the attention sequence (from "feature→sample" to "sample→feature") but also introduces an additional FFN. However, the paper lacks controlled experiments to disentangle the contributions of these two modifications. It remains unclear whether the performance improvement stems from the "order optimization" (aggregating sample-level statistics first before feature interaction) or the "nonlinear transformation of FFN" (compressing sample-wise information), which weakens the rigor of RBA’s design logic.  

4. The number of model parameters is not as critical as implied. Among the compared baselines, TabICL—with the largest parameter count (27M)—exhibits the fastest inference speed, suggesting that inference speed is a more practical metric for tabular models.

5. The claim in Lines 41–44 (that low rank "hinders gradient flow" and "blunts the benefits of scaling deeper or wider backbones") lacks sufficient experimental support. While the paper indirectly supports the low-rank issue via rank comparisons between MiniX and TabPFN-v2 (Fig. 1), it does not provide direct evidence—such as verifying whether MiniX gains more relative performance when scaled up (e.g., deeper layers, more attention heads) compared to TabPFN-v2. This makes the claim more speculative than empirically grounded.  

6. There is ambiguity about whether "RaBEL" and "ReBEL" (mentioned later in experiments like Fig. 2 and Tab. 4) refer to the same component. The paper does not clarify if "ReBEL" is a typo or a modified version of RaBEL. This inconsistency in terminology confuses readers’ understanding of the module design and requires standardization with explicit definitions.  

7. The rationale for limiting datasets to ≤50,000 samples is unexplained. This criterion clearly favors small-sample and excludes common practical scenarios for tabular data (e.g., large-sample datasets with millions of instances or multi-class tasks), making it impossible to verify the model’s generalization to broader use cases. Additionally, the paper does not compare the distribution of filtered vs. unfiltered datasets or explain if this criterion unfairly benefits MiniX.  

8. The transparency and comparability of experimental settings are insufficient:  
   - Details about baselines are missing: It is unclear whether other models (e.g., TabR, ModernNCA, RealMLP) underwent hyperparameter optimization (HPO) or used ensemble strategies. This makes it impossible to explain why some models perform worse than simple MLPs (poor performance may stem from inadequate tuning rather than inherent weaknesses).  
   - Data integrity is lacking: The TabM method (available in TabArena) is excluded, and the paper does not specify whether results for unmentioned methods are reused from TabArena’s official benchmarks or retrained by the authors. Inconsistent experimental protocols (e.g., training epochs, validation strategies) undermine result comparability.  
   - Result traceability is unclear: It is not stated whether the overall results follow TabArena’s evaluation pipeline; any discrepancies (e.g., data splitting, metric calculation details) would distort cross-model comparisons.  

9. Comparisons between RaBEL and other numerical embedding methods (e.g., -PLE, -PLR) are missing, leaving RaBEL’s differentiated advantages unproven. Additionally, the paper does not test whether RaBEL can yield similar performance improvements when integrated into other tabular deep learning architectures. Of course, this is merely a suggestion and not considered a major issue.

### Questions
Refer to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper provides an analysis of tabular foundation models through the perspective of low-rank collapse in their respresentation. This motivates a new, more efficient, but performant architecture with two new components - Radial Bassi Embedding Layer and revamped base PFN transformer layer structure. Authors provide intuitions, toy experiments and introspections into tabular foundation models as evidence plus they demonstrate the resulting model's (MiniX) performance empirically on a set of benchmarks (TabArena, Tabzilla and the OpenML CTR23 regression dataset suite) where MiniX lands second after only the current SoTA (as far as I personally know) tabular foundation model.

### Strengths
I find the paper fresh and useful for the literature on tabular foundation models. In my view the current state of affairs in this field is heavily in need of analysis and understanding. The paper contributes to our understanding a fair amount of new knowledge in my view.

My favorite takeaway from reading the paper  is that tabular foundation models are "appropriately" scaled in terms of parameter counts, or even a bit redundant at present. Contrary to the meta in other domains (e.g. LLMs and computer vision models getting into billions of parameters), tabular foundation models are different, and one can achieve close-to-SoTA results with a very small model. This overall feeling sways me towards accept, I belive this paper provides valuable insight to the community.

Result in Table 1 and Figure 1 are intriguing. We can see that current models hidden-state is heavily underutilized - this is an interesting and new piece of knowledge about ICL-based tabular foundation models.

A new feature embedding (RaBEL) is also interesting, It seems that it may be less sensitive to hyperparameters than prior work alternatives. But this, I belive, requires more testing and reporting. Another change (RBA) also seems notable, judging by the ablation in Table 4.

### Weaknesses
Despite me overall rating the paper positively and leaning towards accept. I think it has some downsides and problems that if addressed would make it better and make me more confident in my assessment.

1. The results reported in Table 2 and especially Table 3 are a bit too minimalistic. The paper provides only the aggregate scores and focuses mostly on the average AUC/R^2. When you take a closer look at the average rank, the MiniX model is near or below XGBoost on CTR23 for example. Providing as much and as granular as possible results would make the paper clearer and stronger in my view. I recommend to report individual dataset results and some form of confidence intervals or any other appropriate mean for a fairer comparison.
2. Regarding the analysis of the low-rank collapse in the representation. Why do you calculate the rank the way you do by flattening samples and features into one dimension. I think this should be explained more thoroughly. I can see an alternative strategy for computing the inner rank e.g. by flattening each individual object, instead of feature (e.g. $N \times M \times D$ into $N \times (M \cdot D)$. Also what about more granular rank statistics? E.g. Per token or per layer - the target corresponding token could be different from the other ones for example. Could you provide more insight into your thinking and maybe provide alternative rank stats?
3. The toy experiments part that introduced the new embedding scheme seems a bit less strong empirically. The only two examples are toy functions. I would've liked to see some more evidence for the proposed scheme (embeddings are also useful in non-foundation models, as can be seen in the original paper ["On Embeddings for Numerical Features in Tabular Deep Learning"](https://arxiv.org/abs/2203.05556) or in the recent RealMLP or TabM models. Demonstrating the efficacy of the proposed scheme in general would strengthen the claim that is made on the strength of RaBEL embeddings.
4. There is a statement made that same performance could be attained with a smaller model. As I outlined in strength - I like this finding in general. But the paper does not dive as deep into this point as it perfectly would in my view. First, the additional architectural improvements are at odds with this a bit. I guess what could imporve the paper is adding the baseline 2M model into the overall comparison (with all the rigor from point 1 I've mentioned). I believe it could be on par with the Mitra model, which is notable for proving the point of model redundancy and raises question on why there is such a difference? data? better hyperparameters? (exactly the missing analysis and ablation-style studies in the current tabular PFN world) Second, is the question whether scaling up MiniX would actually make the best model? This may be harder to address experimentally during the rebuttal period, I believe this should be at least discussed in conclusion.

### Questions
see weaknesses. Each point is an addressable question

### Soundness
2

### Presentation
3

### Contribution
3
