# Does Generative Retrieval Break through the Limitations of Dense Retrieval?

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 2

## Abstract
Generative retrieval (GR) has emerged as a new paradigm in neural information retrieval, offering an alternative to dense retrieval (DR) by directly generating identifiers of relevant documents. In this paper, we theoretically and empirically investigate how GR fundamentally diverges from DR in both learning objectives and representational capacity. GR performs globally normalized maximum-likelihood optimization and encodes corpus and relevance information directly in the model parameters, whereas DR adopts locally normalized objectives and represents the corpus with external embeddings before computing similarity via a bilinear interaction. Our analysis suggests that, under scaling, GR can overcome the inherent limitations of DR, yielding two major benefits. First, with larger corpora, GR avoids the sharp performance degradation caused by the optimization drift induced by DR’s local normalization. Second, with larger models, GR’s representational capacity scales with parameter size, unconstrained by the global low-rank structure that limits DR. We validate these theoretical insights through controlled experiments on the Natural Questions and MS MARCO datasets, across varying negative sampling strategies, embedding dimensions, and model scales. But despite its theoretical advantages, GR does not universally outperform DR in practice. We outline directions to bridge the gap between GR's theoretical potential and practical performance, providing guidance for future research in scalable and robust generative retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper revisits the core limitations of dense retrieval (DR) and investigates whether generative retrieval (GR) can overcome them. The authors identify two fundamental bottlenecks in DR: first, a local normalization bias that causes score inconsistency when the corpus grows, and second, Low-rank representation bottleneck, which constrains expressivity regardless of model scale. They provide both theoretical analysis and empirical validation of these issues, showing that GR, by globally normalizing over docids and decoupling embedding dimensionality from retrieval capacity, can, in principle, bypass these limitations. Theoretical analysis and experiments on MS MARCO and Natural Questions show that GR scales better with corpus size and model capacity, providing a explanation for its potential advantages over DR.

### Strengths
- The paper is well organized and highly readable. The logical flow from theoretical formulation to empirical validation makes a complex topic accessible
- The authors correctly pinpoint two essential weaknesses of dense retrieval, the local normalization bias and the low-rank representation constraint. These are described with both theoretical precision and intuitive clarity, providing a solid foundation for the rest of the analysis.
- The paper effectively argues that generative retrieval (GR) has the potential to overcome these issues through global normalization and by representing a full-rank retrieval distribution over discrete document identifiers. The theoretical reasoning is persuasive and well-motivated.

### Weaknesses
- Although the paper claims that GR can overcome the intrinsic limitations of DR, the experimental results, particularly on MS MARCO, show that GR still performs significantly worse than DR. This gap is acknowledged but never analyzed. The paper does not explain why GR underperforms despite its theoretically superior formulation. This contradiction between claim and observation weakens the central argument.
- Contrary to the paper’s theoretical prediction, GR’s performance still drops as corpus size increases undeniably (Fig 4) . This is inconsistent with the claim that GR resolves corpus-scaling instability. The authors do not provide a explanation for this behavior, whether it arises from.
- The discussion section outlines several potential solutions and practical challenges (e.g., docid compression, hierarchical decoding, better calibration). However, none are experimentally validated. Too much of the work’s contribution is deferred to future research, leaving the current paper largely diagnostic rather than demonstrative. In practice, the results suggest that GR still faces more severe limitations than DR, without strong evidence that these can be overcome.
- The paper lists “text docid” as one of the main experimental setups in the method, but the corresponding results are absent from the main text.

### Questions
- GR’s performance on MS MARCO remains far below DR despite addressing the low-rank and normalization issues. What are the authors’ hypotheses for this discrepancy?
- The discussion section proposes several potential directions (e.g., hierarchical decoding, global calibration, docid compression). Could the authors elaborate on which of these they believe are most practical.

### Soundness
3

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
3

### Summary
This paper presents a systematic theoretical and empirical comparison between Generative Retrieval (GR) and Dense Retrieval (DR). It provides two key theoretical analyses: (1) GR optimizes a globally normalized likelihood, while DR uses locally normalized contrastive loss, leading to calibration drift; and (2) DR’s bilinear interaction imposes a low-rank bottleneck, while GR can approximate higher-rank query-document mappings. These claims are validated through controlled experiments on NQ and MS MARCO, showing that GR degrades more gracefully under corpus scaling and benefits more from model scaling.

### Strengths
- Theoretical formulation is clear and mathematically sound.
- Empirical studies are well aligned with the theoretical predictions. The design is systematic and convincing.
- The paper highlights an important direction for retrieval research under the LLM paradigm, connecting classical IR insights with modern generative modeling.

### Weaknesses
- The theoretical results largely restate known intuitions (e.g., local vs global normalization, low-rank bottlenecks). The contribution is primarily in formalization
- The empirical validation uses relatively small single model (0.6B) and limited corpora such as NQ, leaving open whether conclusions hold at other datasets and models
- The paper assumes ideal docid design and in-distribution queries, robustness under noisy or evolving corpora is not evaluated.

### Questions
- Could the authors provide quantitative comparisons on latency and computational cost between GR and DR under identical retrieval budgets?
- Theoretical analysis assumes fixed docid mapping and in-distribution queries. How would GR behave under corpus updates or unseen documents?

Please refer to the weakness part.

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
4

### Summary
This paper presents a thoughtful theoretical and empirical investigation into the fundamental differences between Generative Retrieval (GR) and Dense Retrieval (DR). It argues that GR, unlike DR, optimizes a globally normalized likelihood objective and encodes corpus and relevance information directly in model parameters, rather than relying on external embeddings and locally normalized training. Theoretically, the authors claim that GR can overcome two inherent bottlenecks of DR: (1) the optimization drift induced by local normalization when scaling to large corpora, and (2) the representation bottleneck caused by DR’s low-rank embedding space. Empirical validation on Natural Questions and MS MARCO supports these arguments—showing that GR benefits more from model and corpus scaling, while DR suffers from calibration and dimensionality constraints. However, the paper also acknowledges that GR’s theoretical advantages do not yet translate to universal empirical superiority. Factors such as docid design, data construction, and decoding strategies significantly affect practical performance.

### Strengths
1. The paper offers a theoretical analysis that supports the observed limitation of dense retrieval, local optimization and low rank representation. 

2. The paper conducts comprehensive and carefully controlled experiments across multiple datasets, model scales, and training configurations to substantiate its theoretical claims.

### Weaknesses
1. While the paper provides solid theoretical support highlighting two limitations of dense retrieval (DR)—local normalization and the low-rank constraint—these issues are already somewhat familiar to the retrieval community. The manuscript would be stronger if it more clearly positions its contributions relative to prior work that has identified or partially characterized these limitations. 

2. There is an internal contradiction in the empirical evidence regarding local normalization and negative sampling. Figure 1 suggests retrieval performance improves as the number K of negatives increases, implying negative sampling partially alleviates the calibration problem; by contrast, Table 1 is interpreted to show that mitigating local normalization via negative sampling is “non-trivial” (i.e., limited benefit). This creates a contradiction in the paper’s message. 

3. The current scaling experiments do not convincingly explore the regime where GR’s claimed advantages should emerge. The model sizes tested are not large enough and the breadth of size is limited. Indeed, Figure 3 can be read to suggest DR is more parameter-efficient than GR—DR achieves higher performance with fewer parameters. 


4. Beyond retrieval quality, practical deployment requires consideration of inference latency, memory footprint, and throughput—areas in which dense retrieval typically excels because of index-based nearest-neighbor search. The paper should include an explicit discussion (and ideally measurements) of inference latency, GPU/CPU cost, and storage requirements for GR vs DR, or at minimum provide reasoned analysis of these trade-offs and how they affect the practicality of GR.

5. The manuscript rightly focuses on two important limitations of DR, but retrieval performance is influenced by many other factors. The paper would be more useful if it discussed how they might interact with the two limitations studied.

6. The proof of Theorem 3.4 in Appendix C appears informal in places and would benefit from greater rigor.

### Questions
Overall, I find this paper to provide interesting and thoughtful theoretical support for the observed differences between generative retrieval (GR) and dense retrieval (DR). However, the core observations themselves are rather intuitive—the issues of local normalization and low-rank embedding limitations in DR have been broadly recognized in the community—and the analysis, while rigorous, primarily formalizes rather than fundamentally extends these intuitions. Please see above weaknesses for more detailed concerns.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper theoretically and empirically investigate how GR fundamentally diverges from DR in both learning objectives and representational capacity.
The authors argue:
1. GR uses globally normalized maximum-likelihood training, unlike DR’s locally-normalized contrastive training.
2. GR scales better with larger corpora and models, avoiding optimization drift and low-rank bottlenecks present in DR.
They experiment over NQ and MSMARCO to validate these theoretical findings.

### Strengths
- The paper provides a well-structured and intuitive comparison between dense retrieval and generative retrieval, highlighting their distinct learning objectives and representational properties.
- The authors conduct comprehensive experiments to verify their theoretical claims.
- The authors propose several potential solutions to improve the practical performance of generative retrieval.

### Weaknesses
Many of the findings seem to have already been explored. Could you clarify the comparison with prior work?:
- The discussion on learning objectives in dense retrieval appears related to [1], while the analysis of representational capacity seems closely connected to [2]. In addition, the comparison of GR vs. DR with respect to embedding size resembles findings in [3].
- For model scaling, why did you vary adapter size rather than model size? Also, when it's on model size, the scaling effects for DR and GR have been explored in [4] and [5], respectively.
- The zero-shot GT setting in Section 4.4 appears related to [6]. It would be helpful to clarify how your setup differs.

[1] Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." EMNLP (1). 2020.

[2] Luan, Yi, et al. "Sparse, dense, and attentional representations for text retrieval." Transactions of the Association for Computational Linguistics 9 (2021): 329-345.

[3] Lee, Hyunji, et al. "Generative multi-hop retrieval." arXiv preprint arXiv:2204.13596 (2022).

[4] Ni, Jianmo, et al. "Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models." arXiv preprint arXiv:2108.08877 (2021).

[5] Tay, Yi, et al. "Transformer memory as a differentiable search index." Advances in Neural Information Processing Systems 35 (2022): 21831-21843.

[6] Yu, Wenhao, et al. "Generate rather than retrieve: Large language models are strong context generators." arXiv preprint arXiv:2209.10063 (2022).

### Questions
- Do you expect the experimental results to generalize to larger model scales?
- While you argue that GR benefits from greater representational capacity, prior work has discussed the limited generalization of generative retrieval [1], which seems to be due to limited capacity. Could you share your thoughts on this?
- Prior work generally shows that hard negatives help training [2,3], but Table 1 reports performance degradation as the hard-negative ratio increases. Could you provide intuition for this?

[1] Mehta, Sanket Vaibhav, et al. "Dsi++: Updating transformer memory with new documents." arXiv preprint arXiv:2212.09744 (2022).

[2] Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." EMNLP (1). 2020.

[3] Izacard, Gautier, et al. "Unsupervised dense information retrieval with contrastive learning." arXiv preprint arXiv:2112.09118 (2021).

### Soundness
3

### Presentation
3

### Contribution
2
