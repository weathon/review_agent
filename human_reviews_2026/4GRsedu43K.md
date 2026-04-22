# Guided Query Refinement: Multimodal Hybrid Retrieval with Test-Time Optimization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Multimodal encoders have pushed the boundaries of visual document retrieval, matching textual tokens directly to image patches and achieving state-of-the-art performance on challenging benchmarks. Recent models relying on this paradigm have massively scaled the dimensionality of their query and document representations, presenting obstacles to deployment and scalability in real-world pipelines.
Furthermore, purely vision-centric approaches may be constrained by the inherent modality gap still exhibited by modern vision-language models. In this work, we connect these challenges to the paradigm of hybrid retrieval, investigating whether a lightweight dense text retriever can enhance a stronger vision-centric model. Existing hybrid methods, which rely on coarse-grained fusion of ranks or scores,
fail to exploit the rich interactions within each model’s representation space. To address this, we introduce Guided Query Refinement (GQR), a novel test-time optimization method that refines a primary retriever’s query embedding using guidance from a complementary retriever’s scores. Through extensive experiments on visual document retrieval benchmarks, we demonstrate that GQR allows ColPali-based models to match the performance of models with significantly larger representations, while being up to 14x faster and requiring 54x less memory. Our findings show that GQR effectively pushes the Pareto frontier for performance and efficiency in multimodal retrieval. We release our code at https://github.com/IBM/test-time-hybrid-retrieval

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Guided Query Refinement is a test-time retrieval ensembling method that iteratively updates the primary retriever’s query representation using gradient descent to minimize the KL divergence of the (query, top k document) probability distribution to the secondary retriever's distribution. On the ViDoRe 2 benchmark,  it outperforms other ensembling methods, as well as the primary retriever when used on its own.

### Strengths
This paper provides an interesting and novel framework to refine query representations at test time through gradient descent on the representation itself, in order to leverage information from a complementary retriever. The method used enables optimizing multi and single vector models, and yields strong results on Vidore V2.

The paper is mostly well written and relatively easy to understand.

### Weaknesses
**Objective function simplification**: The objective function that is minimized here admits a "trivial" minimum when p1(t) = p2. While this may be hard to reach in practice, the role of the "extra-step" of averaging p1(t) and p2 is unclear to me: since p_avg(t) depends on p1(t), it just seems that by minimizing KL(p_avg(t) || p1(t)), we are in practice (mostly) minimimizing KL(p2||p1(t)), and thus nudging p1(t) to match the p2 distribution. The ablations that are run l421 mostly confirm this is the case.  Under this light, it is not clear what the additional complexity of the "averaging step" brings.

**Objective function rationale**: In all cases, it seems that the effect of GQR is to modify the query representation(s) to more closely match the probability distribution yielded by a weaker secondary retriever. I personally find hard to gain intuitions on why this would help in practice - and how this differs from some sort of tuned probability distribution averaging method.

I would find interesting to understand how scores evolve through training steps: intuitively, at t=0, we match the primary model nDCG, then (according to your results), the ranking is slightly improved, and I would imagine that eventually, as more steps are made and p1(t) too closely matches p2, performance drops and starts matching those of the secondary models ? Understanding these dynamics could be key here in explaining why this method works. It would also be interesting to have some plots of the score distributions before/after GQR to better understand what is going on. Finally, a measure of how much p1(0) and p2 impact the final distribution (ratio of KL divs for instance) could be given over the different data subsets on which hyperparam optimization is run to gain a sense of how data dependant GQR is. 

**Hyperparameter sensitivity**:  From my understanding, the dynamic described above would indicate that there is a bounded ranges of optimization step that can be made before collapse, and thus that GQR is sensitive to hyperparameter choice. It also seems you independently optimize parameters for each data subset in Vidore 2. Would it be possible to show results where hyperparameters are not optimized on the same data distribution (ie. optimize on Vidore train set, test on the full Vidore v2 with the same parameters across each split ?). 

*I believe it is key to add the final hyperparameter values (for the method as well as the baselines) used in the appendix. Not having them makes assessing the work much more difficult.*

**Missing files**: The anonymous4science link works (and is appreciated) but most files are missing. I can only see the Readme and the latency benchmarking codefiles, not the rest. I also don't see the chosen hyperparams.

**Clarity**: In this paper, the query embeddings that are optimized are multi-vector embeddings (in most cases). While the query refinement technique applies directly here, this could be more clearly explicited - I feel this "form agnosticity" is one of the strenghts of this method.

**Overall**: I am aware I suggest many things here, the overall point is that the rationale behind "why" GQR works remains unclear, and understanding the method's sensitivity to hyperparameters would be key to inform practical uses of this method beyond benchmark maximisation.

### Questions
1.  (See *Weakness Objective function simplification*) Is there any reason to justify the additional complexity of the "averaging step" in the objective function that seems redundant ? 

2. You state that hyperparameters are chosen on a dev split of 10% of the data. To my knowledge, no dev split exists for Vidore V2. Do you thus split the test split and report results on 90% of the remaining docs ? If this is the case, is there any overlap between docs in the "test" and your current "dev" split here ? 

3. How do you explain GQR does not improve results on ViDoRe 1 ?

4. Under the "nudging towards p2" hypothesis, GQR would be ~roughly~ equivalent to a "tuned" normalized score averaging technique in which the tuning is done through gradient descent. In your results, the best baselines in practice are also "tuned" normalized score averaging, but the alpha parameter that can be optimized is discrete (steps of 0.1). Could the performance gap be closed if alpha becomes a learnable parameter here (or at least is optimized at a more granular level) ? Could you at least report alpha per subset ?

**Suggestions**:
Seems figures are big enough to integrate text from the legends directly in the plot. This may make it a bit easier to understand the plots in one glance. This is very minor of course.
Add x-axis label in figure 3.

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
3

### Summary
This paper introduces Guided Query Refinement (GQR), a test-time optimization technique designed to improve visual document retrieval, by iteratively refining the query embedding of a primary retriever using guidance from a complementary retriever. Instead of aggregating outputs at the ranking or score level, GQR operates at the representation level. The method is evaluated on ViDoRe 1 and ViDoRe 2.

### Strengths
1. The core idea and methodology of the paper are concise, and the inclusion of algorithm pseudocode makes it easy to understand and follow.
2. The authors evaluated nine retrieval pairs on standard retrieval visual document benchmarks, demonstrating the efficiency of the proposed method in most cases.
3. The code has been made publicly available, but there appear to be some minor shortcomings.

### Weaknesses
1. The paper is titled Multimodal Hybrid Retrieval; however, the authors focus solely on a single visual document retrieval task. In fact, multimodal hybrid retrieval covers a much broader scope — it may also include tasks such as video-text retrieval and composed image retrieval. Therefore, the paper does not fully demonstrate that the proposed method is applicable to the broader multimodal hybrid retrieval setting.

2. The algorithm relies on tuning hyperparameters such as $\alpha$ and $T$, yet the authors do not present any analysis of the model’s sensitivity to these hyperparameters, and the paper lacks corresponding ablation studies.

3. The proposed approach is neither plug-and-play nor zero-shot. As I understand it, it requires manually collecting a sub-dataset before fine-tuning can begin. Such a high “startup cost” could hinder its scalability and practical adoption.

### Questions
The authors need to address the above weaknesses. In addition, I have a few further questions and suggestions.

1. If I want to use multiple complementary retrievers in parallel, how should the algorithm be modified specifically? Have the authors explored this direction? I am also curious whether the differences in query representations extracted by multiple complementary retrievers would have an impact on the final results.

2. Figure 1 does not provide sufficient information. I suggest that the authors integrate Figure 4 into the main text.

3. The visuals in Figure 2 and Figure 3 appear rather sparse and monotonous. It would be better if these figures were made more visually appealing.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the problem of efficient visual document retrieval and proposes Guided Query Refinement (GQR), a test-time optimization framework that refines a primary retriever’s query embedding using guidance from a complementary retriever. By integrating rich cross-model interactions often missed by standard hybrid methods, GQR enables ColPali-based multimodal retrievers to achieve state-of-the-art performance. Extensive experiments demonstrate that GQR effectively advances the Pareto frontier of performance and efficiency in multimodal document retrieval.

### Strengths
1.	This paper innovatively adopts a vision-centric aggregation paradigm, supplemented by a text-centric approach, to address the issues of excessive modality gaps and high computational and storage costs in visual document retrieval tasks.
2.	The arguments in this paper are well-substantiated, and the experimental section is thorough and comprehensive.
3.	The description of the GQR algorithm in this paper is clear and easy to understand; even though there are many formulas, they are still easy to follow.

### Weaknesses
1.	The figures in this paper are somewhat rough, and Figure 4, which illustrates the core GQR architecture, should not have been placed in the appendix.
2.	The introduction does not provide sufficient background information and lacks an explanation of the necessity of applying hybrid retrieval methods.

### Questions
1.	The paper does not include experiments on the number of iterations T. How does this hyperparameter affect performance?
2.	In the introduction, the authors claim that the hybrid retrieval approach aims to leverage the advantages of dense text retrievers in terms of lower time and storage costs. However, the main model proposed in this paper is a multimodal, vision-centric encoder, and the dense text retriever merely provides additional supervision. So where exactly does the efficiency advantage of the hybrid retrieval paradigm manifest?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Guided Query Refinement (GQR), a novel test-time hybrid retrieval method. Specifically, GQR iteratively optimizes the representation of a primary retriever using a complementary retriever as a guidance signal. The optimization is performed via gradient descent to minimize the KL divergence between the primary retriever’s distribution and a consensus distribution (initialized as the average of the primary and complementary retrievers). Experimental results show that the proposed method achieves performance comparable to SOTA models while being significantly more efficient.

### Strengths
1. The proposed GQR method is an effective approach to achieving an impressive performance–efficiency trade-off.
2. The experiments effectively validate the proposed method’s advantages compared to both base retrievers and hybrid baselines.
3. The finding that a weaker retriever can still provide a complementary signal to improve a stronger retriever is interesting.

### Weaknesses
1. The authors should clarify the “Score Aggregation – Tuned” results in Table 2. From my current understanding, the Score Aggregation method simply merges the scores from different retrievers. If this is correct, the Score Aggregation (Min-Max) – Tuned method (with just an arithmetic weighting) only drops 0.5% in performance but is much more efficient than GQR, which requires gradient optimization.
2. The hyperparameters are selected based on the task, raising concerns about generalization.
3. There is no ablation study on the impact of the step T on performance.
4. There is a lack of exploration regarding how to choose the consensus distribution and how to define the distance between the consensus and the primary retriever distribution.

### Questions
1. Could the authors clarify each method listed in Table 2?
2. How does the step T influence the performance of the proposed method?

### Soundness
3

### Presentation
2

### Contribution
3
