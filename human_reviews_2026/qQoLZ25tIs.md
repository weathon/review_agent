# Neural Diversity Regularizes Hallucinations in Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Language models continue to hallucinate despite increases in parameters, compute, and data. We propose *neural diversity* — decorrelated parallel representations — as a principled mechanism that reduces hallucination rates at fixed parameter and data budgets. While existing mitigation strategies largely target accuracy, we provide the first formal tail bounds for hallucination probability in ensembled language models, reframing it as a second-moment reliability problem and *explaining 96.2% of empirical reliability variation* seen across parallel configurations. We introduce ND-LoRA (Neural Diversity Low-Rank Adaptation), combining parallel LoRA adapters with Barlow Twins regularization, and *reduce hallucinations by up to 25.6% (and 14.6% on average)* while preserving general accuracy. Ablations show LoRA adapters and regularization act synergistically, causal interventions prove neurodiversity as the mediating factor and correlational studies indicate scale: a 0.1\% neural correlation increase is associated with a 3.8\% hallucination increase. Finally, task-dependent optimality emerges: different tasks require different optimal amounts of neurodiversity. Together, our results highlight neural diversity as a third axis of scaling — orthogonal to parameters and data — to *improve the reliability of language models at fixed budgets*.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper explores a connection between higher diversity and hallucination rate in small language models.

### Strengths
- The paper investigates a relevant research question: how to reduce the rate of hallucinations of a small language model.
- The paper presents a broad set of experiments on several evaluation benchmarks.

### Weaknesses
The clarity and rigor of the paper could be greatly improved. 
- The manuscript is often not self-contained and some of the terminology is not defined before being mentioned. For instance, the abstract mentions “neural diversity”, without describing what it refers to, Figure 1 shows metrics that are not well-defined at this point, line 52 makes a statement about “parallel streams” without properly introducing what they are etc.
- Several claims in the paper are not sufficiently precise. For instance, the contributions in lines 74-76 state “17.9% reduction at 1.12x cost” (reduction in what? what cost?). Moreover, the preliminaries introduce some notation, without properly describing what it refers to. What are the $m_i$ streams? What is the hallucination event and why can it be defined like in line 99?
- Section 2.2 is rather difficult to read. It is not clear what the main takeaways are, and why theorem 1 is significant. It references results that are not present in the main text (a certain Corollary in line 144) and notation that is never defined (e.g. $\overline{\kappa}$).

The paper claims to establish a causal link between diversity and the rate of hallucinations (line 75). However, it is unclear what evidence points to that.

**Minor issues**

- incorrect references in the latex document e.g. lines 315, 583, 600 etc
- figures 1 and 2 are never references in the main text

### Questions
- In lines 100-101 it is stated “The margins [...] represent stream confidence—positive indicates correctness, negative indicates error.” Generally, for prediction tasks, confidence and accuracy are completely independent metrics (a predictor can be correct or not while being confident – or not). Why is it the case that here high confidence is correlated with correctness?

### Soundness
1

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
4

### Summary
The paper argues for 'neural diversity', i.e., diversity in model architecture, as an important component to reduce hallucinations in parallel scaling of small language models. It builds on the work of ParScale, a framework for model scaling by running parallel streams of the model, and argues that these parallel streams should also produce diverse features to reduce hallucinations. The paper supports its arguments, first using theoretical bounds and then with empirical experiments for the same.

### Strengths
1. The inclusion of 'ND-LoRA' by the paper to increase diversity clearly reduces hallucinations, without a significant increase in computation cost. In other words, the biggest strength of the paper is that the technique proposed works. While additional experiments are always appreciated, I believe the current set of experiments are robust enough to suggest that the technique can be expected to work in other settings.

2. Ablation studies and experiments across multiple datasets show robust benefits of the technique proposed by the paper.

3. The theoretical bounds and discussions are solid (although not novel, discussed further in Weaknesses), and thus the work is grounded in strong foundations.

### Weaknesses
Comment on Related Work: The paper needs a better treatment of related work. Many weaknesses discussed below will refer back to this particular lack of appropriate discussion of related works in the paper. There is a section on Related Works towards the end of the paper, but it does not do justice to the rest of the discussion in the paper.

1. I fail to separate the novelty of the theoretical analysis provided in the paper from those that already exist in the ensemble literature. From the general discussion of diversity (see [1], [2]) to an 'optimal ensemble size' (see [3], [4]), and even specific claims like 'signal-to-noise improves by root(P)' (see [4]), have an extensive history in ensemble literature. The references provided are only examples, there is far more literature on ensemble theory. While the discussion in the paper is provided in the context of 'hallucinations', which could have been a novelty, the theoretical analysis simply reduces the hallucination aspect to margins at the very start and thus loses any context-specific addition that it can provide to the literature. 
The paper briefly acknowledges this in Related Works, based on which, it seems that the main contribution is simply that existing literature assumes separate models, but the paper shows benefits for the same architecture. Firstly, separate 'models' and same 'architecture' are two different things, and are not comparable. Secondly, the paper never makes use of 'same architecture' in their theoretical analysis, which only assumes different features (hence, actually assumes different models, same as in the literature). And finally, none of this is reflected in the rest of the paper, which is framed as if the theoretical discussions are contributions of the paper.

2. The connections with ParScale and existing discussions of diversity in the original ParScale paper should be acknowledged and also further discussed in this paper. For instance, ParScale paper claims that using learnable prefixes provides 'diverse enough' feature outputs, which is in contrast with the claim made in the introduction of this paper that naive scaling can degrade reliability (this claim is also in contrast with the empirical results provided by this paper itself in the ablation study). To my understanding, the paper needs to readjust its claim to say that explicitly incorporating diversity can 'further' improve reliability (reduce hallucinations), or add a discussion on why their claim stands despite existing literature and their own results.
On a similar line, the inclusion of LoRA is surprising to me. Original ParScale paper experiments with LoRA alongside the learnable prefix, and concludes that the prefix is a better choice. Why was then LoRA used in this paper? Arguably, LoRA has a higher rank in this paper, compared to the original ParScale paper, which might be the difference (again, this is based on just my observations from the two papers, and no such discussion of how their choices build on previous work exists in this paper). A proper motivation behind the choices made is needed (a simple ablation study is not enough).

3. There is a missing corollary (line 144) in the theoretical analysis, and a missing figure (line 315) in the key results. The missing corollary is not too important in context, since I don't believe the theoretical analysis is a novelty. However, the missing figure is quite important, as it supposedly contains the connections between theoretical bounds and empirical results.

4. I don't like the claim in the abstract, 'ND-LoRA achieves 17.9% hallucination reduction with 1.12× training cost'. The 17.9% improvement occurs only in one subset of one dataset, and the rest of the results are in the range of 1-8% improvements. To be clear, these improvements are still good. I understand the choice made by the authors to showcase their strongest result, but still, the language is misleading. I would recommend (although I'm still not a fan of this) something like 'ND-LoRA achieves up to 17.9% hallucination reduction in some dataset with 1.12× training cost'.

References

[1] Dietterich, Thomas G. "Ensemble methods in machine learning." International workshop on multiple classifier systems. Berlin, Heidelberg: Springer Berlin Heidelberg, 2000.

[2] Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2014.

[3] Bonab, Hamed, and Fazli Can. "Less is more: A comprehensive framework for the number of components of ensemble classifiers." IEEE Transactions on neural networks and learning systems 30.9 (2019): 2735-2745.

[4] Hernández-Lobato, Daniel, Gonzalo Martínez-Muñoz, and Alberto Suárez. "How large should ensembles of classifiers be?." Pattern Recognition 46.5 (2013): 1323-1336.

### Questions
1. Is there work that shows 'naive scaling' can degrade reliability (as claimed in the introduction)? It seems, based on Table 3, that even naive parallel scaling increases the score (no doubt that increasing diversity works even better, but is there a reasoning behind the base claim itself?).

2. Are there contributions in the theoretical foundations that are novel? If so, I recommend highlighting them explicitly, by contrasting with what has been done before and what the paper adds to the literature.

3. Please provide the missing information, as highlighted above in the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper proposes a theoretical framework linking neural diversity (quantified by cross-stream correlation) to hallucination probability, aiming to show that decorrelated neural representations directly lower the likelihood of false outputs. The paper proposes neural diversity as a novel scaling axis, complementary to model size and data volume, for reducing hallucination rates in small language models. Towards this end, the authors introduce ND-LoRA (Neural Diversity Low-Rank Adaptation), which integrates multiple independent LoRA adapters using the Barlow Twins regularization to encourage representational decorrelation among parallel computation streams. Experiments employing 8 benchmarks (TruthfulQA, HaluEval, MemoTrap, Natural Questions, TriviaQA, PopQA, Wikitext, Winogrande) with ND-LoRA are performed on Qwen2.5-0.5B with up to 8 streams showing that the proposed method is able to achieve an up to 17.9% hallucination reduction.

### Strengths
- Theoretical contributions reflect well in the experimental results (specifically, the relationship between reliability and neural diversity and the existence of an optimal neural diversity level in an U-shaped curve).  
- Careful evaluation, accounting for parameter matching and statistical significance thresholds and confidence intervals, showing significant hallucination mitigation results at a minimal induced computation overhead.  
- While the method specifically targets mitigating hallucinations, experimental results show it's not detrimental to overall performance.

### Weaknesses
- Experiments show high variability in optimal number of streams across tasks, implying repeated training or nontrivial hyperparameter search to find the sweet spot.  
- Limited scale, with a small-scale backbone of 0.5B, 20M data tokens and context length of only 1k tokens (as far as I understand, many real-world hallucinations surface in longer chats where retrieval of the right snippet is harder, so external validity to long-context use remains uncertain).  
- Concern about LoRA rank as a confound, given that several comparisons rely on parameter matching achieved through changing LoRA rank (making it 4 times smaller for instance in Table 1), as rank affects both expressivity and optimization dynamics and might confound conclusions about neural diversity.

As a small observation, there are five missing references at lines: 144, 315, 583, 600, 604.

### Questions
1. Theory assumes linearized properties and whitened features at the design layer. Might this be an over-assumption in practice? Have the authors considered any means of quantifying whether this occurs in real-world models?

2. Can an heuristic be found to choose P beforehand? Is there a way around training multiple models in order to try out multiple values for P?

### Soundness
3

### Presentation
3

### Contribution
3
