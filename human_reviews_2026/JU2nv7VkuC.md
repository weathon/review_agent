# Disentangling Locality and Entropy in Ranking Distillation

- Avg Score: 6.00
- Decision: Reject
- Scores: 8, 6, 6, 4

## Abstract
The training process of ranking models involves two key data selection decisions:
a sampling strategy (which selects the data to train on), and a labeling strategy
(which provides the supervision signal over the sampled data). Modern ranking
systems, especially those for performing semantic search, typically use a “hard
negative” sampling strategy to identify challenging items using heuristics and a
distillation labeling strategy to transfer ranking “knowledge” from a more capable
model. In practice, these approaches have grown increasingly expensive and
complex—for instance, popular pretrained rankers from SentenceTransformers
involve 12 models in an ensemble with data provenance hampering reproducibility.
Despite their complexity, modern sampling and labeling strategies have not been
fully ablated, leaving the underlying source of effectiveness gains unclear. Thus,
to better understand why models improve and potentially reduce the expense of
training effective models, we conduct a broad ablation of sampling and distillation
processes in neural ranking. We frame and theoretically derive the orthogonal
nature of model geometry affected by example selection and the effect of teacher
ranking entropy on ranking model optimization, establishing conditions in which
data augmentation can effectively improve bias in a ranking model. Empirically,
our investigation on established benchmarks and common architectures shows
that sampling processes that were once highly effective in contrastive objectives
may be spurious or harmful under distillation. We further investigate how data
augmentation—in terms of inputs and targets—can affect effectiveness and the
intrinsic behavior of models in ranking. Through this work, we aim to encourage
more computationally efficient approaches that reduce focus on contrastive pairs
and instead directly understand training dynamics under rankings, which better
represent real-world settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates the complexities of training ranking models, particularly focusing on the interplay between sampling strategies and distillation processes in neural ranking systems. The authors conduct a comprehensive study to elucidate the effects of example selection and teacher model entropy on ranking model optimization. They derive theoretical insights into the geometry of model training and provide empirical evidence.

### Strengths
1. The paper presents a generalization bound that connects data selection factors to model effectiveness.
2. The authors conduct extensive experiments across multiple benchmarks (TREC Deep Learning 2019/2020 and BEIR) to validate their claims, demonstrating the robustness of their conclusions.

### Weaknesses
1. The connection between the theoretical analysis and the empirical experiments could be better explained. While I gradually understand their connections when reading the experiment analysis, I was almost lost at the end of the theoretical part, not sure where the theory would lead us to.
2. Similar to the previous point, this paper would benefit a lot from a more detailed discussion on the implications and potential applications of the theoretical and empirical discovery. 
3. While the authors criticize existing works on multi-stage distillation training for the waste of energy, there isn’t a direct comparison or analysis of the energy cost between different systems and how the discovery of this paper could help. While multiple-stage training is generally not favored, this may not be a huge concern if the models are not large (much smaller than general LLMs) and can converge fast.

### Questions
From what perspectives the discovery can directly help us reduce carbon footprint for retrieval model training? Any detailed experiments?

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
This paper studies to disentangle two ingredients in neural ranking training – locality (data sampling) and entropy (from teacher).  The authors point out that the modern neural ranking models heavily rely on hard-negative mining with multiple stages without understanding the two key aspects that could independently contribute to the resulting model performance. The paper proposes a theoretical framework to describe how these two factors (locality and entropy) affect generalization in ranking distillation. Through comprehensive ablations, the authors show a) elaborate sampling pipelines yield negligible improvements over simple heuristics under distillation, and (b) moderate teacher entropy improves performance, while overly confident or noisy teachers degrade it.

### Strengths
* Intriguing study that questions the standard recipe (multi-stage hard negative mining) for neural ranking and seeks a deeper, more principled understanding of its training dynamics.
* Presents a theoretical framework that rigorously examines generalization bounds through the lenses of locality and entropy.
* Offers evidences that once locality and entropy are properly managed, complex multi-stage training pipelines add limited value.

### Weaknesses
* The experimental configuration is somewhat narrow, particularly in its treatment of negative sampling. The paper relies on simplified strategies rather than the more sophisticated dual-encoder mining pipelines commonly used in practice.
* The evaluation is restricted to a limited set of ranking benchmarks, excluding broader or multitask-oriented suites such as MTEB that better capture generalization across domains. As a result, the findings may not fully extend to modern multitask embedding training setups (multi-task, generalization to unseen tasks).
* The second observation regarding entropy has been explored in prior work, as noted below.

### Questions
* Did the authors evaluate alternative negative mining strategies such as nearest neighbor retrieval using separate dual encoders, as employed in approaches like RocketQA [1]?
* The second part – overly confident teachers degrade performance was discussed in [2]. Can authors compare their findings with the current findings?

[1] Qu, Yingqi, et al. "RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering." arXiv preprint arXiv:2010.08191 (2020).

[2] Menon, Aditya K., et al. "A statistical perspective on distillation." International Conference on Machine Learning, 2021.

### Soundness
3

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
2

### Summary
This paper investigates the joint influence of negative sampling locality and teacher entropy on neural ranking distillation. The authors derive a generalization bound involving the query manifold diameter and teacher pairwise entropy, providing a theoretical explanation for when hard negatives or complex distillation pipelines are (or are not) beneficial.

### Strengths
Hard-negative mining and ranking distillation are widely used; the paper addresses a real gap in IR training methodology.

### Weaknesses
1. Theoretical reasoning is largely descriptive, not prescriptive
The generalization bound identifies factors but does not provide actionable guidance for selecting sampling policies or entropy levels in practice.
The theoretical exposition is mathematically sound but overly symbol-heavy.

2. While the study provides clear insight into bi-encoder retrievers trained via distillation, it remains unclear whether the identified effects of locality and teacher entropy extend to LLM-based retrieval settings

### Questions
The experimental findings indicate that moderate teacher entropy provides the best performance.
How should practitioners determine or tune this entropy level in practice?

All experiments appear to use a single teacher configuration.
Would the observed conclusions still hold if the teacher were substantially weaker or stronger?

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
Modern neural ranking systems combine hard negative sampling and knowledge distillation to train models efficiently. Due to sparsity of human annotations, especially in MS MARCO, naive sampling procedures can produce false negatives hurting model’s performance. To overcome this issue, we often rely on expensive methods to filter out false negatives, e.g., SentenceTransformers use 12 cross-encoder models.  Their benefits are poorly understood, as the effects of sampling (data locality) and labeling (teacher entropy) are entangled. The paper aims to disentangle these two influences—locality and entropy—to determine which truly improves ranking performance. To this end, authors: 


1. Establish a new generalization bound. 
2. Carry out experiments showing that benefits of “smarter” sampling (i.e., with post-filtering) can be limited, which they “connect” to their theoretical result. 


Authors also mention that they studied the role of data augmentation, but I found no evidence of this in the paper. Although  I like the direction of work and empirical observation, I found the paper to be extremely confusing, experiments to be somewhat limited, and the bound seems to be vacuous.  


I understand that theory is hard and deriving any bounds is a step forward. It is great to have both empirical and theoretical results. However, the narrative “treats” this bound as a tight one, but this is unlikely to be the case. I also suspect that some experimental results are due to training instability rather than genuine trends. I think it can be potentially a great paper, but it requires a thorough revision.

### Strengths
1. An important, although well-studied problem. 

2. A novel (though potentially vacuous) bound that connects the diameter of the query and the teacher’s entropy to the generalization gap. 

3. Paper has both in-domain and out-of-domain experiments.

### Weaknesses
1. The paper is quite confusing, it uses inconsistent terminology (while not defying some crucial parts). It also makes quite a few technical claims (although tangential to their main results) that seem to be incorrect. Math is confusing: formulas appear to have errors (e.g., the risk minimization definition) and unexplained symbols (e.g., eta). Data augmentation is mentioned, but never explained properly. More detailed comments are below.  

2. The proposed bound  appears to be vacuous because it involves a VC dimension, which is astronomically large in modern neural networks (see references  and discussion in the question section). 
 
3. The experimental results are rather limited. Some of these are nice-to-have experiments, but others are more important: 
* Only one training collection is considered. In that MS MARCO is known for its sparse judgements, how about some other collections that are different (e.g. Natural Questions)?. **Important addition** 
* Retrieval uses only BM25, but not stronger bi-encoder models: **Crucial addition** 
* There is seemingly only one training seed per experimental scenario. It is quite possible that some of the experimental results are due to seed-related randomness: **Crucial to verify** 
* Some experimental settings are poorly motivated (see detailed comments). 

* Last but not least, authors consider only basic approaches to sample and filter out false negatives. However, quite a few recent approaches were proposed. Some of the use LLM-judges. I don’t suggest trying every new approach, but something stronger than CE-based filtering should be considered IMHO. **This is in the nice-to-have category, i.e., I would not reject the paper if it didn’t have it.** 
 
Some links: 
1. Negative Sampling Techniques for Dense Passage Retrieval in a Multilingual Setting 
I recommend consulting the following two papers/preprints, which also have references for older approaches as well (ADORE, STAR, AR2, RocketQA): 

PROD: Progressive Distillation for Dense Retrieval. Zhenghao Lin, Yeyun Gong, Xiao Liu, Hang Zhang, Chen Lin, Anlei Dong, Jian Jiao, Jingwen Lu, Daxin Jiang, Rangan Majumder, Nan Duan, 2023 

TriSampler: A Better Negative Sampling Principle for Dense Retrieval. Zhen Yang, Zhou Shao, Yuxiao Dong, Jie Tang, 2024. 

 

**Detailed comments:** 

 

**Do not respond to this list, unless you feel like it’s a big-time misunderstanding on my side that could change my opinion of the paper. These are just for your information. Questions here are rhetorical. Thank you!** 

L069-074. This is not clearly worded. How is epistemic uncertainty relevant here? How does it connect to the appeal of distillation? BTW, it’s not even very clear why it’s an appeal. 
 
> that multiple relevant documents may be present and optimised within a single instance. 

It’s not clear what an “instance is”. In the contrastive ranking framework you can surely select multiple positives and multiple negatives per query. I can argue that a loss term (even if it can be additively decomposed into terms with a single positive) is an instance. 

L108: is that as we produce an ever stronger source of negatives, due to label sparsity, we inevitably sample false negatives -> stronger source of negatives produces negatives that are strongly negative, doesn’t it?  
 
as such the notion of de-noised negatives has been proposed -> This is again a very vague phrase. You should probably call it “filtering out of false negatives”, de-noising is not sufficiently specific. 

Moreover, given that a notion of relevance is, indeed, subjective, the paper would benefit from a short discussion on what “noise” is, how it can negatively affect performance. You should IMHO be more specific that the filtering pipeline is filtering out false negatives rather than some unspecified noise. 

See, e.g.: 
 
Nandan Thakur, Crystina Zhang, Xueguang Ma, and Jimmy Lin. 2025. Hard Negatives, Hard Lessons: Revisiting Training Data Quality for Robust Information Retrieval with LLMs. EMNLP Findings. 

L117  where a larger number of elements would not be ranked **without being utilised** -> what does it mean? 

 

L131-134 In terms of explicit generalisation bounds, Hsu et al. (2021) provide a bound under uniform convergence, using distillation as a vector for understanding the original teacher model, we diverge from this setting as in downstream Information Retrieval we focus on trading off effectiveness for reduced latency -> I can’t see how these two statements (using distillation as a vector for understanding the original teacher model and a following one) are related. Moreover, it’s not understandable what “distillation as a vector for understanding the original teacher model and a following one” means on its own. 

L146-147 I think this is not a grammatical sentence, because it lacks a verb.  

 

L151 I disagree that the ranker is trained as a regression model. Where does this come from? It’s a classification problem with typical classification losses like the margin loss or soft-max-like loss InfoNCE. 

L153 I also think it’s incorrect what you claim about bi-encoders. Both bi-encoders and cross-encoders are trained contrastive losses and the problem is still a classification problem. The main difference with bi-encoders is that the relevance score is computed as an inner-product between two vectors, but the loss is still a classification loss. 

L157 It’s not clearly what solely positive means. 

L158 What is the importance of citing MS MARCO here? The annotation procedures were not introduced by MS MARCO folks. 

L163 It’s not clear what the teacher model g() produces: labels or logits? 

L165-173 This is hardly understandable. First, why do we call them pseudo-negative? Second, you didn’t mention / describe the sampling approach. You should say that typically negatives are sampled from a top-k returned by a retriever. However, not all of the negatives are true negatives due to the sparsity positive labels (i.e., it is not feasible to find and annotate all positive examples, in particular, because this procedure is retriever-dependent. We can find all positive documents returned by one retriever. However, another retriever may “bring” other top positives to top-k). This is, of course, exacerbated by the problem that you touch upon in the next paragraph: what is a positive and what is a negative is not well defined, i.e., a problem. This will be a good place to describe what you mean by “noise” and not in the next paragraph (problem setting). 

Then you should say that there are heuristic approaches to filter out false-negatives (not pseudo-negatives!).  

L179 What’s an auxiliary data? In fact, if you use something like a black-box LLM ranker, there is a non-trivial chance it was trained on your test data: TREC DL or BEIR. These are public collections with publicly available test sets. 

L186 Metric structure matters fundamentally because -> Which metric are you talking about? 

L190-196 This whole paragraph is not really understandable. Without describing RankNet you can’t claim it blurs the boundaries. In fact, I think this not correct, because RankNet is not a distillation loss per se. Virtually any supervised classification loss can be turned into a distillation loss by replacing hard labels with normalized teacher scores (e.g., sigmoids of logits). 

In L193 you mention some downstream task and observations. Why do you need a vague term observation when you already defined one as a set of query-document pairs? This is particularly confusing given you don’t describe top-k sampling before and assume that everyone knows it. 

L213 Why doesn’t Eq (2) incorporate ground-truth ordering (I think it should!)? Are we actually summing up for all pairs when x is ranked higher than x’ or vice versa?   

  

I have the same question for Eq (4) and Definition A.3 and Definition A.7?   

  

A follow-up question regarding Eq (4): Is it true for any probabilities or only for the RankNet style probabilities defined as a sigmoid of score differences?   

  

L215 Equation (2) seems to be incorrect because it does not incorporate ground-truth ordering. 

L216 you need to explain why you deal with generic Bregman divergences and not just KL-divergences. It becomes clear in the experimental section, but that’s too late! 

L246 Applying a PAC bound affects model preference for a hypothesis within a class, effectively governing how a model will generalize. -> I hate to nitpick here, but the PAC bound itself doesn’t govern anything.  

 

L320 by optimising the margin between positive (Xi) and negative (Xj ) elements instead of pointwise scores (Hofstätter et al., 2020). -> I think “instead of” is out of place. It sounds like you optimize 

L325 “It sees increasing application with increasing precision of modern ranking models as it optimizes all x, x’ interactions agnostic of human labels. -> It is not clear what it means and requires a citation. “Interactions agnostic of human labels” is a particularly dubious definition/claim. 

 

 

L329 What is f() in Eq 10-12?  Please clarify/fix 

   

What does eta mean in Theorem 2.1? BTW, it’s not bad to remind that g() is a teacher scoring function.   

 

L340 We show these semi-supervised criteria can be expressed as Bregman divergences -> Only now we learn why your theory is applied to Bregman divergences and how distillation is related to the RankNet loss. This should have been defined much earlier! 

 

L342 All models observe a total of 12M documents -> what does it mean to observe?   

 

L350-354 We consider four sampling sources in our investigation, aligning with those applied in literature. The first is uniform selection from the training corpus (Random). The second is a lexical heuristic BM25 (k1 = 1.2, b = 0.75) (Robertson et al., 1995), a lightweight retrieval model. **The third is our teacher model**, a cross-encoder (CE). Finally, we apply the ensemble pipeline shown in Figure 1 (Ensemble). 

How is a CE model is a sampling source?  

 
Figure 1: I don’t see an ensemble there. 

It is rather confusing than use the term semi-supervision after definining the process as teacher-student learning. I would argue that you should continue to use terms distillation (distillation training) and distillation loss for consistency. 

L355 s where investing computational budget in a strong estimator -> Clarify what is an estimator? You probably should use a different term for a false-negative filtering procedure. 

L360 You mentioned localized sampling a few times (and also at this point). I believe you never defined it properly. 

L369 “empirical values of the query-specific diameter show that the query space does not become more compact.” 

Table 1 needs BM25 as a baseline. 

I think random is a confusing term, because all negatives are selected randomly. What differs is the selection set and a post-filtering procedure. I would argue that corpus-sampling is a better term than random in Table 1.

### Questions
0. Where do you talk about data augmentation? Both empirical and theoretical? 

1. There is evidence that VC-dimension of neural networks is astronomical. Doesn’t it make the bound in Eq (8) vacuous? For example, Barlett et al prove it is in the order of WL log(W/L), where W is a number of weights and L is a number of layers. For 10 million neural net with 12 layers, the VC dimension is about 10^10. Even with the number of “observed” data points in the order of one million, the square root in Eq (6) is at least 100. And what do we know about constant C? How large is it? 

     * Bartlett, Peter L., et al. "Nearly-tight VC-dimension and pseudo dimension bounds for piecewise linear neural networks." JMLR 2019  

     * Zhang et al 2017 (UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION) 

 

 

2. Can you explain the following training irregularities in Table 1? Can they be due to training instabilities, i.e., different seeds would produce very different results? 

       * For CE on BEIR and LCE loss corpus-level sampling (denoted as Random) works better? If post-filtering hurts, why does ensemble-sampling work at par with Random?   
        * For distillation losses RankNet and KL-div, ensembling actually hurst performance of CE on BEIR. However, this is not the case for MSE. This, again, looks more like a training instability rather than a genuine pattern.  
 

 3. How many seeds did you use per experiment?  

4. In Section 2.2 you define a metric-measure space and use an essential supremum. How do you define the measure μQ?  

5. Did you train bi-encoders without in-batch negatives? Any comments on this in the paper?   

 6. “To ensure reproducibility and clear attribution of effectiveness, we train a cross-encoder following (Pradeep et al., 2022) using the ELECTRA architecture trained for one epoch with BM25 localised negatives on the MSMARCO passage training set.” -> How is this related to reproducibility? What do we try to reproduce? BTW, “BM25-localized negatives” isn’t proper terminology IMHO.

### Soundness
2

### Presentation
1

### Contribution
2
