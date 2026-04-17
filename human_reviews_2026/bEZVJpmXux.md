# Ensembling Sparse Autoencoders

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Sparse autoencoders (SAEs) are used to decompose neural network activations into human-interpretable features. Typically, features learned by a single SAE are used for downstream applications. However, it has recently been shown that SAEs trained with different initial weights can learn different features, demonstrating that a single SAE captures only a limited subset of features that can be extracted from the activation space. Motivated by this limitation, we introduce and formalize SAE ensembles. Furthermore, we propose to ensemble multiple SAEs through $\textit{naive bagging}$ and $\textit{boosting}$. In naive bagging, SAEs trained with different weight initializations are ensembled, whereas in boosting SAEs sequentially trained to minimize the residual error are ensembled. Theoretically, naive bagging and boosting are justified as approaches to reduce reconstruction error. Empirically, we evaluate our ensemble approaches with three settings of language models and SAE architectures. Our empirical results demonstrate that, compared to the base SAE and an expanded SAE that matches the number of features in the ensemble, ensembling SAEs can improve the reconstruction of language model activations, diversity of features, and SAE stability. Additionally, on downstream tasks such as concept detection and spurious correlation removal, ensembling SAEs achieve better performance, showing improved practical utility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces two techniques for ensembling sparse autoencoders (SAEs): naive bagging (training many SAEs in parallel and then averaging their outputs) and boosting (sequentially training SAEs to reconstruct the error of the previous one). It then shows that ensembled SAEs score comparably to or better than appropriate baselines SAEs on various metrics adapted from prior literature.

### Strengths
1. The paper is very well-written.
2. The methods are well-motivated and clearly explained.
3. The authors compare against an appropriate baseline "expanded" SAE that was trained from scratch with the same width as (and comparable sparsity to) their ensembled SAEs.

### Weaknesses
I will discuss both weaknesses that I would like to see addressed to raise my score and various pieces of advice to the authors to improve presentation that do not affect my score.

Weaknesses that affect my score:
1. Looking at the quantitative results in table 2 (including the confidence intervals), I think those results should be summarized as ensembled SAEs performing comparably to (or perhaps slightly better than) the baselines. This would be clearer if the results were presented as a bar chart with error bars. To be clear, I don't think this weakens the results substantially; it's valuable to improve SAEs along one measure (SCR, which in my opinion is a more important measure of SAE quality) while not making them worse in other ways.
2. I find it concerning that the results in sections 5.3 and 5.4 use only one model (Pythia-70M), which is a different model from the three used in section 5.2. The authors write that they do this because Pythia-70M was used in prior work involving concept-level tasks, but this does not make sense: SAEs are an interpretability tool that is always meant to be applied to concept-level tasks. This choice also makes it difficult to compare to SAEBench; this is important because the SCR scores look very small overall (0.01-0.06 in this paper, compared to 0.2-0.4 in SAEBench). I request that the authors replicate these experiments on Gemma-2-2B and Pythia-160M; this should not be difficult because they have already trained the SAEs. This will allow greater confidence in the results (important given that the effect sizes are not very large) and allow for easy comparison with SAEBench.
3. Similarly, I request that the authors score their SAEs on the other metrics from SAEBench instead of only the SAEBench two metrics reported in section 5.3 and 5.4 and the less common intrinsic metrics in section 5.2. The point of this is ensure that the two metrics reported were not cherry-picked. To be clear, I'm not looking to see that ensembling improves on all metrics; it is okay if ensembled SAEs are comparable or even slightly worse on some of the metrics. But this provides useful context; e.g. it seems important to know if the boosted SAE's new features are less interpretable, according to autointerp score.
4. If I'm understanding correctly, the boosted SAE is not only trained sequentially, but also needs to be inferenced in multiple sequential steps. This should be emphasized, since it's a relatively fundamental change to the SAE architecture. It would also be good to see the results of training an SAE with this modified architecture end-to-end from scratch. (This is like the expanded SAE baseline, but for boosting.)
5. Since you're choosing the sparsity of the baseline SAE to be comparable with the ensembled ones, it's slightly misleading to use relative sparsity as a quality metric in table 1. That said, it's still important information to include in the table, so I think the table 1 caption should just clarify the situation.
6. Please report the number of runs that you used for computing stability. 

If all of the above weaknesses are addressed and the new results don't substantially change the qualitative story then I will raise my score to an 8. I'll raise my score an intermediate amount if some are addressed. (2) and (3) are the most important.

Other points that don't affect my score:
1. It'd be good to put horizontal lines for the expanded SAE baseline in figure 2.
2. Many of the tables of numbers would be better presented as bar plots with error bars. If it's too many bars, then scores can be averaged across categories with detailed results in the appendix.
3. Typos. there is a dangling comma after the equation preceding line 180. The TopK nonlinearity is not actually an elementwise function. On line 882, "consine" should be "cosine."

### Questions
See weaknesses.

### Soundness
2

### Presentation
4

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
The authors take the finding from previous literature that often SAEs trained on the same data can learn different features and ask if they can leverage this diversity for better performance. The way they propose to do this is via classical ensembling methods from the supervised learning literature: boosting and bagging. They show that ensembling reconstructions is generally equivalent to ensembling features. With their methodology they show improved performance relative to the baseline of the expanded SAE they achieve better explained variance and MSE. This is an interesting and non-obvious result! 
It seems like the bagging approach didn’t perform particularly well despite being intuitively the inspiration for the project. I would say that this likely means that the story given for why the approach works in terms of the variability of SAEs is probably not the best explanation for their good performance (it may slightly contribute but doesn’t seem to be the leading term). In fact, this implies that the reason that the approach works well is the ability to leverage information from the error terms of previous SAE reconstructions and use that as signal for training. It would be interesting to explore that motivation much more and how this connects to previous work in Compressed Sensing and Digital Signal Processing. 

Overall a cool paper that has good empirical results, though possibly the link between the motivation and empirical results may not be as strong as it appears. The paper doesn’t necessarily solve a particular problem but nonetheless presents a possible improvement to the standard SAE training methodologies. It would be interesting to see if there are problems solved here (e.g. perhaps finding more finegrained features is solved with this method which could be evidenced by a feature identification study. I’d be open to increasing my score if some of the below comments are addressed.

### Strengths
- Clear written style
- Related work shows a good knowledge of the relevant literature
- Figure 1 is helpful in understanding the core technique
- The propositions and proofs are helpful in justifying the argument
- The mathematical framework is mostly clear (though the notation is non-standard; see below)
- Presents a useful way to improve the performance of SAEs. 
- Shows results on two downstream tasks
- The evaluation uses multiple models, architectures and tasks/metrics (though see below for the disadvantages of not having clear ablations and where some of the metrics may be misleading)
- The framework is architecture agnostic which allows this approach to be useful for most SAE paradigms, there remains open opportunities for future work to improve on this

### Weaknesses
- Nits: 
    - In Section 3.1 the activation functions and the citations are in different orders
    - Would suggest against using the letter k for the dimension of the SAE hidden layer as this conflicts with the k in the activation function for top k. Perhaps using f would be clearer. 
- All of the notation section uses quite non-standard notation - it would be good to use similar notation to e.g. Gao et al or a similar paper for readability
- A Pareto plot with Sparsity or Description Length (see Ayonrinde et al 2024) on the x axis and the metrics on the y axis would be very valuable - currently it seems that there’s at least one set of hyperparameters for which the authors approach outperforms but its unclear if this represents an improvement in many different settings
- I find Figure 2 to be generally quite misleading: the appropriate baseline should be the expanded SAE rather than the base SAE in all cases as this is the parameter matched (and possibly inference time FLOP matched) version 
- The metrics chosen seem to unfairly advantage the authors methods (especially in Figure 2)
    - For example relative sparsity is not the correct metric for this method as it gives an advantage to larger models by allowing them to have more features. In fact larger models should be able to achieve the same performance with fewer features (as shown in Bricken et al 2023 and many other papers) and so this gives large models a double advantage! 
    - Similarity the diversity metric has the opposite problem, here we really should have a relative metric because of course a model with more features will have more dissimilar features (similarly for the connectivity metric).
    - This makes many of the metrics here quite misleading as vanity metrics.
- Using different SAEs on different models makes it hard to tell if the approach works differently on transformers of different architectures, of different sizes or using different SAE architectures - it would be useful to have a clear ablation on this point.
- I would suggest moving the setup of your Use Cases to the appendix and focusing on the results in this section. This would allow more space for detailed discussion and motivation and for you to add the statements of your propositions in the main paper rather than in the appendices. 
- The main weakness of this work is that the motivation for the work doesn’t connect with what actually performed well - introduction motivates the bagging approach whereas in fact the boosting approach is what works in practise. 
    - I think editing the framing substantially to account for this and explain why the boosting approach is a reasonable way forwards would be a large improvement for the coherence of this work.
- All of the metrics provided are purely mathematical and there are no metrics discussing the interpretability of the learned features either with AutoInterp (e.g. Paulo et al 2025) or with a human study (e.g. Bricken et al 2023). An AutoInterp study (perhaps using SAE-Bench as the authors are already using this), could be a good addition to the paper.

### Questions
- Remark 1 seems to say that c could be folded into either c or W_enc however the reasoning given for this is about W_dec rather than W_enc - could you clarify on this point? 
- Would the naive bagging approach perform better if the ensemble weight $\alpha$ is allowed to be learned in finetuning rather than being restricted to 1/J? 
    - More full finetuning of the bagging approach when ensembled (i.e. allowing the individual SAE weights to be updated as well as just the ensemble weighting) could also be interesting for reducing the feature redundancy problem. 
    - Also note that Mudide et al gives a nice description of the feature redundancy problem which might be useful to leverage to make the argument given here 
    - I'm also interested in if there are other approaches attempted to address the redundancy problem e.g. could you cluster and prune very similar feature directions? 
-  Appendix C suggests that the L0 for the expanded SAE and similarly for the other approaches is 3-8k - is this correct? Normally L0 figures are <100. Is this multiplied by the batch dimension by mistake? 
- What is the wall clock time difference in training and inference for the boosting and bagging approaches? 
    - It seems that this is a considerable limitation and it would be good to quantify this.
    - Similarly for FLOPs. 
- The explained variance metric suggested that after 4 SAEs ensemble there are significantly diminishing returns for the naive bagging case - do you agree with this intuition? 
    - If so why use 8 SAEs for Table 1? I suspect there might be a better parameter/performance tradeoff using 4 SAEs
    - If you’re using comparing the 8 SAE to single SAE basline then for a proper comparison the MDL-SAE framework from Ayonrinde et al 2024 will be a metric that accounts for comparisons across SAE widths better than just sparsity which can be gamed with larger models. I’d be interested in a comparison using this metric
- I’m not totally convinced by the argument given for why NB excels in Table 2 - it seems like there should be some features which are useful in the NB case and which are learned by the Boosting model because they’re useful regardless of hierarchy?  
- How does this approach differ from Matryoshka SAEs (Bussman et al 2025)? This seems like the closest work and is not compared against. Similarly, Switch SAEs (Mudide et al 2025) is also not compared against which similarly seeks to expand SAEs (in their case in a MoE style)
- It would be useful to have a study showing feature identification here - a core question would be “in which cases do the approaches learn redundant or hyper-specific features that make the approach worse than the baselines?” 
- How many dead features are there in the expanded SAE baseline and what techniques were used to overcome dead features? Was the auxiliary loss from Gao et al 2024 used here? It seems possible that one of the reasons that the expanded SAE has poor performance is due to dead features.

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
4

### Summary
For a fixed model and dataset, the learned codes of SAEs vary across different architectures, latent dimensions, sparsities, and weight initializations. The authors turn to bagging and boosting for mitigation. They characterize their method via intrinsic statistics and evaluate the interpretability on two downstream tasks: concept detection and spurious correlation removal.

### Strengths
The overall paper is clearly written and the evaluations cover evaluations expected for the work on SAEs.

I encourage the authors to open-source the naive bagging + boosting by implementing it in one of the popular Dictionary learning repos, such as decoderesearch/SAELens or saprmarks/dictionary_learning.

### Weaknesses
I will defer to AC to judge whether the contribution is big enough to meet the ICLR bar. Idea for follow-up: Apply bagging across Specialized SAEs finetuend on subdomains (SSAE https://arxiv.org/abs/2411.00743).

### Questions
The method is purely evaluated with ReLU SAEs. The field has moved to TopK and BatchTopK nonlinearities, yielding better results across metrics (SAEBench https://arxiv.org/abs/2503.09532). How would you apply naive bagging for TopK / BatchTopK SAEs?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present two methods of ensembling sparse autoencoders (SAEs) that lead to better performance, both in reconstruction loss, as well as some downstream performances. They present a formalization of SAE ensembling as the ensembling of SAE features. Their two ensembling methods, naive bagging and boosting have different tradeoffs between their improvements with respect to simple SAEs and their costs. Bagging can be trained in paralel, while boosting needs to run SAEs in sequences, making it slower to train, while boosting normally achieves higher scores in most metrics.

### Strengths
The authors present a nice formalization of SAE ensembling.

The ensembled SAEs show  signficantly better performance even when compared with SAEs which have a equivalent number of features to the ensembled ones.

Authors not only look at better reconstruction loss, but also a wether these techniques improve training stability, whether these SAEs can be used to do concept detection and also to detect spurious correlation.

### Weaknesses
Even though the SAEs were compared to to equal number of features, they were not compared to the same amount of training compute/train time.

Both boosting and bagging are significantly slower to train, almost a order of magnitude larger training times for some of the model sizes.

Boosting seems very similar to matching pursuit SAES (From Flat to Hierarchical : Extracting Sparse
Representations with Matching Pursuit), if I'm understanding it correctly, but it is never mentioned in the paper

### Questions
The extended SAEs have the same number of features. What if you compared to SAEs trained same amount of compute/wall clock time? Table 3 only shows the case of the Base SAE and not of the extended SAE so it is hard to know how far way these are.

Although I can understand the motivation of looking at relative sparsity, this metric can probably be misleading. Doubling the expansion factor of an SAE does not entail doubling the number of features we allow to be active at any given time. Given that for boosting you get new  coeficients for each 'layer' the fact that relative sparsity goes does seems misleading. Would highly boosted SAEs not have a very high number of active features per token?
 
Can you justify diversity as a metric? A completly random SAE can have high diversity without it being usefull. The same can happen with an  SAE that has a lot of dead latents.

### Soundness
3

### Presentation
3

### Contribution
2
