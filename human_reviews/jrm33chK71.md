# Few and Fewer: Learning Better from Few Examples Using Fewer Base Classes

- Decision: Reject
- Scores: 6, 5, 6, 5, 6

## Abstract
When training data is scarce, it is common to make use of a feature extractor that has been pre-trained on a large “base” dataset, either by fine-tuning its parameters on the “target” dataset or by directly adopting its representation as features for a simple classifier. Fine-tuning is ineffective for few-shot learning, since the target dataset contains only a handful of examples. However, directly adopting the features without fine-tuning relies on the distribution of the base dataset being similar enough to that of the target dataset in order to achieve separability and generalization. This paper investigates whether better features for the target dataset can be obtained by training on fewer base classes, in an effort to bring the distribution of the base dataset closer to that of the target dataset. We consider cross-domain few-shot image classification in eight different domains from Meta-Dataset and entertain multiple real-world settings (domain-informed, task-informed and uninformed) where progressively less detail is known about the target task. To our knowledge, this is the first demonstration that fine-tuning on a subset of carefully selected base classes can significantly improve few-shot learning. Our contributions are simple and intuitive methods that can be implemented in any few-shot solution. We also give insights into the conditions in which these solutions are likely to provide a boost in accuracy. We release the code to reproduce all experiments from this paper on GitHub. https://anonymous.4open.science/r/Few-and-Fewer-C978

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors raise and investigate the idea that fine-tuning a pretrained model on a subset of the (base) classes it has originally been trained on might yield beneficial results by potentially reducing the domain gap between train and target distributions in few-shot settings. The paper presents approaches for three different ‘levels’ of available target information, validated by experiments on several datasets and a variety of different heuristics for the case of least available information (aka uninformed setting). The authors demonstrate that a careful selection can indeed improve results across most benchmarks, but equally point out and discuss potential challenges when domain gaps are relatively big.

### Strengths
### Originality & Significance: 
-  While the underlying idea of using a ‘better fitting’ subset of classes seems quite straight-forward and intuitively makes sense to reduce potential domain gaps, I am not aware of any such detailed study of this aspect;    
&rarr; To me, the authors provide insights that are largely orthogonal to most recent developments in the few-shot area, honestly discuss their insights and provide open questions that could be tackled by the community in the future – therefore satisfying both ‘originality’ and ‘significance’.

### Quality: 
- Realization and thorough investigation of a straight-forward and simple but very neat idea in a solid manner, using three different levels of available information (w.r.t target task/domain)
- Experiments conducted with a good selection of comparative methods to gauge performance improvements: representative baselines in the informed settings, as well as different heuristics in the uninformed setting -- including random selection and oracle upper bound
- The authors honestly discuss both their ‘achievements’ as well as potential challenges that exist with their approach, offering insights and intuitions what could potentially be tackled in future work;
### Clarity:
- The paper is very well written and easy to read and follow; The topic is well motivated with clear presentation throughout
- The authors do a great job of clearly stating the objectives at several stages throughout the paper, combined with the underlying intuitions as well as on-point discussion of the relevant background and results;

### Weaknesses
**TLDR;** I do not see any severe ‘prohibitive’ weaknesses in this work but have some concerns listed in the following; If the authors can address my questions & concerns, I’m happy to further increase my score and support (full) acceptance.

**Pre-training motivation**:
- The authors state that pre-training on the base classes “may even have a deleterious effect if the domain gap is too large” – and hence investigate if *one can reduce the domain gap by fine-tuning on a subset*:  
&rarr;  While I do understand that this is ‘merely’ used as motivation, I somewhat doubt that in cases where the domain gap is so large that it (significantly) harms transfer learning, fine-tuning on a subset of these classes would help? (i.e. the presented approach) – some comments/insights regarding this would be helpful; 

- Along the same lines: Given this motivation, wouldn’t then a reasonable baseline to compare to be the network ‘just’ trained on the subset of classes from scratch? In other words, how important is the pre-training on all the base classes? (as stated, it might even be harmful?)

**Slightly limited scope**:
- The presented results are somewhat constrained to the authors’ own setup – i.e. one baseline architecture that is used and then improved; However, the authors do not provide any indication as to what ‘sota’ methods that use the same backbone currently achieve on the chosen datasets;   
&rarr;  *Note*: I do in no way expect the authors to ‘beat’ any sota method as this paper is mainly about providing insights, but it would be interesting for the reader to know how far the ‘plain’ baseline and the ‘improved’ one are from top-performing methods; Some insight whether this sub-class finetuning setup might help in other methods as well (e.g. during meta-finetuning) would also be helpful to further support the paper’s findings.

**Missing ablation**:
- The authors note that they choose the top 50 classes (AA) in the informed settings to create the subsets, mainly to keep ‘similar size of subsets’ to the uninformed settings;   
&rarr; How important is this selection to achieve ‘good’ results? (robustness) – Some ablation would be helpful here;  
&rarr; It feels like this might be highly dependent on the composition of the dataset (base classes)?  
&rarr; I suspect there might be a more ‘generalizable’, dynamic and potentially better justifiable way to select the classes than a mere “top x” constant, e.g. certain % of the total mass / classes that cumulatively achieve certain threshold in the softmax; or even just a threshold on the ‘minimal contribution within the softmax’.

### Questions
Please see the "weaknesses" section for ‘main’ concerns;   
The following are mostly questions to gain some further intuition & suggestions for improving the manuscript:

- Judging by the result presented in Fig 1, the random subsets seem to be surprisingly better when using AA, FIM and RH – do the authors have any insight/suspicion as to why this might be the case? 
- Fig 8 (Appendix) indicates that the Semantic features generally provide better ‘upper bounds’ as shown by the oracle; Do the authors have some intuition whether that’s due to the ‘more powerful’ underlying CLIP training, or whether semantics (text) generally provide richer features for such selection? (if one has to be chosen)

**Comments/Suggestions**:
- Sec 4.2: The authors state that “MCS […] consistently led to a positive impact”, leading to the “ability to deploy such solutions in applications where strong constraints apply in terms of computation and/or latency” – However, doesn’t MCS actually have quite high compute requirements? (A fact that is also stated in the limitations and therefore somewhat contradictory) 
- Sec 4.1: Explanation and/or reference for the meaning/significance of the silhouette score would be helpful to the reader (can be in appendix)
- Sec 4.2: “SNR […] yields *optimal* boost” -> potentially change wording, as “optimial” to me would mean it achieves the oracle performance 

**Typos**:
- Sec 2 Terminology: “[…] techniques could be extended to the transductive setting is possible” -> Remove “is possible” 
- Sec 2 Leightweight: second last line misses a ‘period’ after the meta-learning references .. Requeima et al., 2019) “.” 
- Sec 4.2: “Figure 19” -> likely “Figure 1”?
- Sec 4.2 last line: “This experiments” -> plural/singular mixup

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the few-shot transfering by fine-tuning on a subset of base classes, proposes simple strategies to select base classes under multiple few-shot scenarios. Better performances are achieved compared to the baseline without fine-tuning.

### Strengths
The paper provides a thorough evaluation of various downstream tasks under multiple setups. 

The paper is well-written.

### Weaknesses
The paper, while presenting a method for selecting base classes and achieving better performance with fine-tuning in few-shot scenarios, lacks a strong sense of novelty. It would be beneficial if the authors could establish the optimality of their method, specifically by showing that it can identify the best 50 classes from a pool of 1000 ImageNet classes. Alternatively, if reaching the optimum is unfeasible, the paper should strive to provide approximate solutions that approach the upper bound.

Furthermore, while the paper does show improvements over a baseline without fine-tuning, it would be more insightful if the upper bound, representing the optimal 50 classes for each task, could be explored and discussed, if practical. This would provide a clearer perspective on the significance of the achieved results.

### Questions
What is the optimal solution? And how to approach it?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
- This paper examines the effectiveness of using a pre-trained feature extractor on a smaller, related subset of data to improve few-shot learning, where traditional fine-tuning on very small target datasets is often ineffective.

- The study explores different few-shot learning scenarios across eight domains, showing that selecting a subset of base classes closer to the target dataset's distribution can enhance performance.

- The authors propose simple, intuitive methods for few-shot classification improvement, providing insights into when these methods are most effective and releasing their experimental code on GitHub for reproducibility.

### Strengths
- The paper explores an approach where a pre-trained model, also referred to as a "base model" or "feature extractor," is fine-tuned using a carefully selected subset of the classes from its original training dataset—the "base dataset". This selection process involves choosing classes that are most relevant to the task the model will perform after fine-tuning. The selected classes should be diverse enough to prevent the fine-tuning process from causing the model to overfit to a small, non-representative sample of data. At the same time, it narrows down the scope of learning to what's most pertinent for the target application.

### Weaknesses
- There is no investigation into exactly why or how fine-tuning on fewer classes helps. The theoretical understanding is limited.

- The simplicity of the NCM classifier limits what conclusions can be drawn about the quality of representations. Testing on more complex classifiers would strengthen the results.

- Clustering classes for the static library has no guarantee of generating coherent subgroups. Better ways to determine class groupings could be developed rather than simple hierarchical clustering.

### Questions
- The class centroids used for NCM classification could be skewed by the fine-tuning. Could improvements just come from this distortion rather than better representations?

- Can the authors provide a rigorous statistical or empirical rationale for selecting $M=50$ as the fixed number of base classes in the Average Activations (AA) and Unbalanced Optimal Transport (UOT) selection strategies? How does this choice influence the balance between the breadth of class representation and the manageability of the subset size, particularly in relation to the diversity of the domain examples $D$ and the overall size of the base class set $\mathcal{C}$? Additionally, what impact might this have on the representational capacity of the fine-tuned feature extractor, especially when considering domains with a significantly higher or lower intrinsic class cardinality?

- How do the proposed heuristics for selecting specialist feature extractors address the issue of distributional shifts between the labeled support set and the unseen query set in few-shot learning tasks? Considering that these heuristics (SSA, SSC, LOO, SNR, RKM, MCS, FIM, AA) rely on the assumption that the support set is a representative sample of the task's data distribution, isn't there a significant risk that the heuristics will fail to predict feature extractor performance accurately in the presence of such shifts? 

- How can the method be robustified to account for potential discrepancies between the support and query distributions, which are common in real-world scenarios?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper empirically investigate how to learn better features for the target dataset by training on fewer base classes. Authors propose several heuristics for selecting a feature extractor. Extensive experiments in eight domains demonstrate the effectiveness of the proposed heuristics.

### Strengths
1. The writing is clear and easy to follow.

2. Compared to the baselines, there are consistent improvements, especially with domain-informed settings.

### Weaknesses
1. In general, authors investigate a collection of heuristics to select subsets for training. While some of these heuristics are effective in certain domains, it is difficult to draw a concise conclusion (e.g., Figure 1). It is more like empirical analysis than a novel approach.

2. In Table 1, the performance improvements seem to dependent on the similarities between the classes of target datasets and those of base classes (e.g., Aircraft and Traffic Signs). This may limit the application of the proposed heuristics.

### Questions
See the weakness

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates few-shot learning. They propose a novel pipeline that fine-tunes a pre-trained model on a subset of training examples ("base classes"), which are similar to the testing domain, to emphasize these parts and benefit the testing performance. They investigate three settings of tasks, including informed and un-informed, and design specific methods for them. The experiment results show that the proposed pipeline significantly improves the results.

### Strengths
- The proposed idea of emphasizing the most relevant subset as the testing domain in the pre-training dataset and the corresponding pipeline is novel.
- The proposed method is simple and effective. The authors cover three reasonable settings and provide reasonable methods for them.
- The experiment results show that the improvement is significant.
- The idea of emphasizing a subset of the pre-training dataset is novel and instructive direction in meta-learning research.

### Weaknesses
- Some necessary contents, including Alg. 1 and details of heuristics, are put in the appendix. This makes the audience unable to see a rigid, complete version of their method from a main paper, raising concern about page limit circumvention.
- In Sec. 3.2 about informed settings, a method is directly put there without any analysis, reasons, and details. I would like the authors to provide the motivation and some details behind these methods: Why is AA designed in this form? What is the implementation of $g$, fine-tuning on a pre-trained, fixed $h$; using a non-learning method on $g$; or co-train $h$ and $g$ on the pre-trained dataset? Is the same $g$ used in the latter testing phase? Does the scale of $g$ (as logits) matter in selecting $\mathcal{X}$?
- In Sec. 3.3 about uninformed settings, the authors introduce the method just by listing and citing a number of related works. So, I would like to know what the authors contribute to this part, or whether they are just summarizing or aggregating existing methods.
- All the tasks are image classifications. I wonder if other tasks, other than image classification, can be applied with the proposed pipeline. I would like to see some simple yet representative results on any other tasks. If the proposed method can be applied not only in image classification but also in many other tasks, the contribution of this paper can be even higher.
- (Minor) In Alg. 1, the set of examples is defined as $\mathcal{X}$, but the pseudo code uses $X$. And, line 2 is actually not required since operation 3 "selecting top $M$" already requires a top-$M$ partition.

### Questions
In "Weaknesses".

If the authors could address some of my concerns and provide results on some tasks other than image classification, I will raise the score.

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent
