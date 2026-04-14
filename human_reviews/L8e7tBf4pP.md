# DaWin: Training-free Dynamic Weight Interpolation for Robust Adaptation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Adapting a pre-trained foundation model on downstream tasks should ensure robustness against distribution shifts without the need to retrain the whole model. Although existing weight interpolation methods are simple yet effective, we argue their static nature limits downstream performance while achieving efficiency. In this work, we propose DaWin, a training-free dynamic weight interpolation method that leverages the entropy of individual models over each unlabeled test sample to assess model expertise, and compute per-sample interpolation coefficients dynamically. Unlike previous works that typically rely on additional training to learn such coefficients, our approach requires no training. Then, we propose a mixture modeling approach that greatly reduces inference overhead raised by dynamic interpolation. We validate DaWin on the large-scale visual recognition benchmarks, spanning 14 tasks across robust fine-tuning -- ImageNet and derived five distribution shift benchmarks -- and multi-task learning with eight classification tasks. Results demonstrate that DaWin achieves significant performance gain in considered settings, with minimal computational overhead. We further discuss DaWin's analytic behavior to explain its empirical success.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper titled "DaWin: Training-Free Dynamic Weight Interpolation for Robust Adaptation" introduces DaWin, a novel method for dynamically merging model weights without additional training. Traditional weight interpolation methods, such as WiSE-FT, use static coefficients for merging models, limiting their effectiveness when models exhibit different levels of expertise across samples. DaWin addresses this limitation by proposing sample-wise dynamic weight interpolation, utilizing entropy-based measures to adjust interpolation coefficients in real-time. DaWin achieves state-of-the-art performance on both robust fine-tuning tasks (ImageNet benchmarks) and multi-task learning (e.g., EuroSAT, MNIST) with minimal computational overhead.

### Strengths
1. Training-free approach: DaWin eliminates the need for additional model training, providing a plug-and-play solution.
2. Entropy-based dynamic interpolation: This novel use of entropy helps improve sample-wise performance across diverse tasks.
3. State-of-the-art results: DaWin outperforms WiSE-FT and other interpolation methods on ImageNet and multi-task benchmarks with minimal computational overhead.
4. Computational efficiency: The use of mixture modeling reduces the inference cost, making DaWin scalable to large datasets.

### Weaknesses
1. **Incremental novelty**: While the method provides refinements over static interpolation, it shares similarities with existing methods (e.g., AdaMerging), which reduces its overall originality.
2. Over-reliance on entropy: The assumption that entropy reliably measures model expertise across all scenarios is not thoroughly validated, and alternative metrics could be explored further.
3. Trade-offs not fully explored: The paper does not sufficiently address the performance trade-offs between per-sample interpolation and simpler methods like weight averaging.

### Questions
1. How well does DaWin generalize to tasks beyond classification benchmarks (e.g., NLP or reinforcement learning)?
2. Can the reliance on entropy as an expertise proxy limit DaWin's performance in certain scenarios, such as multi-modal data?
3. How does DaWin compare in terms of inference speed and energy consumption to AdaMerging and WiSE-FT?
4.What scenarios, if any, might lead to simpler methods outperforming DaWin, and how can this be systematically explored?

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
3

### Summary
This paper describes a method of doing sample based parameter merging in order to improve performance on OOD test samples. The main idea is to estimate the entropy of each model and then to use this to perform the merging step. As this is quite slow, they also describe a faster mixture-modelling technique. They first demonstrate with 'GT' data that if you had an oracle for the entropy you would do well. They then show that this oracle / estimated entropy is correlated and finally that they achieve superior results over similar methods.

### Strengths
The paper is quite interesting and presents their main idea clearly. I summarise the strengths below:
1. Clarity: the paper is clear and gets across the main message (can we use an estimate of entropy at the sample level in order to improve OOD performance).  I like the 'GT' set up as it clearly shows the potential improvements of such an approach and gets across the intuition of why this is useful.

2. Experiments: they have a range of results backing up their claims and showing that their method improves over the baselines. They further generalise their approach to other settings (e.g. dynamic classifier selection or ensemble selection)

3. The consider compute and performance and show how they are trading off compute (though maybe not as badly as others) for improved performance.

### Weaknesses
1. Clarity: There are a few parts of the paper that are lacking details or are confusing:
1a. in the intro a foundational model != ImageNet. People refer to foundational models primarily for large scale models trained on vast amounts of data (GPT,Gemini,Flamingo)
1b. It is unclear how precisely H(-) is computed. I appreciate this is probably obvious to the authors but it would be helpful to briefly mention.
1c. I am unconvinced by their linear mode connectivity explanation. My understanding is the point of the linear mode connectivity is that *along that line between model weights*, the error is bounded to some height. But this is *not* what's being checked -- it's being done for a single interpolated example.

### Questions
Please refer to questions in weaknesses but here are additional questions that are not weaknesses but I'm curious about:

1. Did they see any trends in where they obtained similar interpolation coefficients for samples? Maybe certain clusters of samples are always better explained by a given model or not. Are there conclusions one could draw about the test set from this ?

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
4

### Summary
This paper proposes a dynamic weight interpolation method DaWin. It merges pre-trained and fine-tuned models based on per-sample entropy without training. It further uses Beta mixture models to cluster interpolation coefficients to keep computation efficient. DaWin achieves considerable performance, offering a scalable solution for adaptable model merging.

### Strengths
1. DaWin requires no additional training, making it highly practical for real-world deployment.
2. The usage of Beta mixture models ensures adaptation scalability, reducing the inference-time overhead of dynamic interpolation.
3. DaWin shows its performance on both robust fine-tuning and multi-task learning problems.
4. DaWin does not require extensive hyperparameter tuning.

### Weaknesses
1. The model utilizes Beta Mixture Models for efficiency, but only complexity and wall-clock time results are provided. There are no reported MACs or FLOPs per sample, limiting insights into computational costs.  

2. Since the model performs per-sample interpolation, it requires three inferences per sample, increasing computational demands.  

3. The dynamic interpolation adjustment can be viewed as an incremental improvement over existing interpolation methods, rather than a novel breakthrough. The level of novelty remains unclear.  

4. While the paper presents a comprehensive experimental study, the explanations for the results are shallow. There is limited discussion on why DaWin performs well or poorly on specific datasets or distributions (e.g., SUN397 and Cars). Additionally, relationships among datasets are critical but remain under-explored.  

5. The entropy-based expert assumption is built on simple object concepts, implying that the datasets are sufficiently learnable through fine-tuning a CLIP model, where entropy should ideally align with correct predictions. However, in real applications, entropy can be misleading, resulting in overconfident but incorrect predictions. The model design does not address how to correct these wrong predictions. While the TrueTrue/TrueFalse breakdown in Figure 2 is insightful, it is not directly leveraged in the experiments, which limits the depth of the analysis.

### Questions
1. From my point of view, the dynamic weighting strategy per sample is another form of confidence-based selection, where the preference is determined solely by model confidence. Given this, what if we fuse the outputs instead of the model weights? Do the tasks (e.g., ImageNet - ImageNet-A) reveal any trends in the weighting of these models?

2. How can we determine whether the performance is correlated with the **distribution distances** between datasets?

3. Could the proposed method result in negative weight mixtures for some datasets? If so, why?

4. Is it possible that some checkpoints could **decrease** the final performance on specific datasets? Furthermore, do the semantic distances and relationships between classes from different datasets (and their corresponding checkpoints) affect the performance of one another?

5. For the evaluation of wall-clock time, is it tested per dataset? or per sample?

6. Please also review the drawbacks section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
A new method for robust fine-tuning, called "dynamic weight interpolation" (DaWin) is proposed. It is directly related to the well known model averaging method WiSE-FT. However, while WiSE-FT averages model parameters, using a fixed coefficient $\lambda$ that does not depend on the input data, DaWin proposes to make the coefficient data dependent, i.e., to use $\lambda(x)$. The optimal $\lambda(x)$ is computed, using a heuristic that is based on the entropy of the model output. DaWin is tested against many existing robust fine-tuning methods, yielding slightly better OOD performance compared to WiSE-FT.

### Strengths
In general, the paper written in a clear and understandable way. It provides novel experimental evidence, that averaging the weights of pre-trained and fine-tuned models to keep OOD performance high might not be fine grained enough. The proposed heuristic to select sample based averaging coefficients intuitively makes sense and is backed up by well designed experiments.
The performance and computational complexity of DaWin is compared to a sensible selection of existing methods.

### Weaknesses
I see two problems:

1) In section 2.2, the a pilot study is provided to show the necessity of sample-wise averaging coefficients $\lambda(x)$ for good out of distribution performance after fine-tuning. As I understood, the goal is to find the best possible OOD performance that we can achieve with an oracle $\lambda^*(x)$ for the sample-wise coefficients that actually knows the true labels for a given $x$.

The paper states, that such an oracle would be $\lambda^*(x) = p_{\theta_1}(y|x) / (p_{\theta_0}(y|x) + p_{\theta_1}(y|x))$, i.e., the probability whether data $(x,y)$ comes from the distribution $p_{\theta_1}(y|x)$. However, while this would make sense if we merge the output of the models (analogous to classifier selection), I cannot imagine that it is the seeked best oracle for $\lambda^*(x)$ if we perform weight averaging. To obtain this oracle, we would have to test for each $x$, if there exists a $\lambda^*(x)$, such that the model with parameters $\theta_{\lambda^*}$ predicts the correct $y$.

In the following sections, the paper makes some approximations to compute $\lambda^*(x)$ based on the entropy of the model output, which does not require the knowledge of $y$. The reader is lured into the impression, that the only performance gap to the oracle is caused by these approximations. However, I think this is not true, since the oracle itself might not be optimal. However, this issue is about presentation and could be fixed.

2) I think performance gains compared to WiSE-FT and other methods are often minimal. Although DaWin does not require fine-tuning, there is still a considerable memory overhead. To be able to perform per-sample averaging, we need to store both sets of weights. For large models, this might be a severe problem that is not worth the effort.

### Questions
* Could you please comment on the memory requirement of DaWin and how it compares to other methods, for both with and without mixture modeling?
* I am also not so sure about your provided costs for inference in Table 2. Could you point me where $H$ is defined and elaborate a bit the provided terms? Am I correct, that DaWin (without mixture modeling) requires 3 model evaluations (2 to compute the entropy for the individual models for a given $x$ and 1 for the averaged model)?
* In the experiments, DaWin sacrifices ID performance for better OOD performance compared to WiSE-FT. Could you elaborate in which application scenario I would want to choose DaWin over WiSE-FT. I would think in practice good ID performance still wins over good OOD performance if you cannot have both.

### Soundness
3

### Presentation
4

### Contribution
3
