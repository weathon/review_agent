# Look the Other Way: Designing 'Positive' Molecules with Negative Data via Task Arithmetic

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
The scarcity of molecules with desirable properties (i.e.,'positive' molecules) is an inherent bottleneck for generative molecule design. To sidestep such obstacle, here we propose molecular task arithmetic: training a model on diverse and abundant negative examples to learn 'property directions' — without accessing any positively labeled data — and moving models in the opposite property directions to generate positive molecules. When analyzed on 33 design experiments with distinct molecular entities (small molecules, proteins), model architectures, and scales, molecular task arithmetic generated more diverse and successful designs than models trained
on positive molecules in general. Moreover, we employed molecular task arithmetic in dual-objective and few-shot design tasks. We find that molecular task arithmetic can consistently increase the diversity of designs while maintaining desirable complex design properties, such as good docking scores to a protein. With its simplicity, data efficiency, and performance, molecular task arithmetic bears the potential to become the de-facto transfer learning strategy for de novo molecule design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an exploration of the application of task arithmetic in molecular generation tasks. Results shows that via task arithmetic, the success rate of generating positive molecules can be improved over merely pre-training and fine-tuning over positive molecules.

### Strengths
- This paper is an important initial exploration on using task arithmetic for positive molecule generation. The task setting is meaningful and may be a future way for molecule designing or optimization.
- The proposed method can beat pre-training and finetuning in generaing more clusters of successful molecules.

### Weaknesses
- The task setting is a bit over-simplified. The experimental setting is no positively LABELED molecules, instead of no positively molecules at all. Actually, a large amount of positive molecules exists in the pre-training data, according to Figs A1 and A2. This is also why in Fig. 2, even a merely pre-trained model (grey color) presents a good success rate. I would expect a setting where positive molecules never appeared in both pre-training and fine-tuning, i.e., the results of 'pre-training' should be 0 or close to 0, then the proposed method to be explored in this setting, so people can know if task arithmetic really works for 0-shot design or not.
- This paper is largely not self-contained. Many tables or figures mentioned in the main text turned out to be in the appendix, which harms the readability.
- The definition of successful "clusters" is unclear and not mentioned.

### Questions
- How do you define/find "successful clusters"?

### Soundness
2

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
4

### Summary
To train models to generate molecules with desirable properties, high-quality examples of molecules with such properties are often required. But such "positive" examples may be nearly non-existent in the general training population and extremely difficult to curate.  The authors propose molecular task arithmetic, a transfer learning technique to fine-tune neural networks with only "negative" samples. Rather than fine-tuning on positive examples, the technique uses abundant "negative" data to fine-tune an opposing model. Moving the pre-trained models' weights in the opposite direction of this opposing fine-tuned model, the authors hypothesize, will cause it to generate molecules with the desired properties.

### Strengths
1. The idea of going in the opposite direction from the "bad" model is very interesting and innovative. Being able to unlock potentially new behavior from entirely unseen domains could prove to be of huge significance and could open new avenues of research and application. The proposed solution is very elegant. 
2. The central premise of the lack of "positive" samples in scientific domains, especially drug design, is a significant problem. 
3. It is very interesting that the model obtains good scores on challenging tasks like improving docking score as the negative examples are not necessarily well defined in such a case.

### Weaknesses
1. "Current training strategies rely on rare, positive molecules." This is not true in general. This may be true for the case of fine-tuning approaches. But there is a long line of work in latent space and post-hoc optimization (such as Bayesian optimization, generative algorithms, or gradient-based steering) for generative design. So will this may be a strong candidate for a transfer learning strategy for molecule design models; the aforementioned techniques that do not require transfer learning or new data may be a more practical approach. 
2. According to figure 4, it seems like there is a significant decrease in validity as the $\lambda$ is increased, as a result, finding the optimal model with sufficiently high validity and success rate may be challenging to find, and casts doubt on how steerable these models may actually be.

### Questions
1. How is a randomized SMILES different from a regular SMILES representation of a molecule? Is it a result of data augmentation? 
2. Are there differences in performance depending on the model architecture or size of the model being used? As this is a weight update technique, I would be very curious as to how different models behave with this approach, if any trends are interesting
3. What is the difference between the number of successful design clusters and the number of design clusters in Figure 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper applies the concept of task arithmetic to molecule generation. Task arithmetic describes the process of linearly combining the weights of a pretrained foundation model with those of a finetuned model. This works surprisingly well in the domain of molecule generation and property optimization, where “positive”/desirable examples to fine tune on are exceptionally rare, but “negative” examples are abundant. The method employed in the paper involves finetuning on the negative examples and then taking the difference of the model weights, to lead the foundation model away from the space of negative molecules.

### Strengths
Generally, the writing is very clear and communicates the arguments and ideas without issue, even to a less experienced reader. The concept of task arithmetic is very elegant, and its application to molecular optimization feels almost intuitive because of the quality of the explanations. The experiment design and results are very extensive, which additionally strengthens the main point of the paper. The graphs/tables are clear and easy to read.

### Weaknesses
The lack of mention of chemical LMs, like ChemBERTa or ether0, takes away from the credibility of the article. Though the results are impressive and interesting, LSTMs lack much of the expressive power of LMs. Though task arithmetic appears to work in the case of LLMs on natural language, it’s not clear whether that will extrapolate to models working on the chemical domain, especially given the apparent challenge with this generation task. Additionally,

### Questions
In section 5.2, what is a “cluster”? It does not appear to be defined anywhere previously in the article.
Line 282: “task arithmetic can be a technique to maintain [...]”. Maybe replace it with “task arithmetic is a technique that can maintain [...]” or even “task arithmetic is a technique that maintains [...]”.
It’s interesting that randomized SMILES inputs consistently perform better than canonical SMILES. Though not critical, discussion on this could be informative, if not out of scope.
Line 472: “[...] that leverages the diverse and abundant negative molecules”. Maybe replace it with “[...] that leverages the diversity and abundance of negative molecules”.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a “look the other way” / molecular task arithmetic approach for de novo molecular design: instead of fine-tuning on scarce positive examples (molecules with desired properties), the model fine-tunes on abundant negative examples (molecules that fail the property), obtains a “negative task vector,” and then moves in the opposite direction in parameter space to generate candidate molecules that are more likely to satisfy the target property. The method is evaluated on a set of relatively simple property targets and a few docking targets, showing that (i) negative-only training can still produce valid, property-consistent molecules and (ii) in some settings it improves diversity compared to standard positive fine-tuning.

### Strengths
- Clever use of negative data. The key insight—most real medicinal chemistry data is overwhelmingly negative, so we should learn from what “doesn’t work” and then invert the direction—is both practical and underexplored. It directly speaks to the data imbalance that plagues many molecular design tasks.

- Empirically effective on some tasks. On the tested single-property and docking-based tasks, the method can recover a nontrivial number of “successful” and diverse solutions, sometimes outperforming standard fine-tuning in terms of successful clusters. This suggests the parameter-space editing idea is not just a theoretical curiosity but can actually steer generation.

### Weaknesses
- Property space is too easy / narrow. Most experiments are on relatively simple, monotonic, and well-behaved properties (e.g., logP high/low, TPSA high/low, counts of functional features). Even the docking objectives look like “flat-bottom” tasks—once you’re in a good-enough region, the signal stops being very discriminative. This makes it unclear whether the proposed method would still work on sharper, more structured, or conflicting objectives. The paper should include harder benchmarks such as those in GuacaMol (distribution-learning tasks, scaffold-aware objectives, isomer/chemotype control) or other community-accepted suites to demonstrate generality and push beyond easy scalar filters.

- No discussion of sample efficiency for molecular design. A main motivation of the approach is precisely that positive samples are scarce, yet the paper does not really quantify sample efficiency:

- Comparative baselines are underspecified. If the claim is “we can get good molecules without positives,” then the right comparison is not only vs. a naïve positive fine-tuning, but also vs. (i) property-conditioned generation trained on mixed labels, (ii) reinforcement-learning style optimization (REINVENT, PPO), and (iii) strong open benchmarks like GuacaMol’s goal-directed baselines. Right now, it’s unclear whether the gains come from the negative-task idea itself or from the fact that the tasks are relatively easy.

- Limited analysis of failure modes. Task arithmetic on model weights is a fairly aggressive operation; the paper should analyze when it fails: Does it hurt validity? Does it bias toward weird chemotypes? Does it collapse to trivial solutions when the negative set is heterogeneous? Right now, the story is too positive.

### Questions
- How many negative examples are needed before the “opposite-direction” becomes meaningful?

- How many positives (few-shot) do we actually save compared to a strong low-data baseline (e.g., adapter-style fine-tuning or conditional generation with learned property heads)?

- How does the method behave when the true positive set is not just small but diverse (multi-chemotype targets)?
Without a data–performance curve, it’s hard to assess whether this is genuinely more data-efficient or just a different way to consume data.

### Soundness
1

### Presentation
2

### Contribution
1
