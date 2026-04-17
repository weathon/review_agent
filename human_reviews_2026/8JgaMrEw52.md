# Fine-grained Analysis of Brain-LLM Alignment through Input Attribution

- Decision: Reject
- Scores: 2, 4, 2, 6, 6

## Abstract
Understanding the alignment between large language models (LLMs) and human brain activity can reveal computational principles underlying language processing. This work describes a pipeline to apply attribution methods to the brain-LLM alignment setting to identify the specific words most important for this alignment. As a case study, we leverage it to study a contentious research question about brain-LLM alignment: the relationship between brain alignment (BA) and next-word prediction (NWP). Across two naturalistic fMRI datasets, we find that BA and NWP rely on largely distinct word subsets: NWP exhibits recency and primacy biases with a focus on syntax, while BA prioritizes semantic and discourse-level information with a more targeted recency effect. This work advances our understanding of how LLMs relate to human language processing and highlights differences in feature reliance between BA and NWP. Beyond this study, our attribution method can be broadly applied to explore the cognitive relevance of model predictions in diverse language processing tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a unified input attribution framework to compare how large language models (LLMs) align with human brain activity (Brain Alignment, BA) versus next-word prediction (NWP). Using gradient-based word-level attributions, the authors show that BA and NWP rely on distinct linguistic features: NWP emphasizes syntactic and short-range cues, while BA depends more on semantic and discourse information. The study provides a fine-grained, interpretable view of LLM-brain alignment and highlights differences between predictive and comprehension-related representations.

### Strengths
1. Novel use of quantitative attribution metrics.

The paper introduces and adapts two interpretable metrics, Intersection over Union (IoU) and Center of Mass (CoM), to quantify the overlap and positional tendencies of important words in Brain Alignment (BA) and Next-Word Prediction (NWP) tasks. This design enables a clear, measurable comparison of attribution patterns across tasks and represents a well-motivated methodological contribution.

2. Inclusion of diverse model architectures.

The study incorporates state-space models (SSMs) alongside traditional Transformer architectures in its analysis. This cross-architecture comparison is relatively new in the brain–language modeling literature and broadens the generality of the findings, demonstrating that the observed attribution trends are consistent across distinct model families.

### Weaknesses
1. **Limited conceptual novelty.**

The main weakness of the paper lies in the limited originality of its findings. The preference of brain alignment (BA) tasks for semantic information has been recognized in neuroscience, and the syntactic bias of next-word prediction (NWP) has also been observed in NLP studies. While the attribution-based comparison provides finer detail, it largely confirms existing conclusions rather than extending them conceptually.

2. **Incompleteness of the CoM metric.**

The Center-of-Mass (CoM) index captures only the mean positional tendency of attributions but ignores their degree of dispersion. Visual inspection suggests that in some results, word attributions are highly scattered or concentrated. Adding a complementary measure (e.g., variance or entropy of attribution positions) would provide a more comprehensive characterization of positional patterns.

3. **Gradient-based attribution limitations.**

The study relies solely on gradient-based attribution (Gradient $\times$ Input and Integrated Gradients), which are inherently sensitive to local nonlinearities and model saturation. Although Integrated Gradients mitigates these issues, it does not fully resolve them. Incorporating additional attribution approaches, such as perturbation-based or ensemble methods (e.g., SmoothGrad), would improve robustness and better justify the use of the term "attribution".

### Questions
1. In several figures (e.g., Figure 5, Mamba), the NWP task shows a clear rise in attribution for tokens beyond position 600, yet the reported CoM value remains below 200. Does this indicate that the CoM metric is insensitive to high but localized attribution peaks in distant contexts?

2. The current analysis focuses on comparing attribution distributions but does not address the deeper question that the field increasingly finds important—how strongly BA and NWP are related, and what mechanisms underlie their correlation. Since the authors already measure both tasks, explicitly quantifying and explaining this relationship would make the contribution more meaningful.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a new end-to-end attribution framework for studying how large language models align with human brain activity. The authors compute word-level attributions for both brain alignment (BA) (predicting fMRI signals from model embeddings) and next word prediction (NWP), then compare which parts of the input each task depends on.

They introduce 2 complementary measures: intersection-over-union (IoU) and Center-of-Mass (CoM) to quantify overlap and positional bias of attributions. They use the Harry Potter and Moth Radio Hour fMRI datasets and 5 pretrained LLMs between 1B and 2B (Transformers, State-Space Model and Hybrid). They find that:

- BA and NWP rely on largely distinct sets of words, with little overlap for the top-attributed inputs.
- NWP is dominated by syntactic and edge-biased cues (recency and primacy effects).
- BA relies more on semantic and discourse-level information with broader and more distributed recency effects.
- These trends generalize across the 2 datasets and architectures.

The framework provides a quantitative bridge between representational neuroscience and interpretability in LLMs.

### Strengths
- Methodological rigor: The mathematical grounding of the framework is solid. The use of IoU and CoM as geometric measures of overlap and temporal focus is elegant and interpretable and the end-to-end gradient-based formulation is technically sound. 

- Semantic intepretation of BA: The result that BA emphasizes semantic and discourse cues more than NWP is both intuitive and empirically grounded. It strengthens the argument that the brain's predictive machinery operates at a higher representational level than next-token statistics.

- Potential as a reusable framework: the attribution pipeline could become a standard tool for probing cognitive relevance in multimodal brain-AI comparisons.

The work is carefully executed, but the framework is so promising that it leaves the reader wanting more. Particularly a deeper analysis of the architectural differences and a more thorough exploration of the attribution results.

### Weaknesses
- Underdeveloped discussion of model classes: the transformer vs. SSM comparison is one of the most exciting parts, yet the discussion (a few lines around Fig. 5) is minimal. Why does Mamba behave differently? What architectural biases could explain the observed recency differences?

- Attribution: the paper claims to present an end-to-end framework for attribution, but the application is restricted to three categories: semantic, syntactic, and discourse (the latter is not clearly defined in the paper). Given the setup, one would expect masking or perturbation ablations to confirm causal influence. These are absent, except for a brief mention in the appendix.

- Dataset dependence: a major limitation is that the findings are largely based on the Harry Potter reading fMRI dataset, which is extremely narrow in style. The brief extension to The Moth Radio Hour is not enough to claim generality. It would have been particularly interesting to see whether the same patterns hold for non-narrative or multilingual data.

- Clarity and presentation: using the proportion of important words by distance as the core positional plot (in several versions, especially in the appendix) is not necessarily the most effective way to convey the results. Most figures (except Figure 2) are difficult to interpret, and simpler visualizations could make the findings clearer. The paper could also be shortened substantially (currently 34 pages) without losing substance.

I would be happy to revisit my rating should the authors address the points raised in this review.

### Questions
1. How exactly do transformers and SSMs differ in their attribution profiles? 

2. Which model achieved the highest brain alignment overall and which the lowest? What architectural or representational properties might explain this ranking ?

2.bis. What happened to Llama3.2-1B in Figure 2? Is there an explanation vs. the other models?

3. If BA is more semantic, can this be causally demonstrated by masking high semantic weight words?

4. Why were only 3 categories for attribution grouping? Harry Potter annotations allow richer semantic subtypes. Could the analysis be more granular ?

5. How stable are the attribution patterns across subjects and runs?

6. Can you think of a way this framework handle multimodal (audio, vision) stimuli where tokenization is less discrete ?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper evaluates the alignment between LLMs and human brains (primarily Harry Potter fMRI dataset) with respect to word features, for which the authors use a gradient-based input attribution method. 
Specifically, input properties for brain alignment are compared with those for next-word prediction.
The major claim is that brain alignment and next-word prediction rely on distinct words: syntactic vs semantic information.

### Strengths
Claims are tested on several models and two fMRI datasets (primarily Harry Potter; Moth Radio Hour in Appendix).

Code available (currently in supplemental zip; GitHub promised after review).

### Weaknesses
### 1. Contribution
I am a bit lost with what to actually take from this paper. The methods themselves have previously been developed (brain encoding models as well as input attribution), and I'm not really sure why the claim of distinct word sets is important. I'm not really convinced that different words being deemed "important" (i.e., in a very fine-grain perspective) maps onto a fundamental difference in what NWP optimizes for and what brains optimize for. We know from the broad success of LLMs that NWP yields semantically very meaningful representations and capabilities, how does that reconcile with the claims here?

### 2. Causal relevance of word importance
The input attribution method estimates the importance of input words, which is causally quantified in Appendix C (good!). But I cannot tell from this analysis if the top% selected words are actually more important than e.g. randomly selected words.

I will note here that I find it a bit much to claim the approach the "introduc[tion of] a fine-grained input attribution method" (L013) for importing the Captum library and using a differentiable linear layer for projecting model features to brain activity (again in the conclusion L477 "We introduce the first end-to-end attribution framework for brain-LLM alignment"). Not every paper needs to claim a new method.



### Minor: 
* It is at times difficult to follow the text with the amount of abbreviations used throughout (BA, NWP, TR, HP, IoU, ...).
* there's some inconsistency with the NWP<>BA studies between lines 086 and 051 where the sets of references are different. Hosseini et al. 2024 also seems relevant.
* After going through the paper several times (including the appendices), I just cannot find how NWP is actually measured. What dataset is used here?

### Questions
Why is the importance of words important?

Please addresses weaknesses 1 and 2 above.

### Soundness
3

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
This paper presents a novel end-to-end attribution framework aimed at investigating the alignment between Large Language Models (LLMs) and human brain activity during language processing—referred to as Brain Alignment (BA). The framework is further applied to the debated question of the relationship between BA and Next Word Prediction (NWP). Overall, the work offers a valuable analytical tool and provides fresh insights into the types of information that BA and NWP respectively exploit.

### Strengths
1. The paper puts forward an "end-to-end" attribution framework that enables direct interpretation of the brain prediction model, advancing beyond the traditional representational correlation analyses found in previous research.
2. The authors conduct comprehensive cross-validation, examining various LLM architectures, two distinct fMRI datasets, and multiple evaluation metrics, which enhances the robustness of their results.

### Weaknesses
1.  The findings may primarily reflect what a linear decoder can extract from the LLM representations.  If a more powerful, non-linear encoder, such as a multilayer perceptron (MLP), were employed, the attribution outcomes could be substantially different.  The authors attribute their findings to the nature of the BA task, overlooking the strong inductive bias introduced by their chosen encoding model.
2.  Although Integrated Gradients (IG) are used for verification, this validation step is limited: IG is applied to only two representative models and is mainly used for feature and positional analysis.  However, the paper’s key claims—such as the Intersection over Union (IoU) analysis and attribution spread—are based entirely on Gradient x Input (GXI), making the justification potentially inadequate.
3.  The core linguistic feature analysis is performed solely on the HP dataset, which consists of data from just eight subjects reading one chapter.  Drawing broad conclusions—such as NWP favoring syntax and BA favoring semantics—based on this limited dataset is questionable in terms of generalizability.
4.  The study observes that Llama32-1B demonstrates a unique "oscillatory" attribution pattern on the HP dataset, but this pattern disappears in the MRH dataset and in short text contexts (80 words).  The authors suggest this may be "stimulus and context-dependent,” but this notable phenomenon is not thoroughly explained.

### Questions
see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript presents a gradient-based method for computing token-level importance for brain alignment (BA) using a frozen LLM and gradients of the BA loss. Findings are compared to attribution analyses performed for LLM next word prediction. The relative similarities and differences between these attributions are studied.

### Strengths
The manuscript presents a method for computing token-level importance for brain alignment (BA) using a frozen LLM and gradients of the BA loss. The architecture is sound; the implementation used to backpropagate the BA objective is reasonable; the comparison to next-word prediction (NWP) attributions is rigorous.

### Weaknesses
Regarding masking, I think the current masking procedure is too strong, perhaps to the point that invalidates the procedure and conclusions drawn: it replaces the top words with random words and shows this removes brain prediction, but in doing so,  it also produces an input string that is not part of the training distribution, so the drop in performance cannot be interpreted as being solely due to removal of related information, but due to the introduction of what is nonsense content with respect to the LLM or encoder.  To interpret masking properly, a more sensitive procedure should be used, which removes the target-word information, but keeps the input phrase consistent with statistical distribution the model was train on.

A second point concerns within-method reliability. Although the LLM is frozen, both the BA encoder and the projection head (used to pass gradients back to tokens) can vary across splits and seeds. I suggest reporting seed-wise stability of BA attributions (e.g., seeded refits analyzed using within-BA IoU curves computed across seeds of *same* model). This is an internal ceiling against which BA-to-NWP intersections can be evaluated.

Finally, typical BA performance is modest (e.g., Pearson’s r ≈ 0.06 for predicted vs. observed activity). This corresponds to variance explained (R^2 = 0.0036), which is about 0.36%, i.e., **≈99.64%** of variance in brain activity  is unexplained. It is of course valid to compute gradients of a large loss and, as shown here, identify words whose masking eliminates even this small correlation . Still, attribution is most informative when model performance exceeds a practical threshold, or at least a threshold that is meaningful given prior work. To this end, I recommend reporting noise ceilings (e.g., from inter-subject reliability; often in range of 0.2-0.4 in other studies) and normalizing against them so the values in appendix G can be better understood.

### Questions
Suggestion: The conclusion that NWP and BA rely on different information is supported by the attribution content. The authors could also easily run a cross-task masking, where you mask BA-ranked words and measure NWP degradation, and vice versa. At present, masking is reported within task only.

### Soundness
3

### Presentation
4

### Contribution
3
