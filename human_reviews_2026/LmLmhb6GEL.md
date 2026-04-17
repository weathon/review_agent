# Causality ≠ Invariance: Function and Concept Vectors in LLMs

- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
Do large language models (LLMs) represent concepts abstractly, i.e., independent of input format? We revisit Function Vectors (FVs), compact representations of in-context learning (ICL) tasks that causally drive task performance. Across multiple LLMs, we show that FVs are not fully invariant: FVs are nearly orthogonal when extracted from different input formats (e.g., open-ended vs. multiple-choice), even if both target the same concept. We identify Concept Vectors (CVs), which carry more stable concept representations. Like FVs, CVs are composed of attention head outputs; however, unlike FVs, the constituent heads are selected using Representational Similarity Analysis (RSA) based on whether they encode concepts consistently across input formats. While these heads emerge in similar layers to FV-related heads, the two sets are largely distinct, suggesting different underlying mechanisms. Steering experiments reveal that FVs excel in-distribution, when extraction and application formats match (e.g., both open-ended in English), while CVs generalize better out-of-distribution across both question types (open-ended vs. multiple-choice) and languages. Our results show that LLMs do contain abstract concept representations, but these differ from those that drive ICL performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces an RSA-based approach to identify *Concept Vector Heads*, attention heads in LMs that capture an abstract representation of the task. The authors compare Concept Heads with Function Vector heads identified in [Todd et al, 2023](https://arxiv.org/pdf/2310.15213) and show that Concept Heads better capture the abstract task representation whereas Function vector heads tend to be more susceptible to different input formats.

### Strengths
A simple idea presented nicely.

### Weaknesses
This paper treats MCQ, more generally selecting from a list of options meeting some criteria, as trivial and treats it as a artifact of the input format. I am not sure if that is a valid assumption, See more in Question 1.

### Questions
1. The authors treat the MCQ format as trivial and an artifact of the input format. However, I think that MCQ or more generally selecting from a list of options meeting some criteria itself is a non-trivial task that may trigger different reasoning strategies in the LM, which are investigated in [Wiegreffe et al, 2024](https://arxiv.org/abs/2407.15018) and [Tulchinskii et al, 2024](https://arxiv.org/abs/2410.02343). Could the authors comment on this?

   1.1. When Function Vector heads are being used to capture the task representation in the MCQ format you are actually asking them to capture two things: **(a)** the abstract task representation (e.g., antonym, singular-plural, etc) and **(b)** the MCQ-formatting task (select a/b/c/d). The results presented in the paper suggest that FV heads are capturning more of (b) than (a). However, I wouldn't assume this to be a failure of FV heads as they are still capturing *a* task representation, just not the one you want them to capture. And the CV heads were identified *intentionally* to capture (a) better than (b), unlike FV heads which do not get any such signals. So, of course the CV heads will perform better when evaluated on (a). This is not a fair comparison in my opinion.

2. The authors reported probability difference as a performance metric for steerability. In my opinion, it makes more sense to report the accuracy (i.e., can CV or FV *cause* the LM to output the target answer). Although this is a harder metric, I think accuracy captures more the causal role of such abstract representations in LMs computation and was also reported in the FV paper [Todd et al, 2023](https://arxiv.org/pdf/2310.15213).
   
    2.1. In Figure 7, the scales in the y-axis are not aligned, neither vertically nor horizontally. The authors should at least consider aligning the y-axis scales vertically to make it easier to compare the trends across CV and FV.


### Typos and Minor Suggestions
* Replace "Open-ended" with "Open-ended ICL". Or, at least clarify when you first introduce the term.
* Line 068: *"... FVs extracted from different input formats (open-ended vs. multiple-choice) are nearly orthogonal (cosine similarity = **0.9**)"*. I think there is a typo here. Cosine similarity is a measure of alignment, so 0.9 indicates that they are very aligned, not orthogonal. Your Figure 4 also support that this is a typo.

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates whether Function Vectors (FVs) are input-invariant: whether the performance with FVs vary across different input formats. They find that across different inputs, the representations are unstable. They instead introduce Concept Vectors (CVs) which are optimized via Representational Similarity Analysis (RSA). Their results show that CVs perform better out-of-domain than FVs do and that CVs and FVs are distinctly different mechanisms. Though the general performance of CVs are lower, the analysis is done well and the motivation for CVs make them a compelling source of future investigation in interpretability.

### Strengths
1. The figures in the paper are well made and looks very good.
2. The takeaways are clear and marked in each figure, which makes it very easy to read.
3. Overall, the paper from start to end is also simple to read.
4. The question is answered with a simple approach of RSA, and the authors do extensive analysis over in-domain and out-of-domain comparisons with both FVs and CVs.

### Weaknesses
1. I'd like to see the experiments with more prompts than just open-ended types and MC -- for instance, adding errors to the input prompt or small punctuation differences, if the claim is that CVs are input invariant. It's already good that there are experiments over multiple choice format too, but it would also be helpful to extend this towards formats that are "incorrect".
2. The analysis is done well to contrast in what situations FVs or CVs might be preferable. But the results also do leave the reader wondering why the heads chosen for CVs are better OOD than FVs and why CV performance may be lower than FV performance.

### Questions
1. Fig4, do you have some intuition about why the performance of CVs decrease as the number of K heads increases, in the "Concept" category?

### Soundness
4

### Presentation
4

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
This paper investigates whether LLMs contain representations of concepts that are robust to prompt formatting. It revisits function vectors [Todd et al.], which are implicit representations of functions constructed from attention heads. Using RSA, they find that function vector representations in MCQ settings differ from those in open-ended settings. They propose Concept Vectors, which can steer the model’s output to be invariant to the source prompt format (producing the same output even under language or structure changes).

The main contributions seem to be twofold: (1) a critique that function vectors are not invariant to prompt format, and (2) that there exist attention heads that produce concept representations that are invariant to prompt format which are separate from function vector heads.

### Strengths
- The experiments were clearly explained, and figures communicate key results nicely.
- The idea that the prompt form and underlying concept being tested can be separated is interesting and deserves further study. 
- Robustness of methods to prompting styles/rephrasings is an important problem that others would be interested in learning about
- The paper acknowledges existing limitations of the study (e.g., concept vectors don’t steer as strongly in-distribution)

### Weaknesses
- The motivation of the paper (do LLMs represent concepts abstractly across surface forms?) currently feels disconnected from the experiments and results of the paper. I think it comes from the fact that this paper implicitly takes the view that a “concept” is a relation or function between two entities. Because “concept” is an overloaded term, it may be worthwhile to include an explanation/discussion of why this definition of “concept” was chosen, since it seems central to the motivation of the paper. For example, another common framing of “concept” is that of multi-token entities [Meng et al, Nanda et al], but this is very different from your view.

- The tasks used for evaluation in the paper are very simple, and because of this, the contributions appear to be mostly conceptual. This is fine by itself, though choosing stronger baselines (e.g. prompting, fine-tuning) and more complex tasks to evaluate the proposed concept vectors method would certainly strengthen the paper in terms of methodological contributions. 

- The steering results in section 3.1 are somewhat mixed. In-distribution, concept vectors steer behavior somewhat poorly compared to function vectors (there are no other baselines presented). And while concept vectors do indeed seem to be invariant to the prompt format and steer towards the same answer in English under OOD formats, it’s unclear to me whether this is actually desirable behavior. Can you explain your intuition here as to why we'd want the English answer? Isn’t the idea behind in-context learning that the model should adapt based on the input? In this case, the equivariant behavior of the function vector (i.e. varies with changes in the source prompt) seems maybe more reasonable/desirable in some cases (e.g. if our goal is trying to understand ICL). Or even a composition of the two tasks (e.g. the French antonym). It seems like there is adequate evidence presented to claim that function vectors (FVs) are not exactly "invariant" to prompt format, but a discussion of “invariance” vs “equivariance” might be helpful to include because the behavior of FVs in the OOD setting seems somewhat akin to “equivariance” based on the description/examples in the paper you've provided. 

___
- Todd et al. [Function Vectors in Large Language Models](https://openreview.net/forum?id=AwyxtyMwaG)

- Meng et al. [Locating and Editing Factual Associations in GPT](https://arxiv.org/pdf/2202.05262)

- Nanda et al. [Fact Finding: Attempting to Reverse Engineer Factual Recall](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)

- Xiong et al. [Everything Everywhere all at once: LLMs can In-Context Learn Multiple Tasks in Superposition](https://arxiv.org/pdf/2410.05603)

- Davidson et al. [Do different prompting methods yield a common task representation in language models?](https://arxiv.org/pdf/2505.12075)

### Questions
- A listed contribution is that concept vector heads “encode concepts at a higher level of abstraction than FV heads” - can you elaborate on what is meant by this? (Line 86) 
- In Figure 3, left, it appears that the MCQ function vectors are all similar to each other. Do you think this is because they are steering the model to output a letter? Perhaps they are picking up on MCQ “task/concept” rather than the “concept” being specifically tested (e.g. antonyms). If you have the model output the answer instead of a letter in the MCQ setting and recompute the similarity matrices for function vectors, do they become concept-specific again? 
- If function vectors encode both concept and format, have you tried disentangling function vectors further to extract either the concept or the format from them separately?
- The claim that "invariance and causality" are implemented separately by the model is a bit strange to me. I understand the part of the argument: attention heads for the concept vectors and attention heads for function vectors have little to no overlap. Is there more you're trying to say here beyond that? If so, I'm not sure what the evidence is that supports further claims

Other Notes:
- Related to your ambiguousICL setting, you may be interested in [Xiong et al]’s setup of task superposition, which multi-task behavior seems like a natural consequence of cross-entropy loss.
- You may also be interested in [Davidson et al], who study instruction prompt version of function vectors. They find mostly disjoint attention heads are responsible for the same task when specified via few-shot or instruction prompts.
- From your results it seems like function vectors are more “equivariant” than “invariant”, i.e., the output when steering tends to change along with the input content compared to the concept vectors. The idea that equivariance to prompting style (function vectors) and invariance to prompting style (concept vectors) are implemented separately is interesting (extrapolating from your finding that the heads are mostly disjoint). I wonder if this can tell us something about the nature of in-context learning and adaptive computation in LLMs? Perhaps it is best viewed as a composition of several mechanisms, some of which are equivariant to the context and others which are invariant for certain concepts. Curious if you have thoughts here if you have any. 


- Minor Typos:
    - Figure 6, Line 296, Line 412: “Ambigous” -> Ambiguous
    - Line 852: “mulitple” -> multiple

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper distinguishes two types of interpretable vector representations in LLMs: Function Vectors (FVs) derived via activation patching and Concept Vectors (CVs) derived via representational similarity analysis (RSA). Through systematic experiments across seven conceptual relations and three input formats (open-ended English, multilingual, and multiple-choice), the authors show that FVs capture causal features that drive in-distribution performance but are sensitive to surface format, while CVs encode more abstract, format-invariant concept representations. The two sets of attention heads occupy similar layers but are largely disjoint, suggesting distinct mechanisms for causal effectiveness and conceptual abstraction.

### Strengths
Well-designed methodology combining activation-based and representational analyses; broad and carefully controlled experimental setup (languages, formats, concepts); clear empirical evidence supporting the causal vs. invariant distinction; insightful visualization and analyses of head overlap and cross-format transfer; overall a solid and novel contribution to mechanistic interpretability of ICL.

### Weaknesses
- It is not surprising that the function vector is different from concept vector. Function vector is responsible in "doing this task", where concept vector is tailored for "summarizing the topic of this task". One is execution (based on low-level semantics such as current formats) and one is abstraction.
- A detailed study of the mechanism - how the two sets of heads cooperate with each other semantically - is crucial but currently absent.
- Several related work might be of insteast and warrent a detailed discussion:

Yang et al. Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning

Bu et al. Provable In-Context Vector Arithmetic via Retrieving Task Concepts.

Han et al. Emergence and Effectiveness of Task Vectors in In-Context Learning: An Encoder Decoder Perspective

- 1050 prompts might be insufficient.

### Questions
What suggestions would you provide to deep learning theoretical literature to model the function-vector and concept vector? Especially, for Bu et al., what can you suggest for their theoretical modeling to consider tasks beyond factual recall? Would there be any better algebratic manner to handle theoretical modeling of in-context learning with those vectors based on your findings, in the consideration of broader task types?

### Soundness
3

### Presentation
3

### Contribution
2
