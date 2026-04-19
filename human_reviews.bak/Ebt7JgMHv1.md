# Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching

- Decision: Accept (poster)
- Scores: 8, 3, 8

## Abstract
Mechanistic interpretability aims to attribute high-level model behaviors to specific, interpretable learned features. It is hypothesized that these features manifest as directions or low-dimensional subspaces within activation space. Accordingly, recent studies have explored the identification and manipulation of such subspaces to reverse-engineer computations, employing methods such as activation patching. In this work, we demonstrate that naïve approaches to subspace interventions can give rise to interpretability illusions.

Specifically, even if patching along a subspace has the intended end-to-end causal effect on model behavior, this effect may be achieved by activating \emph{a dormant parallel pathway} using a component that is \textit{causally disconnected} from the model output.
We demonstrate this in a mathematical example, realize the example empirically in two different settings (the Indirect Object Identification (IOI) task and factual recall), and argue that activating dormant pathways ought to be prevalent in practice.
In the context of factual recall, we further show that the illusion is related to rank-1 fact editing, providing a mechanistic explanation for previous work observing an inconsistency between fact editing performance and fact localisation.

However, this does not imply that activation patching of subspaces is intrinsically unfit for interpretability.
To contextualize our findings, we also show what a success case looks like in a task (IOI) where prior manual circuit analysis allows an understanding of the location of the ground truth feature. We explore the additional evidence needed to argue that a patched subspace is faithful.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper identifies a challenge for the approach of looking for subspaces corresponding to causal factors of deep neural network output.  It claims that activation patching and subspace patching approaches are at risk of an interpretability illusion where the patching fails to change the causal pathways actually leading to undesired input.  It develops the concepts of dormant pathways and causally disconnected pathways to explain why the interpretability illusion is possible.  The paper illustrates its model of the interpretability illusion using a toy example.  Furthermore, it presents real-life case studies.  One set of case studies focuses on language models, particularly the indirect object identification task.  For the IOI task, previous works have used activation patching to correct a network to output the indirect object of a sentence rather than the direct object.  The paper identifies an approach to the task, using distributed alignment search, that is subject to the interpretability illusion.  It also identifies an approach, using prior knowledge about the network to identify the nodes responsible for the error, that correctly patches the network.  The paper explains the difference between these approaches using the concepts of dormant and causally disconnected pathways.  Furthermore, the paper also studies factual recall in language models as an additional case study.  It presents a set of experiments where it shows that the vulnerability of activation patching to the interpretability illusion depends on the layer which is targeted.  The paper also includes some additional discussion on connections between activation patching and rank-one model edits.

### Strengths
This is a highly relevant work for the field of deep neural network interpretability, as it identifies a serious obstacle to the activation and subspace patching approaches.  The concepts of dormant and causally disconnected pathways are intuitive, and they are developed with the aid of theory and experiment.  The paper is clearly written, and figures help illustrate the concepts of the interpretability illusion, and the mechanisms involved in the IOI task.  This paper is significant for both identifying a practical problem and for developing intuitions that deepen our understanding of interpretability methods for deep neural networks.

### Weaknesses
The definitions seem rather brittle.  Requiring strict equality in the definitions of causally disconnected and dormant seems to be very limiting for practice.

### Questions
1. How do we know that the theoretical account for the interpretability illusion is actually the explanation for the failure of activation patching in the case studies presented?
2. Are there a set of empirical predictions (e.g. about the directions of activations or gradients) that could either falsify or support the theoretical model?
3. Why not relax the definitions of "causally disconnected" or "dormant" to not require strict equality, but rather to specify that the effect of the the patching be small within some threshold?

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper shows that subspace activation patching, a technique in mechanistic interpretability used to find subspaces that are “causally” responsible for models to produce certain outputs, may not be reliable. In particular, the paper explains why these may be misleading, involving the activation of dormant pathways. Real-world examples involving indirect object identification (IOI) and fact editing are shown where such techniques are misleading.

### Strengths
+ The paper presents an interesting hypothesis that subspace activation patching methods can be misleading because of dormant pathways. On an intuitive level, this is an interesting hypothesis and needs to be considered for future applications of activation patching.

### Weaknesses
The paper does not pose an explicit hypothesis that can be falsified, limiting its scientific validity. Further, the "illusion" seems to make **problematic assumptions** and has unclear implications. 

- The hypothesis seems to be that every patched subspace “v” discovered by DAS can be decomposed into two directions “v_disconnected” and “v_dormant” (with specific properties), and that "patching along the sum of these directions, the variation in the disconnected part activates the dormant part, which then achieves the causal effect". The latter statement assumes that the model behaves somewhat linearly along these subspaces (i.e., the output of the model along direction "v" is given by a sum of its outputs along "v_disconnected" and "v_dormant"), which is a strong hypothesis given that these models are fundamentally non-linear. While the paper provides some evidence for the existence of some "disconnected" and "dormant" directions, unfortunately, I do not see evidence justifying the apparent linear behaviour of the underlying model.

- The definition of the "dormant" subspaces is confusing in the context of this work. If the model output remains unchanged for in-distribution patching and changes only for out-of-distribution samples $x,y \sim \mathcal{D}$, what procedures presented in this work result in out-of-distribution samples/activations? As far as I can tell, all procedures described involve patching with in-distribution data.

- Is the hypothesis that such a decomposition does not exist for “true” subspace directions?

---

The paper (especially sections 4-7) is difficult to read, is very dense, and has **several omitted details**. 

- The main paper on its own does not seem to be self-contained and seems to contain a significant number of references to the appendix. 

- The main hypothesis involves the claims that (1) subspace directions can be decomposed into two directions (disconnected, dormant) with specific properties, and (2) the effect of such directions adds somewhat linearly to the model outputs. However, the paper fails to connect this terminology ("disconnected, dormant directions") in sections 4-7, making it difficult to verify whether the experiments confirm or deny the hypothesis. 

- The writing and presentation are sloppy. For example, in Section 4, “ker W_out” is never defined, and it is unclear how the results in Table 1 and Figure 3 relate to the description in section 4.3. What is ABB / BAB in Figure 3? What does “connected” in Table 1 refer to? Overall, how do Table 1, Figure 3, and Figure 4 illustrate support for the presented hypothesis?

---

Overall, while this work may present a useful point, its writing (especially sections 4-7) makes it impossible to verify this. I suspect that a thorough rewrite is necessary to clarify its central point.

### Questions
- For a future draft, I encourage the authors to present an explicit falsifiable hypothesis that facilitates both experimentation and analysis. 

- It might be helpful to comment on the downstream applications and practical utility of such activation patching techniques and mechanistic interpretability, particularly to better understand the implications of the "illusion". What are the use cases for identifying model components (neurons/subspaces) responsible for model behavior? What evaluation metrics are available to test whether the components have been correctly identified?

- Do these results for subspace activation patching also hold for usual activation patching? It seems like subspace patching is a generalization, and thus it must, and also I see that the toy example is given for the case of usual activation patching, but the experimental results and the paper's messaging are specific to subspace patching.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work finds and shows that subspace activation patching may be subject to interpretability illusions caused by activating a dormant pathway (i.e., does not respond to input changes) but is activated by a causally disconnected feature (i.e., is activated by the input change but is not connected to the model’s output). They show that this illusion also relates to rank-one fact editing (e.g., Meng et al. [1]) and explains its recently found inconsistencies [2]. Finally, they demonstrate that further analysis, e.g., manual circuit analysis, can mitigate the interpretability illusion.

[1] Meng, Kevin, et al. "Locating and editing factual associations in GPT." NeurIPS 2022.

[2] Hase, Peter, et al. "Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models." arXiv 2023.

### Strengths
* Given the rising popularity of mechanistic interpretability and (subspace) activation patching, this paper addresses a very important and significant issue in a timely manner.

* The interpretability illusion is well-motivated and clearly introduced. The formal definition (p. 5) is sound. It is also clear that the illusion is not a mere artifact of the chosen experimental settings but is present in most cases with high probability.

* The experimental design to showcase the interpretability illusion is well-designed.

* The discussion on how one can prevent the interpretability fallacy is important and sound. It paves a way for future work that relies on automatic activation patching methods, to avoid false interpretations of model behavior.

* The discussion on the presence of the interpretability illusion from a mechanistic viewpoint is sound.

* The paper is clearly written, (mostly) easy to follow, and self-containing.

### Weaknesses
This paper is a very strong submission. It is well-motivated, sound, and clear. The only “major” weakness is that code is not provided and minor comments (see below).

### Questions
* Why do the authors solely focus on subspace activation patching? The identified interpretability illusion should also hold for (automatic) component activation patching (i.e., consider the entire activation space as the (sub)space).

* There seems to be an error in the indices in the toy example in Appendix A.3.

## Suggestions

* While Fig. 1 clearly demonstrates the interpretability illusion, it is hard to parse. It may be good to make it more accessible/easier to parse, as it demonstrates the main insight of the paper.

* It’d be good to add the relation of the vector $v$ to the subspace $U$ in Sec. 3.

* It’d be good to match the notation of Tab. 1 and the respective text.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent
