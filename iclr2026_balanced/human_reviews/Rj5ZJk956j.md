## Human Reviewer 1

### Summary
The authors identify neurons that conditionally write (strengthen, weaken) concepts. They study the behavior of these neurons in depth.

### Strengths
The paper includes baselines and conducts a large number of experiments to identify and understand a novel type of behavior in LLMs. It is especially surprising how universal this mechanism seems to be. I appreciated the section on the intuition the authors gained.

### Weaknesses
The presentation could benefit from being more crisp about what the claims are. In addition, it was unclear to me why finding negative gating was meaningful.

### Questions
Could you include concrete examples in your presentation? Of certain neurons and their function based on your analysis. Right now it felt very abstract, it would be beneficial to have particular case studies that highlight the general trend.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper is an interpretability study of neurons in gated MLPs as used in modern LLMs, mainly by considering the angles between input, gate, and output vectors. Neurons are classified on the basis of the rough size/magnitude of the cosines between these angles. Among other findings, the key finding from the paper is that "weakening neurons" have substantial influence despite being not so large in number.

### Strengths
- Given how prevalent gated MLPs are in modern LLMs, and how the function of MLPs tends to still be much less clear than that of attention, a study elucidating their inner workings can be a welcome addition to the literature. 
- The paper studies a substantial variety of models from different families and sizes, establishing some key conclusions (Section 3) across them.

### Weaknesses
I believe the current version has the following weaknesses. I'll be happy to reconsider my score based on the response.

Rigor-related weaknesses:
- Section 4.1: Ablations are done using zero ablation. Zero ablations are quite destructive and may move the model activations out-of-distribution. Could the outsized effect of weakening neurons somehow be caused by a confound related to this ablation method? Could the same effect be shown with a less destructive ablation method (e.g., mean ablation)?

Clarity-related weaknesses:
- line 114: "Additionally, it guarantees that two equivalent neurons by property (2) will now also have the same weights" -- I don't understand what this means
- line 137-142: I found this paragraph confusing and didn't understand what it aims to express. What is "atypical" here meant to express?
- Section 3: What is the input data? I didn't see this specified. I'm also confused that line 262 states that the model is run on a dataset in Section 4, which suggests Section 3 doesn't involve running the model on any data?
- Figure 4: Why is there no "clean" curve even though it appears in the caption?

### Questions
- I have one question about how exactly the alignment between vector directions is defined.
-- line 124 has $\approx \pm 1$, whereas Table 1 has $>> 0$, $<<0$. Can the authors comment on whether there is an inconsistency here?
-- line 126: "roughly collinear" is this supposed to mean the same as $cos(...) \approx \pm 1$? It could improve readability of the paragraph to streamline terminology.

Minor suggestions:
- Since Swish isn't as broadly familiar as ReLU, but is central to the paper, it couldn't hurt introducing it in Section 1.2 explicitly
- Figure 2: this is very small
- page 6: there is a huge waste of space, which the authors could use more efficiently

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
2

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper investigates a new phenomenon, weakening neurons, which occur in MLPs that use the SwiGLU activation function (as most LLMs currently do). These neurons appear in later layers, activate often, and write in the opposite direction as they read; this holds across a variety of models. Ablation experiments indicate that these neurons play a role in sharpening the model's output distribution. In addition to introducing these weakening neurons, this paper discusses other sorts of neurons, such as strengthening neurons, which are involved in early-mid layers.

### Strengths
This paper digs into the ways in which specific neurons act to shape the model's output distribution. It's long been hypothesized that later MLPs work to refine the model's output distribution, so it's interesting to understand how exactly models do this. The methods used are quite simple, but this isn't a bad thing - it keeps the paper clear, and simple causal methods are often good enough. The overall significance of this is unclear, but I think this work will be interesting to some mechanistic interpretability researchers.

### Weaknesses
Though I think this paper makes an interesting and valuable contribution, I also think that it does not yet meet the bar of an ICLR paper. 

The primary issue is that the paper is sparse: it just doesn't have much content. It's 8.5 pages long, but if the figures were tightened up, it would easily be under 8 pages. These pages could be used to address real deficiencies in the paper. For example they could provide a case study in what exactly these weakening neurons do, as right now, discussion of these weakening neurons is entirely abstract. Instead of just noting that the distribution gets sharper, they could look at a specific example and show how these weakening neurons sharpen the distribution. I see that there is a section doing this in the appendix that goes entirely unmentioned in the main text - perhaps it's worth bringing that into the main text? Another good use of this space would be to study a model besides OLMo.

It's also worth engaging more with prior literature. In recent years, there has been quite a bit of debate about whether neurons are the right unit of analysis to study, and many possible solutions to related issues like polysemanticity / superposition. I don't think that these issues are totally critical for this paper, but it's worth engaging with that literature and thinking about how it might inform your results.

There are some minor issues too: unexplained choices (why use attributes score in Figure 4, and how is it measured?), hard-to-read figures (Figure 5 could be made horizontal, and more space could be given to the y-axis), and other such polish-related issues.

Finally, as a core mech interp researcher, I find this paper interesting. But, I think it could do a better job of motivating the question and methods used to study it - not just in terms of e.g. Gurnee et al.'s interest in this sort of mechanisms, but in terms of the broader impact of this question, at least for the interp community.

### Questions
- Why do you use attributes score in Figure 4, and how is it measured?
- How do you think your findings are affected by polysemanticity / superposition?

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
4

### Confidence
4