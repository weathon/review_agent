## Human Reviewer 1

### Summary
The authors study the IOI circuit introduced by Wang et al (2023). [[1]](https://arxiv.org/pdf/2211.00593#page=4.37) They notice that the hypothesized algorithm implemented by the circuit would not work in cases where the IO token appears twice in the prompt (DoubleIO prompt). However, they note that the model and circuit can in fact predict the correct token on the DoubleIO prompt.

The authors hypothesize the circuit is able to perform well on the DoubleIO prompt by "S2-hacking". That is, having the Induction, Previous Token and Duplicate Token heads pay extra attention to the second instance of the S2 token, thereby causing the S-inhibition heads to suppress the prediction of this token and thus predict the IO token more strongly.

They further identify a circuit for the DoubleIO prompt using activation patching and find that it has significant overlap with the original IOI circuit and reuses many of the same components.

### Strengths
Evaluating on prompt variants intended to test specific aspects of the operation of the circuits is a good idea.

It is a useful finding that the IOI circuit components are often reused in the DoubleIO prompt formats.

The authors present some interesting experimental results demonstrating the similarity of the functions of the attention heads in the IOI circuit between the IOI and DoubleIO prompts.

### Weaknesses
- The experimental reports are lacking in many details about the experimental methodology, making it difficult to be confident that the claims are robust.
- The explanations throughout the paper should be clearer to fully communicate the ideas and experiments of the authors.
- The S2 hacking hypothesis is quite vague and the author do not present any deep understanding that would explain the mechanisms by which certain attention heads pay extra attention to the S2 token.
- In the experiments on the DoubleIO and other prompt variations, it is unclear at which token positions paths are being ablated, as this is unspecified by the original circuit.
- The authors write: “Given the algorithm inferred from the IOI circuit, it is clear that the full model should completely fail on this task”. However this is a misunderstanding of the original work. The IOI circuit was discovered using mean ablations that keep most of the prompt intact. Therefore Wang et al. don’t expect it generalize to different prompt formats.
- The authors write “In the base IOI circuit, the Induction, Duplicate Token, and Previous Token heads primarily attend to the S2 token” this is incorrect according to Section 3 of Wang et al., 2023. These heads are _active_ at the S2 token, but do not primarily attend to it.
- The authors write: "The proposed IOI circuit is shown to perform very well while still being faithful to the full model" In fact, the IOI circuit is known to have severe limitations, as shown in concurrent work by Miller et al. (2024) [[2]](https://arxiv.org/abs/2407.08734).

Nitpicks:
- In Figure 2, it is not clear what Head 1, 2, 3 and 4 refer to.
- The paper should include Figure 2 from Wang et al. 2023 [[1]](https://arxiv.org/pdf/2211.00593#page=4.37) to make it easier to follow discussions about the circuit.

### Questions
The authors write: “since the S2 token is the only input with a path to the END token through the Induction, Duplicate Token, and Previous Token heads, these heads do not need to attend to any of the other input tokens” I don't understand what exactly this means. Why do the other heads not need to attend to any other input tokens?

### Soundness
1

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper explores how IOI circuits respond to new modified prompts specifically crafted to disrupt the original pre-defined circuits' functionality. These variations introduce two new sets of IOI prompts: Double IO and Triple IO, named according to the frequency of IO occurrences within the prompts. The study reveals that, while the circuit structures remain largely consistent, their functionalities exhibit slight variations. As the functionality of nodes in the IOI circuits are mainly discussed under the settings of the original IOI prompt, it is not surprising to see the variations in functionality when prompts are different.

### Strengths
The discussions in this paper are comprehensive and closely follow the structure presented in "Interpretability in the Wild." Emphasizing the limitations in estimating the functionality of IOI circuits is crucial for advancing the field of interpretability. Such an emphasis underscores the challenges in understanding how these circuits function across varied prompt structures, particularly when circuit consistency does not guarantee consistent functional behavior.

### Weaknesses
- The evaluations on circuits discovered on IOI double and IOI triple is extremely trivial. The logit difference is not even close to the full model. It is hard to be convinced that these circuits are even faithful and accurate to reflect how the model actually perform such tasks. These also diminish the credibility and validity of the findings from later parts. The logit difference of circuit should at least recover 80% of that from the full model to be considered as faithful and accurate. 
- There is no empirical evidence to show that the originally designed functionality (algorithm) of IOI would fail on the two new prompts. Though I like the idea, it is hard to convince with no evidence that the designed prompts are able to satisfy the purpose. I suggest to run some simulations. 
- Since the circuits are found by hand with path patching, and the logit difference varied significantly from the full model, it is hard to justify if the circuits found are served mainly as to make some specific conclusions in the paper such as 'the circuits are largely reused in different prompt settings'.  Showing KL and logit difference together while being close to the full model would be preferred.

### Questions
- What is the KL divergence of the three circuits for IOI, double and triple IO. 
- Is the functionality of nodes in this paper mainly focus on the change in logit difference similar to IOI paper. If so, why? How about based on change in overall logit distribution? 
- Please refer to weakness section.

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper evaluates the indirect object identification (IOI) circuit in GPT-2 using variant prompts (DoubelIO and TripleIO). These variants introduce additional objects (IO), thus expected to disrupt the identified circuit’s performance. The authors found that the circuit maintains high performance due to a ‘S2 hacking’ mechanism, where the head defaults to the correct token. The base IOI circuit is largely reused for the variant prompts.

### Strengths
1. Originally: Evaluated the IOI circuit using variant prompts, providing new insights into its robustness.
2. Quality: The approach with the DoubleIO variance is motivated and effective. 
3. Clarity: The writing is clear and easy to understand. 
4. Significance: demonstrated that the IOI circuits are more adaptable than previously understood.

### Weaknesses
1. The paper could benefit from a clearer discussion of its motivation and broader context. The authors highlight "circuit analysis as a promising angle.” Is it expected that a "simple, human-interpretable algorithm" exists for any task? So far, the study feels narrowly confined to a specific problem. The authors can discuss how the approach in the paper might generalize to more complex tasks. One potential way to address this is to relate the findings to more general problems, such as in NLP, recognizing the relation between subjects in a sentence.
2. The paper relies heavily on differences in attention scores to evaluate circuit functionality, but it lacks a thorough justification for this choice. Is it possible for the same functionality to be represented by different attention patterns? Alternatively, could similar attention scores result in different interpretations of functionality? A clearer explanation of the link between attention and underlying mechanisms is needed. 
3. Some statements lack contextual clarity, such as the "first demonstration of generalization via circuit reuse." Providing concrete context or evidence for each claim would enhance rigor and clarity.

### Questions
1. How might the proposed circuit-based approach manage tasks that involve multiple subjects or ambiguous variables? For example, are there instances where tokens other than the IO or S have the highest probability? If so, would the ratio score introduced in the paper sufficiently capture the model’s mechanism in these cases, or would normalizing by the highest predicted token likelihood be necessary?
2. The current approach relies on identifying the functionality of attention heads by using predefined target tokens. Could you elaborate on its adaptability to tasks where reasoning components are less clearly defined, or multiple relevant variables may need to be considered?
3. Minor Point: Please consider reporting variances for all values presented, such as in Figures 7 and 8, to give a more comprehensive view of the data.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper investigates circuit reuse in language models.
It does so in the context of the IOI task and its circuit. 
In particular, it proposes 2 variants to the standard task which follow a similar 
structure to the original, and then proceeds to the analyze the circuits/behavior of the
model for these variants. 
In these investigations it finds two different things.
1. The standard IOI circuit has really good performance on these tasks despite it being unfaithful.
The authors explain this behavior by a phenomenon they call S2-Hacking. In this phenomenon 
the S2 token, via S-inhibition heads + incorrect the mean ablation, ends up artificially pushing 
the name mover heads to return the IO token.
2. The circuit for the DoubleIO tasks is very similar to the circuit in the standard IOI task.
The authors discover this by applying standard path-patching techniques to discover the new circuit and comparing it to the original. 
This observatin is important as it shows that circuits are "robust" 
and are shared in similar enough tasks.

### Strengths
- I think the paper was presented in a very clear manner, and the authors did an excellent job of explaining the methodology and their ideas.
- Their methodology seems adequate, as it follows standard practices for circuit discovery.
- I appreciated that the authors investigated the S2-Hacking issue in detail, as I believe this sort of problem can lead to "interpretability illusions" where we might be deceived into thinking a circuit is appropriate when, in reality, it is simply a quirk of the methodology. (I don’t think this fits precisely into that category, as the circuit was quite unfaithful, but it remains an interesting finding.)
- As the authors mention, to the best of my knowledge, this is one of the first careful looks at circuit reuse in LLMs.

### Weaknesses
- In my view, the main weakness of this paper is its scope. Although I think the authors did an excellent job presenting their results, I believe that it is necessary to validate the conclusions with more circuits. I understand that several papers only focus on one specific circuit (the IOI paper being a prime example). However, as the techniques presented in the paper are not new and not necessarily its main strength, I believe that more empirical validation should be provided to strengthen the paper's thesis. For example, I would like to see similar analysis (which could be automated) for other circuits, such as Greater-Than [1] or Docstring [2]. I think this would greatly strengthen the paper. Otherwise I think the contributions, despite the excellent presentation and great use of the methodology, might not be enough. 



[1] M. Hanna, O. Liu, and A. Variengien. *How Does GPT-2 Compute Greater-Than*. Interpreting Mathematical Abilities in a Pre-Trained Language Model, 2:11, 2023.

[2] S. Heimersheim and J. Janiak. *A Circuit for Python Docstrings in a 4-layer Attention-Only Transformer*. Alignment Forum, 2023. URL: [https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only)

### Questions
Nitpicks:
-  In paragraph 4 of section 4, should we have S2 rather than S? "These heads then feed into the S-Inhibition heads, which will always suppress attention on the S tokens as a result and push the Name Mover heads toward returning an IO token."
- Have the authors attempted to analyze other circuits in a similar way? If so what have been their findings? 
- Are the authors aware of other instances of other "illusions" similar to S2 hacking?

### Soundness
4

### Presentation
4

### Contribution
2

### Rating
5

### Confidence
4