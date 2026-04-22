# Prototype Transformer: Towards Language Model Architectures Interpretable by Design

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4, 4

## Abstract
While state-of-the-art language models (LMs) surpass the vast majority of humans in certain domains, their reasoning remains largely opaque, undermining trust in their output. Furthermore, while autoregressive LMs can output explicit reasoning, their true reasoning process is opaque, which introduces risks like deception and hallucination. In this work, we introduce the Prototype Transformer (ProtoT)—an autoregressive LM architecture based on prototypes (parameter vectors), posed as an alternative to the standard self-attention-based transformers. ProtoT works by means of two-way communication between the input sequence and the prototypes, and we show that this leads to the prototypes automatically capturing nameable concepts (e.g. “woman”) during training. They provide the potential to interpret the model’s reasoning and allow for targeted edits of its behavior. Furthermore, by design, the prototypes create communication channels
that aggregate contextual information at different time scales, aiding interpretability. In terms of computation scalability, ProtoT scales linearly with sequence length vs the quadratic scalability of SOTA self-attention transformers. Compared to baselines, ProtoT scales well with model and data size, and performs well on text generation and downstream tasks (GLUE). ProtoT exhibits robustness to input perturbations on par or better than some baselines, but differs from them by providing interpretable pathways showing how robustness and sensitivity arises. Reaching close to the performance of state-of-the-art architectures, ProtoT paves the way to creating well-performing autoregressive LMs interpretable by design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The presented paper proposes a transformer variant that integrates prototypes inspired from the prototype learning done in computer vision with the aim to build an LLM architecture that is more explainable by design than regular attention.

### Strengths
- Interesting approach to combine prototype learning from the computer vision space with modern LLM architectures
- Insightful analysis on noise robustness of the prototypes
- Competitive performance of the introduced architecture with linear computational complexity compared to standard attention-based architectures

### Weaknesses
###  Weaknesses

- The intervention approach to the prototypes is not much different to what has been done to specific translation heads to determine their function e.g. there exist works that have specifically identified specific "translation heads" for multilingual mappings and especially crucial for the translation task. Similarly to the "male" and "female" prototypes, one could analyze translation heads in a similar fashion and we do not necessarily need the prototypes.
- To understand the role of each prototype one needs to have a reasonable hypothesis and probing dataset e.g. as done in the paper, there should be a prototype somewhere that encodes the gender (male/female) attributes. To me this seems rather difficult to distill for a large number of prototypes and might also have issues with confirmation bias. In an ideal world, we would have some sort of "prototypical" visualization similar to how it is done in computer vision to assess each prototypes function more holistically. 
- I found the interpretability portion of the paper to be rather short and would've hoped for a more thorough discussion of the benefits compared to similar masking experiments on attention heads and/or more findings distilled from the prototypes. Appendix A. 8 is interesting as we can see that `L7 P31` seems to be closely related to diseases or `L10 P8` seems to be related to germany.

### Minor Comments

- l. 152: `They are R parameter vector` -> `There are ...`

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the Prototype Transformer (ProtoT), an autoregressive language model that replaces standard self-attention with a prototype-based mixer. Each layer contains a set of learnable prototype vectors that communicate bidirectionally with token embeddings through “read” and “write” gates. The architecture enforces strict causality (tokens can only aggregate information from previous positions) and achieves linear scaling in sequence length, compared to the quadratic cost of self-attention.

The authors argue that this design enhances interpretability, as prototypes act as semantic hubs capturing coherent, nameable concepts (e.g., “woman,” “verb,” “disease”). They show qualitative examples of interpretable prototypes and conduct targeted interventions (e.g., reinitializing gender-related prototypes) to illustrate functional specificity. ProtoT is evaluated on perplexity, scalability, and GLUE benchmarks, showing performance close to or slightly better than comparable transformer baselines. Additional experiments assess robustness under noise, semantic interventions, and prototype clamping, claiming that ProtoT offers more interpretable and stable behavior.

### Strengths
1. Novel architectural contribution: Replacing attention with prototype-based communication is an original and thought-provoking idea. The design draws on interpretability intuitions (discrete, nameable concept vectors) while maintaining competitive performance.

2. Linear computational scaling: The causal prefix-based computation provides linear time complexity with respect to sequence length, which is valuable for long-context modeling.

3. Empirical competitiveness: Despite removing self-attention, ProtoT achieves perplexity and GLUE scores comparable to transformer baselines, suggesting that the architecture is not just an interpretability toy model but a viable alternative.

4. Prototype interventions: The paper demonstrates that disrupting individual prototypes can cause semantically targeted effects (e.g., lowering the probability of “woman” when a “female” prototype is reinitialized). This provides concrete, though limited, evidence of concept-level disentanglement.

5. Comprehensive robustness analysis:
The study includes several robustness evaluations (noise perturbations, prototype clamping, semantic interventions), providing a broader view of how the model responds to changes than typical interpretability papers.

### Weaknesses
1. Interpretability evaluation is not systematic. 
The interpretability claims rest on a few qualitative examples and one prototype intervention. There is no quantitative evaluation or comparison to standard attention heads (e.g., using interpretability metrics or human judgment).
For example, attention heads in GPT-style models also show easily nameable functions (e.g., syntactic dependency heads, gender-related heads, or token-copying heads). Without such comparison, it is unclear whether ProtoT’s prototypes are inherently more interpretable or simply a different representation of similar phenomena.

2. Lack of architectural clarity.
The model description is highly technical and equation-heavy, but lacks a schematic or clear visual illustration of the prototype mixer, read/write gates, and causal structure. This makes it difficult to grasp how information flows through the network and what distinguishes it conceptually from attention.

3. Questionable link between robustness and interpretability.
The robustness experiments primarily assess stability under perturbations (noise, synonym replacement, etc.), which is not equivalent to interpretability. While “prototype-mediated robustness” is an interesting idea, it measures internal routing stability rather than human-understandable reasoning transparency.

4. GLUE benchmark does not demonstrate inference or reasoning interpretability.
GLUE mainly tests fine-tuned classification tasks, not generative or reasoning-heavy inference. Thus, it does not show whether the prototype-based routing would perform comparably to attention in open-ended language modeling or long-context reasoning.

5. Arbitrary and under-justified hyperparameter choices.
Some architectural design decisions appear ad hoc. For instance, the “local convolution across 4 past tokens” at layers 0 and 1 is introduced without justification or ablation beyond a brief perplexity mention. It is unclear why 4 tokens were chosen, how sensitive results are to this choice, or whether such local convolutions meaningfully relate to the prototype mechanism itself rather than compensating for architectural limitations.

6. Unclear optimization target for low-rank/value projection.
The authors use a low-rank projection at half of the hidden size on the value path to reduce compute, claiming only minor perplexity cost. However, it is unclear whether this step prioritizes performance over interpretability. Since the paper’s main goal is interpretability, it is not justified how this choice affects prototype clarity or whether it interacts with the interpretability of the learned representations.

7. Limited evidence and lack of quantitative validation for polysemanticity and half-life correlations.
While the authors note that polysemanticity appears in some prototypes and suggest a correlation between half-life values and the types of encoded concepts (e.g., lower half-life capturing local elements like entities or punctuation), the evidence is largely qualitative. They do not provide:
- A systematic overview of which prototypes are polysemantic or how prevalent this phenomenon is.
- Quantitative measurements to substantiate the claimed correlation between half-life values and concept types.

As a result, the claims about prototype hubs being largely disentangled and interpretable remain suggestive rather than rigorously demonstrated, which weakens the interpretability conclusions.

8. Cherry-picked interpretability examples.
The prototypes shown in the paper appear selected to illustrate the interpretability claim (e.g., a “female” prototype). The main paper includes only one such example, with more shifted to the appendix. There is no evidence that interpretable prototypes are common rather than rare.

9. Limited experimental scope.
Experiments are performed on small-scale models (6 layers, 256 hidden units, context 256), using a 250M-token dataset. It is unclear whether the interpretability and performance claims hold for larger models or longer contexts.

10. Causality claim lacks direct validation.
The prefix-mean formulation is described as enforcing strict causality, but the paper does not empirically analyze whether this design changes the model’s information flow compared to attention.

11. Presentation issues.
The paper is dense, with heavy algebraic exposition and limited conceptual framing. Important design motivations (e.g., why this form of bidirectional gating should yield interpretability) are obscured by implementation details. Visual aids and clearer methodology would make it much more accessible. Furthermore, the paper is not rigorously proofread, containing minor typos such as line 305 (wrong capitalization) and line 376 (misplaced comma). The writing also frequently mixes results into the methodology section, with justifications often phrased as “we observed that this works best,” which blurs the line between design rationale and empirical findings. In addition, references are inconsistently formatted, for instance, arXiv citations appear in two different styles.

### Questions
1. How do you quantitatively evaluate interpretability? Do you have any statistics on how many prototypes correspond to coherent or nameable concepts, for example using automated mechanistic interpretability methods or human-based evaluation?


2. Can you provide a comparison with interpretability in attention heads (e.g., showing analogous heads from LLaMA on similar examples) to support the claim that ProtoT is inherently more interpretable? Human-based evaluation could also help establish this.


3. How does ProtoT perform on generative or long-context tasks beyond GLUE fine-tuning? Does the prefix-mean and time-discount mechanism maintain performance at longer contexts or larger model scales compared to self-attention?


4. Some architectural/hyperparameter choices appear ad hoc (e.g., layer-0 4-token convolution, number of prototypes, read/write temperatures, alpha-gate initialization). How sensitive are the results to these choices, and what is the rationale behind them?

### Soundness
2

### Presentation
1

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
This work proposes an alteration to standard self-attention-based transformers, titled Prototype Transformer, that enforces model computation to route through prototypes, aimed at increasing interpretability and controllability. The authors compare their ProtoTs to Llama, Mamba, and DeltaNet, finding that ProtoTs have much worse perplexity but perform relatively on par for downstream evaluations like GLUE finetuning. They also report results on robustness, finding that ProtoTs offer improved robustness. They qualitatively evaluate the interpretability of the learned prototypes, providing examples of prototypes corresponding to gender attributes that can be controlled.

### Strengths
- The aim of creating an inherently interpretable language model is very interesting and a difficult and under-explored problem. Prototype learning is an intuitive way to go about this, particularly given its success in computer vision.
- The authors perform extensive training experiments, including very interesting scalability analysis.
- Furthermore, the evaluation comparing ProtoTs to Llama/SSMs finding improved robustness and on par benchmark performance despite worse perplexity is also quite interesting.
- The proof of concept interpretability analysis seems promising for the few provided qualitative examples.

### Weaknesses
- It is unclear to me how well the ProtoTs function as language models. While the perplexity is clearly worse, examples of text completions and general language modeling utility would be helpful to provide intuition as to how meaningful in the difference is. 
- How do the authors test the actual interpretability of the learned prototypes? An automated interpretability evaluation pipeline (for example, see SAEBench [1]) could be a way to do this.
- The downstream utility of ProtoTs, and more generally inherently interpretable transformers, over “post-hoc” explanation methods like circuit tracing/SAEs/transcoders/etc was not made very clear. Are ProtoTs expected to be better for intervention and steering, debiasing, safety auditing, etc? Given the lack of sparsity for ProtoTs over alternative methods, the added utility is not immediately apparent to me. An exploration of any potential downstream application would significantly strengthen this work.

While this work is very promising, given that interpretability is the main motivation, I feel that the interpretability analysis is not very thorough. No quantitative analysis is performed on the interpretability, nor any comparison to alternatives such as popular post-hoc methods. I recommend that the authors complete a more fleshed-out evaluation of the interpretability and downstream utility of ProtoTs to convince readers and reviewers of the benefits of using ProtoTs given the perplexity cost. 

[1] Karvonen, Adam, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau et al. "Saebench: A comprehensive benchmark for sparse autoencoders in language model interpretability." *arXiv preprint arXiv:2503.09532* (2025).

### Questions
- What ensures that the learned parameters are actually “prototypes” of the data? Are they constrained to lie on the data manifold?
- How are prototypes labeled?
- How different would ProtoTs be from replacing all MLPs in a LM with transcoders?
- Lines 120-121: “by treating proto-s as semantic routing vectors that create R distinct communication channels, each with separate read/write gates and learnable time-discount param-s.” These abbreviations decrease legibility. Similar for “ppl” in lines 251-252.
- Line 363: “Polysemanticity is present in a few prototypes but remains limited overall.” Is this a qualitative analysis? Can more details be provided?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ProtoT, a prototype-based autoregressive language model architecture. ProtoT introduces a prototype communication channel to aggregate contextual information and explicitly exposes routing activations, enabling interpretable and intervenable concept representations. The authors conduct evaluations and further examine the model’s robustness under various conditions, including semantic-preserving noise, prototype clamping perturbation, and causal interventions. In addition, a series of interpretability experiments are performed to analyze the model’s internal behavior. Experimental results demonstrate that ProtoT achieves competitive task performance while offering additional tools for robustness analysis and a certain degree of model interpretability.

### Strengths
1. Introducing a prototype mechanism into black-box models is an interesting direction. It enhances interpretability and allows us to explicitly observe and selectively modify the concepts learned by the model.

2. The paper examines model robustness from multiple intervention perspectives and provides interpretability insights based on the routing mechanism, offering a more transparent view of the model’s internal reasoning process.

### Weaknesses
1. There is a noticeable performance gap with top-tier models. Except for RTE and WNLI, ProtoT still shows a significant performance gap compared to LLaMA, indicating substantial room for improvement in terms of model expressiveness and generalization ability.

2. Insufficient baseline comparisons: The experimental section lacks a systematic comparison with a broader range of mainstream methods, primarily comparing with relatively weaker models such as LLaMA. 

3. Limited depth of interpretability analysis: Although the paper presents some activation visualizations, it does not provide a thorough explanation of the semantic meaning, formation mechanism of the prototype concepts during reasoning.

4. The distinction from prior work is unclear. Previous studies have already explored incorporating prototype-based ideas into various neural architectures. To highlight its originality, this work should better articulate its unique contributions and the challenges it addresses compared to existing prototype-based interpretability approaches.

5. Questionable practical significance of interpretable architectures. In the era of LLM, post-hoc interpretability may hold greater promise. Despite their theoretical transparency, many “interpretable architectures” have seen limited adoption in practice. In practice, both researchers and industry practitioners tend to prefer black-box yet highly effective models, suggesting that the practical value of explicitly interpretable architectures should be redefined—perhaps use post-hoc interpretability as diagnostic or auxiliary tools is better.

### Questions
Please see weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the ProtoT, an autoregressive language model that replaces self-attention with a prototype-based communication mechanism. Instead of pairwise token interactions, the model introduces a fixed set of learnable prototype vectors that bidirectionally communicate with input tokens through read and write gates. These prototypes act as semantic channels that aggregate contextual information over time, regulated by per-prototype decay parameters. The design aims to yield an interpretable model architecture that exposes explicit “concept hubs” while offering linear computational complexity in sequence length. The authors evaluate ProtoT on FineWeb-Edu for perplexity, GLUE for fine-tuned downstream performance, and several interpretability and robustness analyses (concept discovery, prototype interventions, and perturbation tests). They show that prototypes appear to capture semantically coherent patterns (e.g., gendered terms, functional words), enabling interpretable interventions with minimal degradation in task performance. ProtoT achieves comparable results to linear-attention baselines but trails behind standard transformers like LLaMA.

### Strengths
The paper pursues an ambitious and timely goal: integrating interpretability directly into the LM architecture, rather than relying on post-hoc analyses. The introduction of prototype vectors as explicit representational channels is conceptually elegant, drawing inspiration from case-based reasoning and slot-attention architectures, and potentially bridging feature disentanglement and mechanistic interpretability.

### Weaknesses
- ProtoT essentially replaces self-attention with a learned prototype mixer that uses fixed latent vectors and exponential decay. While this is a clean design, similar ideas have been explored extensively in slot attention, Perceiver IO/AR, and prototype networks in both NLP and vision. The main mathematical formulation (Eq. 1) is a direct adaptation of cross-attention with time-discounting, without introducing new theoretical mechanisms for interpretability. The paper’s claim that prototypes “capture nameable concepts” is intriguing but not demonstrated beyond anecdotal examples and small intervention cases.

- The visualization and intervention analyses are compelling illustrations but not rigorous evidence of interpretability. For instance, the “female” and “male” prototypes (L9 P7 and L9 P18) are selected manually from hundreds of candidates; the effects shown in Table 6 (e.g., −17.8 % probability for women) are not statistically validated and could arise from spurious correlations in FineWeb-Edu. The concept discovery procedure lacks quantitative measures such as mutual information, sparsity metrics, or concept alignment scores that are standard in interpretability research. Moreover, the authors acknowledge that polysemanticity remains “limited overall” without defining how it is measured. 

- The model’s perplexity (Table 1) and GLUE results (Table 2) indicate clear performance gaps versus LLaMA and Mamba. ProtoT performs comparably to DeltaNet, but that baseline itself is weak. The large-scale setting reduces this gap somewhat, yet ProtoT still trails by several perplexity points, and its long-context scalability is explicitly acknowledged as “poor.” Given ICLR’s standards for architectural contributions, the empirical competitiveness is insufficient to justify acceptance. 

- Although the model claims linear complexity, the real-world throughput (Table 9) shows it is 2×–3× slower than LLaMA at equivalent width/depth, despite linear scaling. The efficiency claim is thus more theoretical than practical. Furthermore, the interpretability-by-design argument conflates readability of intermediate states with faithful reasoning transparency. ProtoT provides observable slots, but it is unclear whether these slots correspond to causal features used by the model during generation. 

- The paper does not explain why prototype-based routing should inherently yield disentangled or interpretable representations. The prefix-mean and time-discount mechanisms are heuristic, and the correlation between decay parameters and “concept timescales” is empirically observed rather than derived. While Appendix A.7 includes layer-0 ablations, these primarily measure perplexity changes, not interpretability effects.

### Questions
- How are “concepts” operationally defined and verified? Are prototypes annotated or evaluated against external semantic taxonomies or concept datasets?

- How reproducible are the concept-specific findings (e.g., gender prototypes) across random seeds and datasets? Could similar prototypes emerge by chance?

- Since prototypes act as a fixed-size communication bottleneck, how does interpretability scale with model size or number of prototypes (R)? Does increasing R simply reintroduce entanglement?

- Could the observed prototype activations be artifacts of token frequency or co-occurrence patterns (e.g., gendered words dominating local contexts)?

- Have you compared ProtoT’s interpretability to recent mechanistic interpretability methods (e.g., sparse autoencoders on transformer residual streams)?

### Soundness
2

### Presentation
3

### Contribution
2
