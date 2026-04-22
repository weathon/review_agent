# A circuit for predicting hierarchical structure in-context in Large Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Large Language Models (LLMs) excel at in-context learning, the ability to use information provided as context to improve prediction of future tokens. Induction heads have been argued to play a crucial role for in-context learning in Transformer Language Models. These attention heads make a token attend to \emph{successors} of past occurrences of the same token in the input. This basic mechanism supports LLMs' ability to copy and predict repeating patterns. However, it is unclear if this same mechanism can support in-context learning of more complex repetitive patterns with hierarchical structure or contextual dependencies. Natural language is teeming with such cases. For instance, the article \texttt{the} in English usually prefaces multiple nouns in a text. When predicting which token succeeds a particular instance of \texttt{the}, we need to integrate further contextual cues from the text to predict the correct noun. If induction heads naively attend to all past instances of successor tokens of \texttt{the} in a context-independent manner, they cannot support this level of contextual information integration. In this study, we design a synthetic in-context learning task, where tokens are repeated with hierarchical dependencies. Here, attending uniformly to all successor tokens is not sufficient to accurately predict future tokens. Evaluating a range of LLMs on these token sequences and natural language analogues, we find adaptive induction heads that support prediction by learning what to attend to in-context. Next, we investigate how induction heads themselves learn in-context. We find evidence that learning is supported by attention heads that uncover a set of latent contexts, determining the different token transition relationships. Overall, we not only show that LLMs have induction heads that learn, but offer a complete mechanistic account of how LLMs learn to predict higher-order repetitive patterns in-context.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates whether induction heads in LLMs can handle hierarchical, context-dependent patterns, beyond simple token copying.
Using synthetic tasks with multi-level repetitive structures, the authors show that adaptive induction heads learn to attend to contextually correct successors, supported by context-matching heads that route latent contextual information.
The findings are validated on both synthetic and natural-language data, and replicated across multiple open-source LLMs, offering a mechanistic account of how LLMs learn hierarchical patterns in-context.

### Strengths
- The focus on phenomena unexplained by existing induction head studies is highly interesting. The idea itself is novel and compelling.
- The task design is excellent. The hierarchical synthetic sequence task is well thought out and captures the essence of the question.
- Section 3.2 is particularly strong: the authors go beyond purely synthetic tasks and demonstrate the same phenomenon on natural language data. This direction, connecting toy-task findings to real-world data, is both rare and extremely important for the field of mechanistic interpretability.

### Weaknesses
- Figure 1 is hard to understand. It’s unclear what token the model is predicting when the attention map is shown,
- In Figure 2A, the label “Generic ICL heads” is ambiguous; it’s not explained what metric or value this represents.
- The statement in Line 213 (“Induction heads that learned in-context were located in later layers, whereas induction heads that did not show signs of learning were embedded in earlier layers.”) is not supported by quantitative evidence.
    - Figure 2B only shows two examples, which are insufficient to substantiate the claim.
- The task description lacks clarity on how the subset V of tokens was chosen.
It’s also unclear whether sampling introduces variance in results. For instance, whether differences arise when using familiar vs. rare tokens.

### Questions
About the task setup.
- What subset V of tokens is used for the experiments?
Also, does the sampling choice affect the results? For example, are there differences when using familiar tokens or low-frequency tokens?

About Figure 2A
- In Figure 2A, there seem to be differences in induction headness depending on model size.
Do the authors have any interpretation or hypothesis regarding this trend?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether attention heads in pretrained LLMs can perform context-sensitive in-context learning on synthetic hierarchical token sequences. The authors try to identify induction heads that copy past successors and describe a variant, “adaptive induction heads”, which appear to select which past successor to copy depending on latent chunk context. They present linear probes that decode chunk identity from certain heads and ablations, showing that removing those “context-matching” heads reduces both induction-head context-sensitivity and downstream accuracy on the synthetic tasks.

### Strengths
1. **Main idea:** The paper targets a mechanistic question: how in-context learning for hierarchical structure is implemented, which is worth studying.  
2. **Analysis:** The combination of attention inspection, probing and ablation is the right toolbox for causal mechanistic claims.

### Weaknesses
1. **Main conceptual weakness** I find it difficult to understand why the task presented in the paper requires more complex circuits than standard induction heads to solve. In particular, I do not see why the task cannot be viewed simply as a higher-order in-context n-gram or associative recall task, for which the standard induction head circuit is already known to be sufficient (see, for example, [1]).
2. **Paper organization** The paper could be more clearly structured, what is the precise question the authors intend to investigate in this work ? The setup and results appear to be mixed in the section maybe clearly separating the setup and analysis from the results would help clarity of the manuscript. 
3. **Undefined term  adaptive induction head.** The manuscript repeatedly uses this term but never gives a precise definition or scalar metric for deciding when a head is adaptive or not.
4. **Task description and data generation under-specified.** Task description and data generation are under-specified.** The parameters of the synthetic data generator are scattered across the main text and appendix, with no concise pseudocode or illustrative example showing precisely how `V`, `P`, `P'`, and the chunk composition interact. It is also unclear how the next symbol in the sequence is generated. Is it a deterministic function of the current chunk, or is it sampled probabilistically?
5. **Induction-head identification.** The paper mentions a conventional “score matching” but no more details are provided on what is this procedure and how is it used in the analysis. 
6. **Probe methodology lacks detail.** It is not stated whether probes use pre/post LayerNorm residuals or per-head `z` values, what regularization and cross-validation were used, or how stable probe results are across seeds. Only reporting a single accuracy number is insufficient.  
7. **Ablation implementation ambiguous.** “Zeroing the output of self-attention” is vague: which tensor exactly is zeroed, at what point in the residual/LayerNorm pipeline, and whether this induces side effects? More targeted ablations are needed to localize the mechanism.  
8. **Organization and terminology.** Important methods are in the appendix or omitted; the relationships between “context-matching heads”, “induction heads”, and “adaptive induction heads” should be made explicit to clearly understand the difference.  


[1] Varre, A., Yüce, G., & Flammarion, N. (2025). Learning In-context n-grams with Transformers: Sub-n-grams Are Near-stationary Points. arXiv preprint arXiv:2508.12837.

### Questions
1. **Definition of adaptive induction head:** Could the authors provide a definition of the different kind of  induction heads mentioned in the paper and how are the respective circuits are constructed? could the authors better explain why the adaptive induction head would solve the task and the standard induction head does not?
2. **Score matching:** Could you provide more details about the score matching procedure  
3. **Probe inputs & hyperparameters:** Which representation do probes take (per-head `z`, residual pre/post-LayerNorm, etc.)? could you give more details about the experiments with pribing in general  
4. **Ablation specifics:** Which tensor is zeroed (name & shape), and at what point in the forward pass is it zeroed? Did you confirm that the decoded chunk signal disappears at the ablation site?  
6. **Generality:** Do the findings hold when permutations include noise, partial matches, or variable chunk lengths rather than strict repeats? This would indicate whether the circuit handles realistic variation or only strict synthetic structure.  
7. **N-gram question** could the author demonstrate empirically why a simple 2-layer multi-head circuit, where the first layer copies all the tokens in the chunk and the second layer does the matching it is not sufficient to solve their task ?
8. **Latent context** Could the authors clarify  what is a latent context in their task and better define equation 1

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies whether induction heads in pre‑trained LLMs can learn in‑context which past successors to attend to when sequences exhibit hierarchical repetition.

### Strengths
1. Clear task construction for hierarchical repetition and explicit definitions of 2nd/3rd‑order contexts with concrete generation parameters.
2. Mechanistic proposal that context‑matching heads route latent context info (Eq. 1; Fig. 3–4) that induction heads then use; single‑head ablation illustrates directional dependency (Fig. 5).

### Weaknesses
1. Group ablations compare targeted sets (all induction heads; all context‑matching heads with decodability > 85%) to random sets of equal size (non‑induction; ≤ 55% decodability). This ignores known structure: adaptive heads sit later in the stack (near logits) and context decodability rises in later layers (Fig. 2B, Fig. 3). 

2.Fig. 2A reports the averages of the best performing 5 heads.
 Selection and evaluation happen on the same 32 sequences, so curves can be spuriously high.
 There is no hold‑out for head selection, no distributional summary across all candidate heads, and no error bars.

3.  The synthetic sequences are perfectly periodic and equal-length, so absolute and relative positions are tightly correlated with the intended successor relationship.
The context-matching heads (e.g., 1-back, N-back patterns in Fig. 4) are exactly the type of positional heads described in "Rethinking Associative Memory Mechanism in Induction Head" in COLM2025—their attention maps reflect deterministic offset patterns, not learned context association.

### Questions
1. Could you select control heads by comparable output-norm magnitude or layer position, rather than purely random sampling, to remove confounds due to layer depth and activation scale?

2. Is it possible to use split-fold evaluation: performing K-fold or hold-out validation so that head selection (score computation) and evaluation (accuracy measurement) use disjoint folds, mitigating winner’s-curse bias.

3. Could you provide Non-periodic / length-jittered controls: running the experiments on non-periodic or length-randomized sequences to exclude positional-periodicity artifacts, as reported in "Rethinking Associative Memory Mechanism in Induction Head"?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a key gap in understanding Large Language Models (LLMs)’ in-context learning (ICL): traditional induction heads fail to handle hierarchical, context-dependent repetitive patterns. I think this paper has three main contributions. 
- Designing a Rational Synthetic Setup: It constructs hierarchical synthetic token sequences (1st-order: simple repetition; 2nd-order: random switching of base chunks; 3rd-order: recombination of 2nd-order chunks) to isolate "hierarchical dependency" as a variable. This setup avoids noise from natural language, providing a pure testbed to precisely validate LLMs’ ability to handle complex repetitive structures—laying the foundation for subsequent mechanism discoveries.
- Discovering Adaptive Induction Heads: Unlike static induction heads (early layers, uniform successor attention), adaptive induction heads (later layers) learn to select successor tokens based on latent context (e.g., 2nd/3rd-order sequence chunks). Validated in synthetic sequences and natural language (e.g., disambiguating "San"’s successors), they expand induction heads’ capability beyond simple bi-gram patterns.
- Proposing a Collaborative Circuit: It identifies "context matching heads" (encode latent context, with >90% accuracy for 2nd-order chunk decoding) that work with adaptive induction heads. Ablation experiments confirm causality—zeroing out context matching heads eliminates adaptive heads’ context-aware selection, while random head ablation has little impact—providing the first mechanistic explanation for hierarchical ICL.

### Strengths
This paper’s key strengths lie in its extensions and rigor:  it builds on prior induction head research—moving beyond their traditional focus on simple, context-agnostic patterns to explore hierarchical dependencies, thus deepening understanding of LLMs’ in-context learning. It also features a well-designed synthetic setup (hierarchical token sequences of 1st to 3rd order) that isolates core variables, providing a controlled testbed for validating complex pattern processing. Additionally, it makes novel discoveries of adaptive induction heads and their collaborative circuit with context matching heads, while verifying these findings across multiple LLM families (e.g., Qwen2.5, Gemma2-2B), enhancing the reliability and generalizability of its conclusions.

### Weaknesses
Despite the aforementioned strengths, there are doubts regarding the significance of this research direction. The authors’ survey of relatedwork is insufficient, and the study may be overlapped by existing relevant research. For example, a work at ICLR 2024[1] analyzed LLMs’ in-context learning (ICL) capabilities from the perspective of input and output probabilities, identifying a phenomenon of co-occurrence self-reinforcement between tokens within the context. It remains unclear how this phenomenon relates to the problem explored in the current paper—specifically, the multi-token-level self-reinforcement identified in that work seems similar to the 2nd-order and 3rd-order structures investigated herein. Additionally, while the discovery of induction heads in LLMs is certainly important, if further performance analysis shows that the behaviors of various `specialized' induction heads are consistent with the aforementioned self-reinforcement phenomenon, it raises the question of whether additional in-depth exploration in this direction is still necessary.

[1] UNDERSTANDING IN-CONTEXT LEARNING FROM REPETITIONS. ICLR 2024. Yan et al. https://arxiv.org/pdf/2310.00297

### Questions
Q1: Can you elaborate on how your findings will inspire further development of LLMs? 
Q2: What's the connection and different between your work and the aforementioned ICLR work?

### Soundness
3

### Presentation
3

### Contribution
2
