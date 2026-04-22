# CAD-Tokenizer: Towards Text-Based CAD Prototyping via Modality-Specific Tokenization

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 2

## Abstract
Computer-Aided Design (CAD) is a foundational component of industrial prototyping. 
where models are defined not by raw coordinates but by construction sequences such as sketches and extrusions. 
This sequential structure enables both efficient prototype initialization and subsequent editing. 
Text-guided CAD prototyping, which unifies Text-to-CAD generation and CAD editing, has the potential to streamline the entire design pipeline. 
However, prior work has not explored this setting, largely because standard large language model (LLM) tokenizers decompose CAD sequences into natural-language word pieces, failing to capture primitive-level CAD semantics and hindering attention modules from modeling geometric structure.
We conjecture that a multimodal tokenization strategy, aligned with CAD’s primitive and structural nature, can provide more effective representations. 
To this end, we propose CAD-Tokenizer, a framework that represents CAD data with modality-specific tokens using a sequence-based VQ-VAE with primitive-level pooling and constrained decoding.
This design produces compact, primitive-aware representations that align with CAD’s structural nature. 
Applied to unified text-guided CAD prototyping, CAD-Tokenizer significantly improves instruction following and generation quality, achieving better quantitative and qualitative performance over both general-purpose LLMs and task-specific baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel framework for unified text-guided CAD prototyping that jointly handles both Text-to-CAD generation and CAD editing.

The authors propose a primitive-level VQ-VAE pooling strategy to compress sketch–extrusion pairs into discrete, compact and semantically meaningful tokens.

These tokens are aligned with the LLM’s embedding space via lightweight adapters, enabling efficient fine-tuning.

Additionally, a finite-state automaton(FSA) guides decoding to enforce syntactic validity.

Experiments show that CAD-Tokenizer compresses and reconstructs sequences well across quantitative metrics. Ablation study proves the effectiveness of designed FSA-based sampling.

### Strengths
1. The proposed primitive-specific pooling layer is thoughtfully designed and its effectiveness is convincingly demonstrated through the reconstruction experiments reported in Table 2.

2. The evaluation across different levels of primitives—curves, loops, and single primitives—in Table 2 and Figure 4 clearly illustrates the trade-off between reconstruction quality and compression rate, making this key insight readily interpretable.

3. The utility of the finite-state automaton (FSA) is well supported by the ablation study in Table 4, which compellingly demonstrates its contribution to overall performance.

### Weaknesses
While the technical contribution of the paper is sound, the writing suffers from several issues that hinder readability and clarity:

Major writing weaknesses:

1. The formula at Lines 191-194 is really confusing. 

Several notational and conceptual ambiguities:

- The term $\mathbf{l}_i$ on the right-hand side of the EMD is not explicitly defined; while it likely represents a target distribution or point set at step $i$ in a sequential modeling context, this assumption requires clarification.

- The notation { $p'$ } (set-bracketed) is weird, which are typically indexed without set notation. And its subscript seems to be missing.

- The definition of $E$ (the superscript of $T$) is unclear.

- What is $t_{1,i-1}$?

2. The reviewer recognizes that the main technical contributions of the paper are the primitive-specific pooling layer and FSA-based sampling, supported by thorough experiments on these components.

However, the details of the proposed primitive-specific pooling are entirely placed in the Appendix. The reviewer believes that this component should be elaborated in the main body of the paper.

> Meanwhile, the Abstract and Introduction emphasize the main contribution as the unification of Text-to-CAD generation and CAD editing. The reviewer does not observe an obvious technical novelty that directly enables this unification. (Notably, the primitive-specific pooling layer demonstrates its value by enabling a trade-off between reconstruction quality and compression rate, while FSA-based sampling appears to significantly reduce the Invalidity Ratio.)

Minor writing weaknesses:

1. The abbreviation “FSA” is introduced in line 156 without prior definition, which may confuse readers encountering the term for the first time.

2. Lines 88–89 introduce the finite-state automaton (FSA) for the first time, yet it is neither mentioned in the abstract nor in any preceding section.

3. Some recent works on CAD generation are missing from the related work section. LLM-based methods are evolving rapidly, including a more comprehensive coverage of recent work would strengthen the paper’s survey:

- CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM, Xu et al. (https://arxiv.org/abs/2411.04954)

- CAD-GPT: Synthesising CAD Construction Sequence with Spatial Reasoning-Enhanced Multimodal LLMs, Wang et al. (https://arxiv.org/pdf/2412.19663)

- CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric Reward, Guan et al. (https://arxiv.org/abs/2505.19713)

- CADmium: Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design, Govindarajan et al. (https://arxiv.org/abs/2507.09792)

### Questions
The reviewer acknowledges the technical contribution of primitive-specific pooling layer and FSA-based sampling. 

However, if the authors insist that the main focus of the paper is the “unification of Text-to-CAD generation and CAD editing”, the reviewer would have some essential concerns/questions:

1. Are the primitive-level VQ-VAE tokenizer and FSA-based sampling genuinely helpful for the **CAD-Editing** task? If so, how can this be demonstrated through ablation studies?

2. Is the "unification of Text-to-CAD generation and CAD editing" beneficial to both **CAD-Editing** and **Text-to-CAD generation** if we exclude the improvements introduced by the primitive-level VQ-VAE tokenizer and FSA-based sampling?  How to prove it through experiment?

Some additional important questions:

1. In Figure 2(4), why is the instruction fed into the Encoder+Adapter? In contrast, in Figure 2(1), only the CAD sequence is fed into the encoder.

2. Why does Figure 2(1) include “PRIM. TOKENS(RAW)”? It appears that both the VQ latent tokens and raw primitive tokens are input to the decoder for sequence reconstruction—can this be clarified?

3. Why does Figure 2(2) also include “PRIM. TOKENS(RAW)”? What does the arrow labeled “For Training” signify?

4. Although FSA-based sampling's ablation study is convincing, the reviewer is curious about why IR is not an exact 0% because invalid sequence may not be sampled due to masking in Figure 8 from Appendix.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a unified LLM-based model for text-driven generation and editing of Computer-Aided Design (CAD) models. The authors design a CAD VQ-VAE that compresses CAD operations into compact representations, which can tokenize CAD sequences into discrete codes. In addition, they also introduce linear adapters that align VQ embeddings with the LLM through an additional training phase. Experimental results demonstrate that the proposed method achieves state-of-the-art performance.

### Strengths
The paper demonstrates thoughtful design of the CAD tokenizer to enable seamless integration with the LLM framework, effectively bridging the gap between continuous CAD representations and discrete language model inputs.

### Weaknesses
1. What are the typical lengths of CAD sequences in your dataset? I have concerns about the scalability of the proposed pipeline. While the transformer encoder can handle variable-length inputs and capture relational information, the primitive pooling operation fuses all tokens within each primitive into a single latent vector before VQ encoding. As CAD sequences grow longer, the number of operation types remains constant, but the complexity of inter-operation relationships increases significantly. It is unclear whether this pooling-then-quantizing approach can adequately capture these increasingly complex relationships in longer sequences.
2. Each extrusion operation contains precise parameters such as translation and rotation values. However, the VQ tokenizer maps continuous embeddings to a finite discrete codebook, which may result in quantization loss. This raises concerns about whether fine-grained geometric details are preserved, potentially leading to inaccuracies in the reconstructed parameter values.

### Questions
Please refer to the weaknesses.

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
3

### Summary
This paper introduces CAD-Tokenizer, a unified framework for text-based CAD prototyping that integrates both Text-to-CAD generation and CAD editing within a single model. The authors argue this joint formulation better reflects real industrial design workflows that are inherently iterative.

The core contribution is a tokenization method that replaces standard LLM tokenizers. Using a VQ-VAE, CAD operations are converted into discrete primitive-level tokens, yielding representations aligned with the structural and semantic nature of CAD data. An LLM is fine-tuned on CAD tokens to perform both generation and editing. This joint training improves performance and computational efficiency.Finite-State Automaton is used during decoding to enforce syntactic validity, ensuring outputs adhere to the CAD grammar. Experiments show CAD-Tokenizer outperforms task-specific baselines (CADFusion, CAD-Editor) and general models like GPT-4o across diverse metrics, including geometry, instruction following, and human evaluation.

### Strengths
Unifying CAD generation and editing is an important conceptual step that aligns research with practical design workflows.

The insight that standard tokenizers fail to capture CAD’s structured semantics is compelling, and the proposed primitive-aware tokenizer addresses this gap effectively.

The combination of VQ-VAE tokenization, adapter-based embedding alignment, and FSA-guided decoding is well-justified. Ablation studies clearly demonstrate each component’s benefit.

The experimental design is rigorous, using rich metrics and strong baselines. Results show consistent and convincing gains.

### Weaknesses
Due to limited open-source CAD datasets, experiments focus on simple shapes. The model struggles with complex geometries requiring advanced spatial reasoning.

Standard distributional metrics (COV, JSD) are shown to be unreliable for editing tasks, as they can reward unchanged outputs. Although alternative VLM and human metrics mitigate this, the issue underscores a broader challenge in evaluation.

Additionally, some errors stem from the LLaMA-3-8b model’s limited spatial reasoning, capping overall performance.

### Questions
- How does the tokenizer’s 2048-codebook design scale with increased CAD complexity? Would a larger codebook affect alignment or fine-tuning efficiency?
- How adaptable is CAD-Tokenizer to other CAD formats beyond Sketch-and-Extrude Modeling? Would retraining be necessary or could components be reused?
- What types of geometric errors (e.g., self-intersections, open sketches) persist despite FSA-based enforcement?
- Can you provide concrete comparisons of training or inference time relative to Vanilla-LLaMA?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Authors proposed a new tokenizer to unify text-based CAD generation / editing. More specially they first train VQ-VAE to represent CAD sketch and extrude as CAD-specific DSL tokens. They then align those token embeddings with pretrained LLM tokens, effectively converting DSL tokens into LLM text indices. Then the base LLM is SFT on the new sequence of token indices. Authors demonstrate good results compared to using HNC-cad and the default BPE tokenizer.

### Strengths
(1) The proposed VQ-VAE approach with max pooling is more token efficient than HNC-CAD.

(2) Aligning CAD DSL to pretrained LLM embedding is interesting and works surprisingly well.

(3) Authors demonstrate better result using the new LLM aligned CAD tokenizer.

### Weaknesses
(1) Sketch seqence and extrude representation largely follows previous work. Authors should make this more obvious as opposed to categorying those relevant works as "classical cad tasks". The CAD primitive-level parsing is actually a more compressed verison of hnc-cad with masking and single VQ-VAE as opposed to multiples. 

(2) The main comparison to hnc-cad is not sound. hnc-cad encodes deepcad data into a sequence of discrete codes. So the claim that it is not designed for LLM training is not fair. Anything that representat data as discrete indices can be used for LLM training. In fact an autoregressive Transformer was trained from scratch for hnc-cad. Perhaps authors should more clearly state that a pretrained LLM was not trained on those discrete codes in that paper. But it should be easy to train one for ablation study, which the author did not do.

(3) The main comparison to BPE tokenzier is not correct. Authors directly apply text-based tokenizer on the parsed sketch-and-extrude sequence, which is very abstract dsl and different from text. Some works might have did this but it is very obviously bad practice. In this case the performance could be bad because the compatiblity issue with data format and pretrained LLM model, not because of the BPE tokenizer. A more fair comparision is to convert CAD DSL into text like CADQuery, JSON, or python-like code. Then after finetuning a LLM or Code-LLM with default text tokenizer and test if the model understand CAD constructions. Actually there are already many works in this direction (cad-llama) and has proved that this works well with low invalid ratio. This should be much better than the 88% BPE invalid ratio reported in the paper. 

(4) Failure to compare to more recent text-based CAD generation / editing works such as CAD-Llama, CAD-MLLM and etc. Text generation and editing visual results also look much more simplier than those in related works. It will be more convincing to show that taken out the default text encoder in those models and switching to the one from the paper also leads to improvements. 

(5) Unclear if the alignment step is still necesary if the LLM vocabulary was expended to include the new CAD tokens, and LoRA finetuning uses larger rank and also update the embedding and logits layer.

Overall, when it comes to something as fundemental as tokenizer, I hope the authors can be more careful about the comparision and make sure it is not a model or training issue that affects the final result. Right now it lacks comparsion to text-based CAD generation works and some important settings are not sounds. I feel liket this will have a high impact on the community. If the claims are true it will basically disencourge future works to directly train on DSL or text / code data paired with default text tokenizer. This is a huge change and should be better validated.

### Questions
(1) Please make fair comparison to hnc-cad and espeically other text-based CAD generative models. Directly applying text tokenizer on abstract DSL is not a sensible comparison. BPE baseline should be considered in the setting when CAD DSL is converted to text or python code etc.

(2)  After finetuning, would the LLM still maintain its original text generation and reasoning capability? Have authors did any ablation study on this? This could potentially be a huge disadvantage for this approach compared to text-based CAD generation, since for them the grammar and code structured are aligned with the pretrained model. 

(3) Visual editing results are very simple, much more simple than what I am seeing in other works using pretrained LLM. I highly suggest authors compare to those more recent methods (cad-llama, ca-mllm etc).

(4) Have authors tried different LoRA settings? Recent experiments show that when data distirbution is different from training, it is ususally beneficial to use larger rank and sometimes full training is even better.

### Soundness
1

### Presentation
3

### Contribution
2
