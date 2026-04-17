# Decoding Layer by Layer: Uncovering Hierarchical Reasoning in Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Decoder-only language models, such as GPT and LLaMA, generally decode on the last layer. Motivated by humans' hierarchical reasoning capability, we propose that a hierarchical decoder architecture could be built with different layers decoding texts in a streaming manner. Due to limited time and computational resources, we choose to adapt a pretrained language model into this form of hierarchical decoder. Language heads of the last layer are copied to different selected intermediate layers, and fine-tuned with different task inputs. By thorough experiments, we validate that these selective intermediate layers could be adapted to speak meaningful and reasonable contents, and this paradigm of hierarchical decoder can obtain state-of-the-art performances on multiple tasks such as hierarchical text classification, classification-guided generation, and hierarchical text generation. HdLM outperforms all baselines on WoS, DBpedia, ESConv, EmpatheticDialogues, AQuA, CommonSenseQA, and several cognitive tests. We also provide a thorough theoretical analysis to validate the convergence and computational savings of our methodology. This study suggests the possibility of a generalized hierarchical reasoner, pretraining from scratch. Our code and model can be found on https://anonymous.4open.science/r/HdLM-2119.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper discusses work on optimizing LLMs to exploit hierarchical reasoning through layer-specific probing. The call this approach HdLM. The authors claim that hierarchical reasoning is grounded from how humans reason sequentially through conceptual abstractions and sequential determination. The authors explore this setup across different tasks including hierarchical text classification (HTC), classification-guided generation (CG), and hierarchical text generation (HTC). Results show that HdLM obtains better performance across BERT-based and retrieval-based methods for HTC from WoS and DBPedia benchmarks, better performance than prompting-based optimizations like vanilla CoT, SelfRefine, direct, and FSM for CgC from ESConv and EmpatheticDialogues, and better performance against commercial models for HTG. The authors also provide ablation experiments on effects with model scale and layer selection which also supports the effectivity of the approach.

### Strengths
The paper is readable and tackles and interesting and timely work on probing the hierarchical reasoning for LLMs and its effectiveness across reasoning-based tasks. This is an important research topic where this work can shed light on specific knowledge encoded in an LLM’s layers and how this can be exploited for better reasoning capabilities. Technicality-wise, I appreciate the thorough ablations such as the layer-wise effects, model scale performed by the authors to support effectiveness of their approach. Showing that the method works across these variables is an advantage.

### Weaknesses
The paper can benefit from a better framing and motivation. The authors mention drawing motivation from how humans use hierarchical reasoning but the supporting information provided are limited (lines 42-48). What specific tasks or examples can you provide that draws from both the “multi-timescale abstraction” and “sequential determination”? The figure 1 should be anchored on these two aspects. Likewise, the authors should reference interdisciplinary literature from cognition and psychology (which I believe is lacking from the current paper) since the work claims to be inspired by human paradigm of sequential reasoning and abstraction.

The related work at the very end is weak and uninsightful despite the rich literature in hierarchical LLMs. Please improve the depth of the search and connect literature on effectivity of probing LLM layers across tasks, efficiency of hierarchical transformer models, and applications of hierarchical encoder/decoder layers. As mentioned above, the paper seems to be missing a crucial connection with cognitive science literature if the authors are intent on grounding the motivation to the way humans reason hierarchically.

One main issue is that the discussion and results seem very surface-level and limited in insights. For example, one of the most important part is analyze which layer works with respect to task, however, the paper does not convincingly discusses **why** the these specific layers work and whether these changes for specific tasks, optimization setup, model architecture or model scale. What’s with earlier layers that seem to be very poor across both tasks? Does this mean earlier layers do not encode information? How about mid to later layers? Does converging performance mean they encode semantic information better than earlier or later layers? How do these tie to the type of knowledge required by tasks  WoS and ESConv? All these should be discussed succinctly in the paper but are currently missing.

One novel improvement that can potentially strengthen the paper technically is an automatic search-based algorithm for selecting the minimum number of layers that can still reach the highest performance. In 4.4, the authors mention selecting layers as much as 25 and 28 for each task but this seems to be excessive and unintuitive. This way, you do not have to go through all layers and maybe start intuitive such as with middle layers based from ablation experiments.

Minor: “Motivated by both mechanisms, models with hierarchical reasoning capabilities have recently been explored, either implementing on (i) (Barrault et al., 2024; Wang et al., 2025) or (ii) (Yang et al., 2024; Cai et al., 2024).” - On what? This is a hanging sentence and leaves readers confused. Please provide succinct supporting information from the cited references.

Minor: Please fix citations: “(Jason Wei, 2022)”, “Llama (AI@Meta, 2024)”, “ (Eric Zelikman, 2022b)”

### Questions
In Figure 1, the figures show exact same outputs between conventional LM vs. HdLM. While I do get the idea that the authors are relaying, I was confused at first why they are the same and not slightly different since you are getting outputs from various layers for HdLM. The authors should revise this figure just to ensure that readers will not assume that the output from a vanilla LM is exactly the same as HdLM.

The layer-wise analysis shows almost no difference in performance for both tasks across mid to later layers for WoS. Do you think the difference by selection layer 25 as the best one compared to layers 10 to 31 statistically significant? How about for the layers for ESConv task?

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
4

### Summary
The paper introduces hierarchical decoding language models (HDLM), a post-hoc adaptation of decoder-only LLMs that replicates the final language head onto selected intermediate layers and trains those layers to decode intermediate responses.

### Strengths
The authors introduce the interesting setting of hierarchical decoding and present a simple, broadly applicable recipe that augments existing decoder-only LLMs without architectural redesign. The core mechanism of replicating the final language head onto intermediate layers and applying multi-layer decoding losses, turns a standard transformer into a model capable of producing multi-stage responses. This idea is well-motivated by prior observations that intermediate layers carry semantically rich representations and is implemented cleanly as a post-hoc hierarchical supervision method, requiring no retraining from scratch.

The evaluation spans a wide spectrum of benchmarks in hierarchical text classification (WoS, DBpedia), classification-guided dialog generation (ESConv, EmpatheticDialogues), and hierarchical reasoning/generation tasks (AQuA, CommonSenseQA, ToM). Across these diverse settings, HdLM consistently improves over supervised fine-tuning (SFT) and other baselines. 

The theoretical FLOPs analysis is carefully derived and linked to empirical timing measurements. The paper explicitly computes training and inference complexity and demonstrates measurable compute savings that scale with the intermediate-layer index k and output length, aligning well with theoretical expectations.

### Weaknesses
The novelty is somewhat overstated. Replicating language heads onto intermediate layers for auxiliary supervision is conceptually close to existing multi-layer or hierarchical decoding methods, but no direct comparison is provided. Without including baselines like Hierarchical Skip Decoding (https://arxiv.org/abs/2404.16710) or intermediate-layer probing (https://arxiv.org/abs/2407.10795) variants, it’s hard to assess the relevance of the proposed method that requires task-specific tuning.

Further, the paper doesn’t clearly explain early on how intermediate-layer decoding aligns with the token-level autoregressive process. For example, Figure 1 is more confusing than helpful. One has to dig deep into the paper and learn how to parse complicated notation in order to understand what is going on.

The “proof” in Section 3.2 on convergence feels disconnected from the rest of the paper. It gestures toward theoretical depth but doesn’t establish any practically meaningful guarantee or insight about HdLM. The argument mostly reuses an existing theorem and reformulates it with different notation, without connecting it back to model behavior or experiments. A more focused empirical or ablation-based discussion of convergence behavior would be far more valuable.

Also, removing Section 3.2 would be a good option to create space for the related work section, which is far too short and reads more like a placeholder than a serious engagement with prior literature. Given the number of existing studies on intermediate-layer decoding, hierarchical supervision, and multi-level reasoning, this omission stands out. The paper would benefit from a more comprehensive positioning — not only citing prior work but actually clarifying where HdLM diverges conceptually and empirically. As it stands, this section undersells the novelty and gives the impression that the authors have not fully mapped the landscape.

The provided implementation and reproducibility details, while commendable, appear incomplete. The public repo’s folder structure doesn’t match the README, and the src directory required by the training script is missing. This undermines the otherwise strong claim of reproducibility. Overall, while the approach is well-motivated and shows consistent improvements, the paper needs clearer baselines, a more transparent decoding description, and complete reproducibility materials to reach the standard of a strong ICLR paper.

### Questions
How does the intermediate decoding process intuitively map to the partial responses in hierarchical decoding? A clear, high-level explanation in words (without equations) would help readers understand how the mechanism relates to standard autoregressive decoding.

Including baselines such as LayerSkip (https://arxiv.org/abs/2404.16710) would significantly strengthen the paper and directly improve my evaluation of its contribution. These comparisons are critical to demonstrate that HdLM provides a distinct advantage over existing adaptive or hierarchical decoding methods.

The related work section should be expanded substantially. Works on layer skipping, adaptive compute for decoding https://arxiv.org/abs/2312.16392, and mechanistic interpretability studies on intermediate decoding (https://arxiv.org/abs/2401.06102) are directly relevant. Addressing these connections would materially improve the paper and my rating.

Further fixing the code submission and addressing my raised weaknesses would improve my impression about the work.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Hierarchical decoding Language model (HdLM), a new architecture designed to embed hierarchical reasoning capabilities directly into a language model. Motivated by human cognition—which moves from coarse-grained strategy to fine-grained action—the method adapts a pre-trained decoder (Llama3-8B) by copying its final language head to selected intermediate layers. These new heads are then fine-tuned to decode specific, intermediate steps of a complex task in a streaming, layer-by-layer fashion.
The authors evaluate HdLM on three categories of hierarchical tasks: Hierarchical Text Classification (HTC), Classification-guided Generation (CgG), and Hierarchical Text Generation (HTG). The empirical results are strong, showing that HdLM achieves state-of-the-art performance on multiple benchmarks and, notably, outperforms much larger models like GPT-4 on specific reasoning tasks. The paper also provides a theoretical analysis of the method's computational efficiency gains during training and inference.

### Strengths
- The paper's primary contribution is moving beyond prompting (like CoT) or latent-space manipulation. It introduces a powerful structural inductive bias for reasoning by physically modifying the model's architecture. Aligning computational depth  with conceptual abstraction  is a fundamental and intuitive idea that provides a more native solution for multi-step tasks.
- The experimental results are highly compelling. The fact that an 8B-parameter HdLM can outperform GPT-4 on complex reasoning (ToM tests) is a significant finding. It provides a strong argument that intelligent architectural design can be more effective and efficient than brute-force scaling, a highly relevant conclusion for the field.

### Weaknesses
- The method's design fundamentally requires the number of hierarchical steps (D) to be strictly less than the total model layers (K). This hard constraint (D < K), makes the model incapable of handling tasks with long, variable, or arbitrarily deep reasoning chains (e.g., D > K). This is a scenario where traditional CoT methods, which decode all steps at the final layer, remain far more flexible.
- The model's performance hinges critically on two sets of hyperparameters: the intermediate layer indices (k_d) and their corresponding loss weights (f_d). The paper provides no systematic, principled method for selecting these. The sensitivity analysis is limited to D=2, and the choice for D=3 (k=[20, 30]) appears entirely empirical. This reliance on expensive, task-specific manual tuning severely limits the method's scalability and generalization to new, unseen tasks.
- The HdLM architecture enforces a strict, unidirectional (shallow-to-deep) information flow.  This makes the model highly fragile: any error generated at an early layer (k_1) is irreversibly passed on and likely amplified at subsequent layers. This is a critical flaw that is masked by the paper's experimental design. By evaluating only on shallow tasks (D=2 or D=3), the impact of this error accumulation is not apparent. It is highly probable that on deeper, long-chain reasoning tasks (e.g., D=10), HdLM's performance would collapse due to this cumulative error, potentially performing much worse than a more robust CoT baseline that theoretically allows for self-correction.
- The paper is critically ambiguous about its core mechanism, hindering reproducibility. It fails to clearly explain how information is passed between layers. It's unclear how a newly decoded discrete text r_d is integrated with a continuous hidden state e_d for processing by subsequent layers. The (presumed) mechanism—that the sequence length (L) of the hidden state grows ([L_prev + L_d, E]) while the hidden dimension (E) remains constant—is never explicitly stated.

### Questions
- For tasks with D > 2, is there any systematic or principled method to select the layer indices (k_1...k_{D-1}) and loss weights (f_1...f_{D-1})? Or is this purely a matter of empirical, task-specific tuning?
-  Please explicitly confirm the information flow mechanism during inference. Is it correct to assume that the hidden states e'_d (shape [L_d, E]) corresponding to the decoded tokens r_d are concatenated with the input hidden states e_d (shape [L_prev, E]) to form a new, larger tensor of shape [L_prev + L_d, E], which is then fed as input to the next block of layers M_{k_d:k_{d+1}}?
- How do the authors justify the D < K hard limit versus the flexibility of CoT?
- Given that experiments were run only on shallow (D=2, 3) tasks, what evidence or argument do the authors have that HdLM would not suffer from catastrophic performance collapse on deeper tasks (e.g., D=10) due to the error propagation inherent in its unidirectional architecture?
- Given the high risk of error propagation, have the authors considered any mechanisms to improve robustness, such as a feedback loop that allows a deeper layer's state to correct or refine the output of a shallower layer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper propose Hierarchical decoding Langauge Model (HdLM), which enables different intermediate transformer layers to decode text hierarchically. Pretrained model can be adapted to HdLM by copying the language heads of the last layer to different selected intermediate layers, and fine-tunining with different task inputs. The paper evaluates HdLM’s performance on three different tasks: hierarchical text classification (HTC), classification-guided generation (CgG), and hierarchical text generation (HTG). HdLM outperforms baselines on multiple datasets including DBpedia, AQuA, and CommonSenseQA in some metrics.

### Strengths
- The idea of enabling intermediate layers to decode is creative and well-motivated by human hierarchical reasoning capabilities.
- The paper experiments with multiple datasets, and achieve performance that can surpass GPT-4 + CoT / SimTom with a much smaller model in HTG tasks.
- The paper conduct ablation on the sensitivity in the intermediate decoding layers choices, scalability on model sizes, and computational time.

### Weaknesses
- The automatic evaluation metrics used in this paper (BLEU and Rouge-L) are more reliable when evaluating tasks with significant lexical overlap, but might not be appropriate to serve as a quality metric in this paper’s setting. Furthermore, the absolute values of BLEU-2 scores are very low (<6). Although the performance of the classification task is higher, I am not convinced that the improvement in empathetic dialogues is significant. I would love to see LLM-as-a-judge / human evaluation results.
- Although theoretically speaking, this framework can be adapted to depth D that < K, this also requires the users to know the value of D, and need to tune the selected layers accordingly. Based on figure 4,  the performance is sensitive of the intermediate layer choices. However, the paper provides no principled selection method, making HdLM significantly more difficult to apply than SFT in practice.

### Questions
- Quotes on line 37: try `` and ''
- Last line of page 1 doesn’t look like a finished sentence?
- In Table 1, SFT’s MaF1 is better than HdLM’s MaF1
- In Table 2, SFT+FSM on B-2 is actually the best, rather than +HdLM
- Why isn’t SFT+Self-Refine tested in table 2? Direct+Self-refine is the best performing on for direct settings in terms of ACC and MaF1.
- How does the training converge?

### Soundness
2

### Presentation
3

### Contribution
3
