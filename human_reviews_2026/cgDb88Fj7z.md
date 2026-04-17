# Modeling Expert Interactions in Sparse Mixture of Experts via Graph Structures

- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Sparse Mixture of Experts (SMoE) has emerged as a promising solution to achieving unparalleled scalability in deep learning by decoupling model parameter count from computational cost. By activating only a small subset of parameters per sample, SMoE enables significant growth in model capacity while maintaining efficiency. However, SMoE struggles to adapt to distributional shifts, leading to reduced robustness under data contamination. In this work, we introduce SymphonySMoE, a novel family of SMoE that introduces a social graph to model interactions among experts. This graph-based structure enhances the token routing process, addressing the robustness challenges that are inherent in conventional SMoE designs. SymphonySMoE is lightweight, modular, and integrates seamlessly with existing SMoE-based models such as the XMoE and the Generalist Language Model. We provide both theoretical analysis and empirical evidence demonstrating SymphonySMoE's advantages over baseline SMoE. Extensive experiments on language modeling and visual instruction tuning validate our method's effectiveness. We further highlight the scalability of SymphonySMoE to models with 4.2 and 7.4 billion parameters, showcasing its applicability in fine-tuning tasks for large-scale systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SymphonySMoE, an extension of Sparse Mixture of Experts (SMoE). The paper first views routing as a probabilistic graphical model. Next, it incorporates a social graph to model interactions between experts and adjusts the SMoE gating values accordingly. This approach seeks to enhance token routing by promoting the co-selection of expert pairs with high-confidence activations. The authors present a theoretical analysis and empirical evaluation across language modeling (WikiText-103), visual instruction tuning (LLaVA), and GLUE fine-tuning.

### Strengths
- The paper's primary strength is the introduction of a novel social graph framework for modeling expert-to-expert interactions within  SMoE system.
- The paper also presents a strong theoretical analysis that rigorously formalizes the co-selection properties of experts.
- Extensive empirical evaluation on several models and multiple domains.

### Weaknesses
- The paper's probabilistic graphical model is primarily conceptual; the practical method simply uses an adjacency matrix to smooth gating scores, with the PGM adding no material impact to the final routing.
- The paper does not benchmark against other recent advanced routing strategies (see [1] for a list of possible baselines). 
- Across most benchmarks, the reported improvements are modest (e.g., 1–3% absolute in some multimodal tasks, ~0.5–1 perplexity drop in WikiText-103). Without a rigorous statistical significance analysis or evaluation on more challenging datasets, such as mathematical reasoning, it is difficult to conclude that the improvements are meaningful in practice.
- The GLUE experiments are limited to Phi3-SMoE with top-2 selection among 4 experts, which again does not stress test the method’s scalability to larger, more realistic SMoE architectures.
- While the adjacency matrix update is claimed to be lightweight, according to the complexity analysis in Table 5 for large N (e.g., long sequences) or large M, this could become non-negligible.

[1] Do et al. "On the Effectiveness of Discrete Representations in Sparse Mixture of Experts.", TMLR 2025.

### Questions
- How sensitive is the method to the way the adjacency matrix is constructed (e.g., co-activation frequency, normalization, smoothing)?
- What are the results with more experts in the GLUE benchmark?

### Soundness
2

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
This paper introduces SymphonySMoE, a novel MoE routing mechanism that incorporates experts' co-selection information into the routing decision. 
The authors frame the routing process in MoE as a graphical model and provide a theoretical analysis of their design. 
They conduct experiments to demonstrate that this routing design enables MoE to adapt to distributional shifts, leading to more robust routing. 
This approach presents an interesting perspective to MoE routing design, though further empirical validation and improved paper presentation would enhance the overall impact of the paper.

### Strengths
- The idea of using experts' co-occurrence information to provide a smoothing signal for MoE routing is novel. To the best of my knowledge, there is little prior work in this area, making it an interesting contribution.

- The authors provide a therotical analysis to support their idea.

### Weaknesses
- The authors' claim that SymphonySMoE addresses the distributional shifts in traditional MoEs lacks a logical foundation.
I do not see, nor can I understand, any motivation linking SymphonySMoE to this concept of robustness throughout the paper.
I understand that SymphonySMoE uses the mutual information between experts to help MoE routing, but the connection between this mechanism and robustness is unsubstantiated, lacking proper explanation and empirical validation.

-  Some of the author's claims are not adequately supported by experimental evidence, as the experiments suffer from significant setup issues.

I question the validity of the authors' claims (i), (ii), and (iv) in the overview of Section 4.

Regarding claim (i) that "SymphonySMoE enhances model performance across both pre-training and fine-tuning tasks," I have the following concerns:

(1) In Section 4.1, the authors train a MoE model with a total of 200M parameters from scratch on only **100M** tokens (WikiText-103) and report this as a **pre-training** task. 
It is difficult to draw convincing conclusions from such a limited **100M** token **"pre-training"** experiment and believe it can justify a new MoE routing strategy. 
Could the observed results simply be due to SMoE enabling faster convergence?

Furthermore, I cannot accept the results of a language model **without any pre-training** on the attacked dataset as sufficient evidence to support the claim that SymphonySMoE is more robust. (claim (ii))

(2) I do not consider the experiments in Sections 4.1 and 4.2 as fine-tuning tasks, as there is **no** pre-trained MoE model involved. 
These experiments are conducted with MoE initialized from the upcycled dense model, without any further training. 
Similar to point (1), I find it difficult to accept conclusions drawn from tuning a newly initialized MoE with billions of parameters on such a limited dataset.
As a result, this also fails to support the conclusions of claim (iv).

- The presentation of this paper could be improved. Most theoretical proofs in the main text do not focus on addressing the problem this paper try to resolve and could be moved to the appendix, while some key experimental details that support the effectiveness of the approach are placed there instead.

### Questions
Q1: Can the authors conduct experiments pre-training language models with more tokens and a bigger model scale?

For instance, pre-train the language model with 80B tokens, similar to the setup in Appendix E.1.

Q2: Can the authors provide an explanation for their choice to upcycle dense models into MoE in the experiments presented in Sections 4.2 and 4.3?
What's the performance of continual pre-training performance of an MoE model into SymphonySMoE in the same experimental setup?

Q3: What's the performance of the fine-tuned dense Phi-3 mini's performance on GLUE?

Could the authors consider testing on other benchmarks, as GLUE may not fully capture the capabilities of modern LLMs?
As a kind reminder, the statement "this setup reflects a realistic deployment scenario" seems somewhat overstated, as modern MoEs are clearly much sparser in practice.

### Soundness
2

### Presentation
2

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
This paper proposes SymphonySMoE, a novel framework for improving the robustness and interpretability of Sparse Mixture of Experts (SMoE) models by explicitly modeling expert-to-expert interactions through a graph structure.
Traditional SMoE frameworks select top-K experts independently for each token, which leads to unstable routing under distribution shift or noisy inputs. SymphonySMoE addresses this by constructing a social graph among experts, where edges represent co-activation frequency. This graph is dynamically updated via exponential moving average and used to smooth routing logits during expert selection.
Experiments on large-scale benchmarks—including WikiText-103 (language modeling), GLUE (text classification), and LLaVA-665K (vision-language instruction tuning)—demonstrate consistent performance gains over strong SMoE baselines (e.g., X-MoE, GLaM, Switch Transformer), particularly under noisy or adversarial conditions.
The paper further provides theoretical analysis showing that the learned adjacency matrix converges to an ideal co-activation measure, explaining the enhanced routing stability.

### Strengths
The paper introduces a new perspective on SMoE routing, framing it as a graph-based probabilistic inference problem that models dependencies among experts, rather than treating expert activations as independent.
The concept of a “social graph of experts” is both intuitively appealing and technically original, bridging ideas from graph neural networks, probabilistic modeling, and mixture-of-experts learning.
It offers a lightweight and modular extension that can be integrated into existing SMoE frameworks with minimal architectural modification.
The method is mathematically well-motivated and empirically validated across multiple modalities (text, vision-language).
Experiments are comprehensive, ablation studies isolate the impact of graph modeling, and robustness tests under data corruption demonstrate practical benefits.
Theoretical analysis provides a convergence guarantee for the adjacency matrix, which strengthens the credibility of the approach.

### Weaknesses
The paper reports stable gains on multiple benchmarks (such as table results in the directions of WikiText-103, GLUE, and LLaVA), and conducts a detailed complexity/runtime analysis of the overhead, but does not characterize the theoretical or empirical upper limit of Symphony routing: How far is the current improvement from the "ideal route", under what conditions will it reach its peak, and where will the diminishing returns occur?

### Questions
1.How sensitive is model performance to the EMA decay rate in updating the adjacency matrix? Would a fully learnable adjacency (trained via gradient) perform better or risk overfitting?
2.The experiments show improvement on text and vision-language tasks—does the method generalize similarly to purely visual MoE models (e.g., ViT-MoE) or speech experts?
3.Could the authors provide quantitative metrics for “expert interaction strength” or visualize how the graph evolves across training stages? This might better substantiate the social-graph analogy.
4.Please answer Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work enhances the expert routing process in SMoE by incorporating the experts co-activation information. The proposed method, SymphonySMoE, construct an a social graph (co-occurrence matrix) to model the co-activation frequency of experts during training and modify the routing process. The authors provided theoretical analysis of SymphonySMoE and validate its efficacy on several scenarios, from pre-training to fine tuning and visual instruction tuning.

### Strengths
- The idea of incorporating expert co-activation frequency is well-motivated.
- SymphonySMoE is quite elegant. Despite its simple implementation, it is theoretically-grounded and the empirical results are encouraging.

### Weaknesses
- My major concern of this work is the empirical evaluation is quite limited. 
    - First, the pre-training experiment is very small. Training ~220M models on WikiText-103 is quite limited. Furthermore, evaluation is also on the same dataset, such in-domain evaluation is not used in modern SMoE settings, most of which focus on zero-shot evaluation. A minimum scale for pre-training should be MoEUT [A] or preferably OLMoE [B].
    - Second, finetuning Phi 3 on Glue seems to be unnecessary as it is a very old benchmark and Phi 3 is likely to see the data during its pre-training. For this experiment, it is mandatory to report the original Phi 3 performance, and also consider challenging benchmarks like SuperGlue. Preferably, the authors should consider finetuning on more recent datasets like OpenCodeInstruct [C], or even doing RLHF.
    - Lastly, the visual instruction tuning experiment followed LibMoE, which reported 11 benchmarks, why did the authors only consider 7?
- Some presentation/typos/citation errors at L128, L139, L156, L161, etc. Table 1 appears too early before it was first mentioned. 

[A] Csordás, Róbert, et al. "Moeut: Mixture-of-experts universal transformers." Advances in Neural Information Processing Systems 37 (2024): 28589-28614.

[B] Muennighoff, Niklas, et al. "Olmoe: Open mixture-of-experts language models." arXiv preprint arXiv:2409.02060 (2024).

[C] Ahmad, Wasi Uddin, et al. "OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs." arXiv preprint arXiv:2504.04030 (2025).

### Questions
- The empirical evaluation of SymphonySMoE is quite limited.

- It is nice to see that the overheads during evaluation is minimal. What is the wall clock training time of SymphonySMoE compared to the baselines?

- The number of baselines considered in all experiments is quite limited. The authors should try to include more recent baselines such as MoEUT, Autonomy-of-Experts Models [D], etc.

[D] Lv, Ang, et al. "Autonomy-of-Experts Models." ICML (2025).

### Soundness
3

### Presentation
2

### Contribution
2
