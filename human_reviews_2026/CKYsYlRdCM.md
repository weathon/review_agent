# ACCORD: Alleviating Concept Coupling through Dependence Regularization for Text-to-Image Diffusion Personalization

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Image personalization enables customizing Text-to-Image models with a few reference images but is plagued by "concept coupling"—the model creating spurious associations between a subject and its context. Existing methods tackle this indirectly, forcing a trade-off between personalization fidelity and text control. This paper is the first to formalize concept coupling as a statistical dependency problem, identifying two root causes: a Denoising Dependence Discrepancy that arises during the generative process, and a Prior Dependence Discrepancy within the learned concept itself. To address this, we introduce ACCORD, a framework with two targeted, plug-and-play regularization losses. The Denoising Decouple Loss minimizes dependency changes across denoising steps, while the Prior Decouple Loss aligns the concept’s relational priors with those of its superclass. Extensive experiments across subject, style, and face personalization demonstrate that ACCORD achieves a superior balance between fidelity and text control, consistently improving upon existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses concept coupling in T2I personalization and proposes ACCORD with two plug-and-play losses to control denoising/prior dependence. Results show consistent alignment/fidelity gains across several backbones with modest overhead.

### Strengths
1. Concept coupling is formalized as conditional dependence and decomposed into denoising vs. prior components, each tied to a dedicated loss.
2. Provides a closed/near-closed form via the “model-as-implicit-classifier” view—no auxiliary discriminator or teacher model required.

### Weaknesses
1. Some works need to be discussed and compared in the Related Work section，Update-space constraints: Custom Diffusion [1], PaRa [2], Attention-level preservation: Perfusion [3], Attend-and-Excite [4]. Embedding-only personalization: Textual Inversion [5].
2. Human-eval reporting lacks agreement stats



[1] Nupur Kumari, Bing Li, David Forsyth, Jia-Bin Huang, Vishal M. Patel, Oncel Tuzel, Ali Farhadi, Anima Anandkumar. Multi-Concept Customization of Text-to-Image Diffusion. CVPR, 2023.

[2] Shangyu Chen, Zizheng Pan, Jianfei Cai, Dinh Phung. PaRa: Personalizing Text-to-Image Diffusion via Parameter Rank Reduction. ICLR, 2025.

[3] Omer Tov, Yuval Alaluf, Yotam Nitzan, Daniel Cohen-Or, Tali Dekel. Key-Locked Rank-One Editing for Text-to-Image Personalization (Perfusion). SIGGRAPH, 2023.

[4] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, Daniel Cohen-Or. Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models. 2023.

[5] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, Daniel Cohen-Or. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. 2022.

### Questions
1. How sensitive are your losses to scheduler/noise settings and guidance scales?
2. Can ACCORD be demonstrated in multi-concept personalization and report cross-concept interference?

### Soundness
3

### Presentation
3

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
The author introduce the concept coupling issue in personalization. They provide the theoretically analysis of this issue and propose several loss for resolving it. Experiments shows the performance of their method is good in some cases.

### Strengths
- This paper is theoretically interesting, author introduce several formula for explaining the phenomenon of dependence discrepancies.
- The author introduce Denoising Decouple loss(DDL) and Prior Decouple Loss(PDL) for resolving the issue.

### Weaknesses
- The scope of this work is limited, regard the real-world application, it's not hard for users to take a set of pure images including only desired objects at most cases. For case in figure 1, it's more likely a data issue. Decoupling is important in personalized generation when being applied to concepts that cannot be easily decoupled by the user, such as atmosphere, lighting, posture, and materials, which is called abstract concept personalization. Could author explore more qualitative experiments on this domain?
- Concern about the robustness of the method. Is it sensitive to prompts? When prompts fail to accurately describe the background beyond the personalized object, will the DDL negatively impact model performance? Furthermore, would more refined prompt improve model performance?

### Questions
- Would final optimization object be like \mathcal{L}  = \mathcal{L}_{reconstruction} + \alpha\mathcal{L}_{DDL} + \beta\mathcal{L}_{PDL}? If so, how do authors decide the value of \alpha and \beta, should have some ablation experiments here.
- Considering the first case of figure2, why DDL/PDL works here? Shoes and feet are semantically dependent concepts, and this dependency is not introduced by the source image. As I understand it, such examples are not something that DDL/PDL can address. Could the author elaborate on the theoretical basis behind this; Do DDL and PDL here only serve as a regularization term instead of turning the direction of optimization process goes to a better trade-off?
- What percentage of DreamBench examples resemble "people carrying red envelopes?" Is this the majority of the benchmark? If this data is extracted, will the DDL & PDL performance improve more than on the general benchmark?
- Theory in the main text is well explained but too abstract, better to move figure 4&5 to main paper for clarify.

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
3

### Summary
They propose ACCORD, a framework incorporating two regularization losses to improve personalization performance. The Denoising decouple loss (DDLoss), designed to mitigate concept coupling, minimizes dependency discrepancies to prevent conditional dependence between the personalized concept $c_p$ and the general text concept $c_g$ across successive time steps. The Prior decouple loss (PDLoss) addresses prior dependency by leveraging CLIP projections to align the learned concept’s relational structure with that of its original class, thereby preserving semantic consistency. With theoretical justification and experimental validation, the framework demonstrates its effectiveness across various fine-tuning methods for personalization, e.g., DreamBooth or Custom Diffusion.

### Strengths
**S1**. Concept decoupling in personalization is an important issue, and unlike existing indirect approaches that address it through explicit masking or regularization, this work resolves the problem with a simple yet effective decoupled loss design.

**S2**.  The proposed method demonstrated its effectiveness through extensive experiments across various fine-tuning frameworks and comprehensive ablation studies.

### Weaknesses
**W1**. Although the paper proposes a method for concept decoupling, the evaluation appears to focus primarily on single-subject personalization benchmarks such as DreamBench. It would be necessary to include evaluations on multi-subject personalization tasks, such as Break-a-Scene [1], to better assess how effectively the method achieves concept decoupling.

[1] Avrahami et al., Break-a-scene: Extracting multiple concepts from a single image, 	SIGGRAPH Asia 2023

**W2**. While the motivation for the proposed metho, avoiding explicit masks or priors (e.g., in attention maps or diffusion losses), is understandable, a performance comparison with such approaches is necessary. The paper compares various training frameworks, but lacks baseline comparisons specifically focused on different learning strategies.

**W3**. Recent powerful text-to-image models (e.g., FLUX-dev) show effective concept decoupling even with purely descriptive prompts. However, the paper lacks sufficient validation to demonstrate whether the proposed method can further improve performance in such flow or transformer based models.

### Questions
**Q1**. Is the proposed method capable of handling more challenging decoupling cases? For example, in Figure 2, could the model successfully decouple attributes such as the blue hat from the brown teddy bear if an appropriate subclass were defined?

**Q2**. How much slower is the training compared to the standard DreamBooth? The proposed method seems to involve multiple regularization terms, which may considerably increase training time.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets {concept coupling}—a critical limitation in text-to-image (T2I) diffusion personalization where target concepts  become unintendedly entangled with irrelevant general concepts. Unlike prior indirect solutions , ACCORD formally models coupling as a statistical dependence problem: coupling arises when the conditional dependence between a target concept ($c_p$) and a general concept ($c_g$) in generated images deviates from the prior dependence between $c_p$’s superclass ($c_s$) and $c_g$.

### Strengths
Theoretically Grounded Innovation: Framing coupling as statistical dependence  provides a unified language for understanding coupling, which was previously described via ad-hoc examples. This foundation enables reproducible, extensible loss design—unlike heuristic methods that require case-by-case tuning.  
 Loss Design: DDLoss and PDLoss address complementary aspects of coupling: DDLoss stabilizes dependence during denoising (preventing sudden coupling), while PDLoss aligns with superclass priors (preventing persistent coupling). Ablation results (Tab. 5) show their combination yields 15–20% higher text control than either loss alone, demonstrating thoughtful, non-redundant design.  
 Comprehensive Experimental Validation: The paper tests ACCORD across three distinct personalization scenarios (subject/style/face) and uses both automatic and human metrics. This breadth ensures generalizability—critical for a method claiming to solve a "general" coupling problem. Human preference tests (72% of annotators favor ACCORD) also address a key limitation of automatic metrics (e.g., CLIP-T) that may not capture subjective quality.

### Weaknesses
Insufficient Ablation of Loss Weight Interactions: The paper ablates individual losses (DDLoss only, PDLoss only) but not how their weights ($\lambda_D$, $\lambda_P$) interact. For example, does increasing $\lambda_D$ improve control but harm fidelity when $\lambda_P$ is low?  
Without this analysis, users cannot optimize weights for specific use cases (e.g., style personalization may require higher $\lambda_P$ than subject personalization).  

The Prior Decouple Loss (PDLoss) is entirely dependent on Assumption 1, which posits that $p(c_j|c_k) \approx \frac{e^{\tau \cos(f_j, f_k)}}{Z_k}$. This is a very strong theoretical leap. While CLIP's objective aligns image-text pairs, extending this to a proxy for the conditional probability between any two text concepts ($c_p$ and $c_g$) is not rigorously justified.

The method relies on a VLM to generate captions and thus identify the co-occurring concepts $c_g$ to be decoupled. The paper states this is superior to templates. This introduces a critical, unevaluated dependency. The method's performance is now tied to the VLM's ability to correctly identify all relevant coupled concepts. If the VLM fails to mention the "girl" in the backpack's caption, for example, the framework may fail. No ablation study is provided to test the sensitivity to caption quality or source.

### Questions
How should a user practically apply ACCORD if the reference images for $c_p$ contain multiple coupled concepts (e.g., a dog $c_p$ always on a "blue rug" $c_{g1}$ and next to a "red ball" $c_{g2}$)? Must the VLM identify both $c_{g1}$ and $c_{g2}$?

### Soundness
3

### Presentation
2

### Contribution
3
