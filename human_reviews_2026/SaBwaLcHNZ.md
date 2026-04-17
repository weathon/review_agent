# Character Mixing for Video Generation

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Imagine Mr. Bean stepping into Tom and Jerry---can we generate videos where characters interact naturally across different worlds? We study inter-character interaction in text-to-video generation, where the key challenge is to preserve each character’s identity and behaviors while enabling coherent cross-context interaction. This is difficult because characters may never have coexisted and because mixing styles often causes **style delusion**, where realistic characters appear cartoonish or vice versa. We introduce a framework that tackles these issues with Cross-Character Embedding (CCE), which learns identity and behavioral logic across multimodal sources, and Cross-Character Augmentation (CCA), which enriches training with synthetic co-existence and mixed-style data. Together, these techniques allow natural interactions between previously uncoexistent characters without losing stylistic fidelity. Experiments on a curated benchmark of cartoons and live-action series with 10 characters show clear improvements in identity preservation, interaction quality, and robustness to style delusion, enabling new forms of generative storytelling. Our project page https://mi-mi-x.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new framework for enabling natural interactions between multiple characters from different visual domains, such as mixing Mr. Bean with Tom and Jerry using text-to-video (T2V) generation. The authors address two key challenges: non-coexistence, where characters from separate shows never appear together in training data, and style delusion, where mixed-style characters lose their original visual fidelity. To overcome these, they propose Cross-Character Embedding (CCE), which learns disentangled identity and behavioral representations from multimodal sources through structured character–action captions, and Cross-Character Augmentation (CCA), which generates synthetic co-existence data by compositing characters across domains while preserving native styles. As a result, the paper shows improvement compared to the baselines.

### Strengths
A key strength of this paper lies in its clear motivation, to bridge the gap between distinct fictional universes by enabling natural, style-consistent multi-character interactions. This being imaginative and well-grounded in practical generative modeling challenges. The authors present a thoughtfully designed data composition strategy, curating a dataset that balances diversity across cartoons and live-action domains while maintaining detailed character and style annotations. 

This well-structured dataset, combined with the Cross-Character Embedding and Augmentation techniques, provides a solid foundation for learning both behavioral and stylistic nuances. Overall, the study is well-organized, and the presentation neatly ties together motivation, method, and results into a coherent and convincing.

### Weaknesses
[Major]
While the paper presents an impressive dataset and compelling demonstrations, its technical contribution beyond data construction and fine-tuning remains relatively incremental. The proposed Cross-Character Embedding (CCE) and Cross-Character Augmentation (CCA) modules, though effective, largely extend existing ideas of prompt-based disentanglement and synthetic compositing rather than introducing fundamentally new architectural or generative mechanisms. 

The framework heavily relies on caption engineering, LoRA-based adaptation, and GPT-assisted annotation which approaches that are conceptually straightforward and depend more on large-scale data quality than algorithmic novelty. Consequently, while the work excels in implementation and application scope, its methodological innovation is modest compared to the scale of its dataset and the strength of its empirical results. 

[Minor]
Some references related to video personalization are missing. ToonCrafter [1] proposed a fine-tuning technique for adapting video diffusion models to general cartoon domains, while AnyMoLe [2] introduced a video fine-tuning framework for generating motion-consistent videos of a single character.

There is typo in L464 (Section ??).

[1] ToonCrafter: Generative Cartoon Interpolation via Diffusion Models. Liu et al., SIGGRAPH ASIA, 2024.

[2] AnyMoLe: Any Character Motion In-betweening Leveraging Video Diffusion Models. Yun et al., CVPR, 2025.

### Questions
How many A100 GPUs were used and how long does it took for training?

Are you planning to open-source the code and dataset?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors identify two core challenges: non-coexistence, meaning some characters never appear together in training data, and style delusion, meaning cartoon and live action styles bleed into each other. To address this, they propose Cross Character Embedding, which uses structured captions of the form [character: name], action to disentangle identity and behavior, and Cross Character Augmentation, which pastes segmented characters into foreign style backgrounds to simulate cross-universe co-occurrence while preserving each character’s native look.

### Strengths
1. The paper tackles a concrete and highly visible capability gap in current video generation systems. Existing models can often render a single customized subject, but coherent multi-character interaction across different shows and even across cartoon versus live action domains remains very brittle.

2. The authors curate a reasonably large, annotated, behavior-aware dataset, and define evaluation metrics that target identity, motion, style, and interaction quality in multi-character settings.

3. Qualitative demos show convincing multi-character interactions that typical text-to-video systems struggle with.

### Weaknesses
1. Baseline fairness is underspecified. The proposed model is LoRA fine tuned on an 81 hours character specific dataset, whereas baselines may be evaluated mostly zero shot. This makes it hard to attribute the gains to CCE or CCA rather than just stronger task specific tuning.

2. Generalization is narrow. All results focus on about ten characters from four shows. The paper does not show that the approach scales to arbitrary new identities or unseen characters without new fine tuning.

### Questions
1. How were baselines adapted. Did you fine tune SkyReels A2, Wan2.1 I2V, or other baselines on the same eighty one hour dataset with LoRA style adapters, or are they evaluated zero shot. Please clarify to ensure a fair comparison.

2. Can you quantify failure modes. The discussion notes occasional breakdowns in highly complex multi character scenes with overlapping motion patterns.

### Soundness
2

### Presentation
3

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
This paper introduces a new task termed Character Mixing, aiming to generate multi-character videos where characters from different IPs or styles appear and interact in the same scene. The authors construct a dataset by mixing existing videos and captions with the help of large language and vision models, and design two modules (CCE and CCA) to align cross-character and cross-style information. Experiments on several cartoon characters are reported.

### Strengths
1. The idea of character mixing is creative and intuitively interesting, which could inspire further exploration in cross-style or cross-domain video generation.

2. The authors contribute a dataset with structured captions for multi-character interaction videos, which may be useful for future research.

### Weaknesses
1.Low problem significance.
While entertaining, the problem does not address a clear or impactful research challenge. It is more of a creative application than a fundamental scientific question. The motivation for why character mixing matters for the video generation community is weak.

2.Limited methodological novelty.
The proposed CCE and CCA modules mainly rely on prompting and data augmentation rather than introducing new modeling or learning principles. The improvements largely stem from the underlying base model's(Wan2.1 14B used, smaller size or other open-source model should be presented to support model-independent claim)  capability rather than the proposed method itself.

3.No formalization or mathematical clarity and training details.
The paper lacks any formal task definition, notation, or training objective. Without clear training details, it is difficult to reproduce and evaluate the soundness of the approach.

4.Limited scalability and generalization to new characters.
The reliance on explicit character-level annotation and LoRA-based fine-tuning means adding a new character (especially from unseen domains) requires re-training or substantial data preparation (highlighted in Discussion, Page 9). This approach does not scale gracefully for open-world or user-specified characters.

5.Missing analysis and discussion.
There is little exploration of failure cases, interaction quality (e.g., temporal consistency, occlusion handling), or computational cost. The paper reads more like a demo than a scientific study.

### Questions
1.Can the method handle unseen characters or styles at inference time?

2.What would happen if GPT-4o/Gemini were replaced with open-source models (e.g., LLaVA, Qwen2-VL)?

3.How is the proposed CCE/CCA architecture implemented and trained?

### Soundness
2

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
This paper introduces a novel framework for generating videos that feature multiple characters from different fictional universes (e.g., cartoons and live-action shows) interacting with each other. The authors identify two primary challenges: 1) the "non-coexistence" of characters in training data, and 2) "style delusion," where characters' visual styles blend undesirably. To address these, they propose a two-part solution. First, **Cross-Character Embedding (CCE)**, a prompt engineering strategy that uses structured captions of the form `` `[Character: <name>], <action>` `` to disentangle character identity and behavior. This allows the model to learn character-specific traits from their respective source videos and compose them at inference time. Second, **Cross-Character Augmentation (CCA)**, a data augmentation technique that synthetically creates training examples of cross-style interaction by segmenting characters from one domain and pasting them into scenes from another. These augmented clips are captioned with an additional `` `[scene-style: <style>]` `` tag to help the model preserve stylistic integrity. The authors fine-tune a large-scale text-to-video model (Wan2.1) on a curated 81-hour dataset and demonstrate through extensive experiments that their method significantly outperforms existing baselines in identity preservation, interaction quality, and style consistency. They also introduce a new benchmark and a set of VLM-based evaluation metrics for this specific task.

### Strengths
*   **Problem Formulation and Significance**: The paper formulates a compelling and significant research problem: enabling characters from different "universes" to interact naturally in generated videos. This is a natural evolution of personalized generation and has high potential for creative applications.
*   **Novel Methodology**: The proposed framework is original in its combination of two distinct ideas to solve two well-defined problems. CCE (via prompt structure) tackles the non-coexistence of characters, and CCA (via synthetic compositing) addresses the style delusion problem. This two-pronged approach is elegant and shown to be effective.
*   **Strong Empirical Results**: The qualitative results are visually impressive and clearly demonstrate the superiority of the proposed method over existing approaches, which either fail to maintain identity or cannot produce coherent interactions. The quantitative results, despite the aforementioned issues with the ablation tables, generally show a strong performance lead, especially on the task-specific VLM metrics.
*   **Benchmark Contribution**: The introduction of a dedicated benchmark for multi-character interaction, including a new suite of VLM-based metrics tailored to character identity, motion, style, and interaction, is a substantial contribution in its own right. It provides a more meaningful way to evaluate models on this task than standard metrics alone.

### Weaknesses
*   **Scalability and Data Dependency**: The method's primary weakness is its reliance on fine-tuning using a large corpus of video data for a pre-defined set of characters. As implied in Section 3.3, each new character universe (e.g., a TV show) requires collecting hours of video footage and undergoing an expensive fine-tuning process. This makes it difficult to scale to new, arbitrary characters in an open-world setting. As acknowledged by the authors, this is a significant limitation.
*   **Lack of Human Evaluation and Unjustified Metric Choices**: For a task where success is highly subjective (e.g., "authenticity", "plausibility"), the absence of a human study is a major weakness. The paper instead introduces VLM-based metrics but fails to justify why established identity preservation metrics (e.g., face recognition similarity) were not used or compared against, especially for the human-like characters. While VLM evaluation is innovative, its reliability is questionable without proper protocol.
*   **Questionable Reproducibility of VLM-based Evaluation**: The use of a VLM as a core evaluation tool introduces significant reproducibility concerns. The output of VLMs can be stochastic due to parameters like `temperature`. The paper fails to describe the protocol used to ensure deterministic and reproducible scores. Key details are missing: Was `temperature` set to 0? Were scores averaged over multiple runs? What were the exact prompts used? Without this information, the quantitative results in Table 1, 2, and 3 are not fully credible.
*   **Unverified Claim of Model-Agnosticism**: The paper claims its framework is "model-agnostic" (Section 3.1) but provides no empirical evidence. All experiments are conducted by fine-tuning a single base model (Wan2.1-T2V-14B). Without applying the CCE and CCA framework to at least one other distinct T2V backbone, this claim of generalizability remains unsubstantiated.
*   **Clarity and Consistency of Ablation Studies**: The quantitative results in the ablation section (Tables 2 and 3) are poorly presented. There are numerical inconsistencies for what should be identical experimental conditions across different tables. For example, the results for the full model ("Ours" in Table 1, "w/ Both" in Table 2, and "10%" in Table 3) all report different scores. This makes it impossible to confidently assess the individual contributions of the proposed components. This must be fixed.

### Questions
1.  **On Data Requirements**: Could you please clarify the data requirements more explicitly? The paper implies a single fine-tuning on a mixed dataset. Does this mean to add a new character from a new TV show, one must re-run the entire fine-tuning process on the combined old and new data? What is the approximate training time (e.g., in GPU-hours) for the reported 5-epoch fine-tuning on the 81-hour dataset?
2.  **On Evaluation Metrics**:
    *   (a) Could you justify the decision to exclusively use VLM-based metrics for identity preservation over established methods like face recognition ID similarity, at least for the human characters (Mr. Bean, Young Sheldon's cast)? A comparison showing the VLM's superiority would strengthen your metric choice.
    *   (b) Crucially, what protocol was used to ensure the VLM evaluation is deterministic and reproducible? Please provide the exact prompts, model version, and parameter settings (especially `temperature`) used to query Gemini-1.5-Flash for the Identity-P, Motion-P, Style-P, and Interaction-P scores in the appendix.
3.  **On Generalizability**: To substantiate the "model-agnostic" claim, could you provide any results or insights, even preliminary, from applying your CCE and CCA framework to a different base T2V model?
4.  **On Inconsistent Ablation Results**: Could you please clarify and unify the results presented in Tables 1, 2, and 3? Specifically, please ensure that the results for the same experimental setup are consistent across all tables and explain the counter-intuitive findings in Table 2 (e.g., the drop in `` `Subject-C` ``).
5.  **On CCE's Mechanism**: The term "Cross-Character Embedding" suggests a specific learned representation. Can you clarify whether your method learns an explicit, separable embedding for each character, or if CCE is more accurately described as a structured prompting technique that influences the text-conditioning of the frozen T2V model?

### Soundness
3

### Presentation
3

### Contribution
4
