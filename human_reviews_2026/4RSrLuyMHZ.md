# TeNet: Text-to-Network for Compact Policy Synthesis

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 4

## Abstract
Large vision-language-action (VLA) models such as PaLM-E, SayCan, and RT-2 enable robots to follow natural language instructions, but their billions of parameters make them impractical for high-frequency real-time control. At the other extreme, compact sequence models such as Decision Transformers are efficient but not language-enabled, relying on trajectory prompts and failing to generalize across diverse tasks. We propose TeNet (Text-to-Network), a framework that bridges this gap by instantiating lightweight, task-specific policies directly from natural language descriptions. TeNet conditions a hypernetwork on LLM-derived text embeddings to generate executable policies that run on resource-constrained robots. To enhance generalization, we introduce grounding strategies that align language with behavior, ensuring that instructions capture both linguistic content and action semantics. Experiments on state-based Mujoco and Meta-World benchmarks show that TeNet achieves robust performance in multi-task and meta-learning settings while producing policies that are orders of magnitude smaller. These results position language-enabled hypernetworks as a promising paradigm for compact, language-conditioned control in state-based simulation, complementary to large-scale VLAs that tackle vision-based robotics at massive scale.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents TeNet (Text-to-Network), a framework that generates lightweight, task-specific policies conditioned on natural language input. The authors propose two variants: Direct TeNet, which generates policies directly via a hypernetwork conditioned on text embeddings, and Grounded TeNet, which introduces a trajectory encoder to align hypernetwork inputs with trajectory embeddings. As for the grounding method, the paper explores both MSE–based alignment and contrastive alignment strategies. Experimental results on the MuJoCo locomotion and Meta-World manipulation benchmarks demonstrate that TeNet achieves strong performance and outperforms several baseline methods.

### Strengths
This paper is clearly presented, and the results seem reasonable. I appreciate the detailed training iteration in Figure 2 and ablations in Table 1. The implementation details in the appendix are also very helpful.

### Weaknesses
There are two primary concerns related to the method clarity and experiments of this paper:

1. The paper’s main contribution appears to be a novel way of integrating language embeddings with robot state inputs and action outputs through a hypernetwork-generated policy. However, several prior works (e.g., π₀ [1], RoboVLMs [2]) have already explored similar design choices. It would strengthen the paper to include direct comparisons against these baselines (not necessarily the ones I mentioned earlier) and to clearly explain how the proposed approach differs from existing methods, particularly those employing cross-attention mechanisms. Providing either empirical evidence that TeNet achieves better performance or theoretical justification for its advantages would greatly enhance the contribution of this paper.

2. The use of Meta-World is appropriate since it includes diverse tasks with associated language instructions, however, the MuJoCo benchmarks are less language-centric. It's fine to keep them, but they contribute less to validating the paper’s core claims about language-conditioned policy generation. To better support the conclusions, additional experiments on language-focused benchmarks such as LIBERO [3] or CALVIN [4] are recommended.

Given these concerns, I would currently recommend rejection, as further clarification and stronger experiments (more baselines and benchmarks) are needed to establish the paper’s contributions.

Reference:

1. https://www.physicalintelligence.company/blog/pi0

2. Liu, H., Li, X., Li, P., Liu, M., Wang, D., Liu, J., ... & Zhang, H. (2025). Towards generalist robot policies: What matters in building vision-language-action models. 

3. Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., & Stone, P. (2023). Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36, 44776-44791.

4. Mees, Oier, et al. "Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks." IEEE Robotics and Automation Letters 7.3 (2022): 7327-7334.

### Questions
How to supervise the trajectory encoder $f_{\text{traj}}$ in Grounded TeNet? Is it only supervised by $L_{\text{ground}}$? If so, could you clarify the intuition behind that?

### Soundness
2

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
This paper proposes TeNet, a hypernetwork-based framework designed to bridge the gap between large but impractical vision-language-action (VLA) models and lightweight but language-agnostic models such as the Decision Transformer (DT).

TeNet uses text and trajectory embeddings to generate task-specific policies that are compact and efficient for deployment on resource-constrained robots. 

The proposed TeNet enables direct text-conditioned policy generation through a lightweight hypernetwork, offering a new paradigm for scalable, language-enabled robot control.

### Strengths
1. The paper introduces an interesting paradigm in which a hypernetwork conditioned on natural-language embeddings generates the weights of compact, task-specific policy networks.
2. The paper is clearly written, well structured, and easy to follow.
3. Hyperparameter and experimental details are provided thoroughly, supporting reproducibility.
4. Empirical results show consistent improvements over Decision Transformer baselines across the evaluated benchmarks.

### Weaknesses
1. Architecture justification:

   The authors use a large-scale language model (LLaMA-8B) as the text encoder, followed by a two-layer MLP hypernetwork (hidden size = 128) to produce compact policy weights.

   Given the limited number of unique task descriptions (50 × 10 = ~500 text embeddings), this setup appears unbalanced—heavy on preprocessing and light on the core policy backbone.

   The authors should justify why such a large encoder is necessary and whether smaller models (e.g., T5-small [1] or BERT [2]) would achieve comparable results.

2. Contrastive text–trajectory loss lacks novelty:

   The paper presents the contrastive alignment between text and trajectory embeddings as a key contribution.

   However, prior work such as Contrastive Language, Action, and State Pre-training (CLASP) [3] already explored contrastive alignment between language and behavior, explicitly addressing the *many-to-one* and *one-to-many* mapping issue.

   CLASP uses distributional encoders (mean + variance) for both modalities, capturing the inherent variability of multiple trajectories that can satisfy a single textual description.

   In contrast, TeNet employs a deterministic contrastive loss without modeling this variability, risking over-clustering of distinct trajectories and potential loss of behavioral diversity.

   This omission weakens both the novelty and robustness of the proposed approach.

3. Limited experimental scope:

   Experiments are confined to state-based simulation benchmarks (MuJoCo and Meta-World), which limits the strength of the proposed new frames.
   These benchmarks omit perception, making them less representative of practical robot learning.

   More challenging vision-based benchmarks, such as LIBERO [4], would better demonstrate scalability and real-world relevance.

4. Missing modern baselines:

   The paper compares TeNet only with Decision Transformer (DT) and Prompt-DT.

   It omits comparisons with more recent approach such as Diffusion Policy [5], or other language-conditioned control frameworks.

   Including these baselines would strengthen the empirical positioning and clarify TeNet’s true contribution.

### Questions
1. What is the performance impact of using a smaller text encoder (e.g., T5-small [1] or BERT [2]) instead of LLaMA-8B?
2. In the state-based Meta-World setting, since the 39-dimensional observation includes goal position information [6], could the authors provide results on vision-based variants of Meta-World or on LIBERO [4]?
3. As mentioned in the weakness part, I think the text trajectories contrastive needs more careful design to reflect the many-to-many mapping nature. Can the authors justify their choice in the work?


References:

[1] Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 2020.

[2] Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.

[3] Rana et al., *Contrastive Language, Action, and State Pre-training*, *arXiv:2304.10782*, 2023.

[4] Wang et al., *LIBERO: Benchmarking Knowledge Transfer for Language-Grounded Manipulation*, *arXiv:2306.03310*, 2023.

[5] Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, *arXiv:2303.04137*, 2023.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TeNet, the core idea is to use a hypernetwork conditioned on LLM text embeddings to generate the parameters of a compact, task-specific policy network at inference time. This allows a user to provide a simple natural language instruction (e.g., "push the mug"), which TeNet then uses to instantly synthesize a small, lightweight policy (e.g., ~40K parameters) that is ready for high-frequency control.

The framework is trained offline using expert demonstrations paired with task descriptions. To improve generalization, the authors introduce "Grounded TeNet," which uses a grounding loss (e.g., contrastive) to align the text embeddings with corresponding trajectory embeddings. Experiments on Mujoco and Meta-World show that TeNet produces policies that are orders of magnitude smaller and faster than baselines like Prompt-DT, while achieving superior performance in diverse multi-task settings (MT10, MT50) and competitive performance in meta-learning settings.

### Strengths
1. Clear Novelty: The paper's core idea—using a language-conditioned hypernetwork to synthesize a compact policy network—is novel and well-motivated. The related work section is thorough, successfully positioning this as a new paradigm distinct from large VLAs, trajectory-prompted models, and other existing hypernetworks. And the successful implementation experience of hypernetworks is valuable to the community
2. Efficiency: The results in Table 1 are highly compelling: the framework generates policies that are ~100x smaller (40K vs. 1M-39M params) and >15x faster (>9kHz vs. ~200-600Hz) than the Prompt-DT baselines. And it actually improves performance in the most complex multi-task settings (MT10, MT50), where the Prompt-DT baseline struggles significantly (Fig. 2, Table 1). This shows the architecture is robust to high task diversity.
3. Effective Grounding Mechabnism: The paper clearly demonstrates the value of "grounding" language in behavior. The comparison between Direct TeNet, TeNet-MSE, and TeNet-Contrast (Fig. 2) shows that aligning text and trajectory embeddings is crucial for generalization.

### Weaknesses
1. The current framework operates on state-based inputs. This is a reasonable first step but it limits the immediate applicability of the work, as most modern embodied agents are expected to operate from vision.
2. No trajectory-based hypernetwork baseline is provided, although TeNet is the first attempt to directly use language descriptions to generate policies, a comparison with trajectory based hypernetworks would be appreciated.
3. A related work for trajectory based hypernetworks is missing [1].

Reference:

[1] Liang, Yongyuan, et al. "Make-an-agent: A generalizable policy network generator with behavior-prompted diffusion." *Advances in Neural Information Processing Systems* 37 (2024): 19288-19306.

### Questions
1. The primary motivation is real-world deployment. What are the anticipated challenges in transferring TeNet to a real robot? How would the "grounding" process be handled with real-world trajectories, which may be less consistent than the expert demonstrations used here?
2. For the grounded variants, how was the trajectory encoder (Prompt-DT) trained? Was it pre-trained separately on its own objective, or was it trained *jointly* with the TeNet framework (i.e., its weights were updated via the grounding loss)?
3. How sensitive is the final performance with regard to the text prompting? Could the hypernetwork understand more detailed descriptions like "move slowly" or directions like "move to the left"?

### Soundness
3

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
2

### Summary
The paper introduces TENET, a framework for generating compact task-specific policy networks from language conditioned hypernetworks. In particular, the framework trains a hypernetwork conditioned on a natural language description of a task, which then outputs a set of parameters for a policy network. The language description is embedded with an LLM and further aligned with a trajectory encoding through a contrastive objective. The authors evaluate TENET on Mujoco gym environments and three Metaworld settings. Experiments show that TENET outperforms baselines while achieving high control frequency.

### Strengths
- Interesting Idea
- Good presentation
- Strong results: high-frequency, efficient policies.

### Weaknesses
The main weaknesses are misleading scope and overstated claims:
- The paper's core motivation, which frames TENET as complementary to large VLAs, is misleading, since VLAs operate in complex, high-diversity vision-based, real-world scenarios, whereas TENET is evaluated only in state-based, simulation environments.
- The claim that the framework makes “TeNet more scalable and practical for diverse task sets.” is not supported by experiments
- The language instruction and task diversity are very limited and seem to only differ in numerical target values such as target location or velocity. Language instructions might actually not be needed and could be replaced by a one hot vector or similar task encodings. Ablations in that regard are missing. Also, comparisons against task-embedding or archive-based methods are missing.
- In general, more complex experiment settings such as generalization to new objects or different tasks that actually require a semantic understanding of the task instruction would support the claim that the architecture applicable to  diverse tasks.
- Actual language instructions are not shown and task settings for MT are not described in enough detaill

TENET is a promising research direction, but its claims are currently overstated. The comparison to VLAs is not appropriate, and the work requires substantial new experiments in more challenging and diverse settings to prove that the language component provides a significant, generalizable advantage over simpler task identifiers. Without this evidence, the paper's novel contribution is limited.

### Questions
- How does this scale to different model sizes? Such a high control frequency is usually not needed (or usable) in downstream robot manipulation applications.
- Does the hypernetwork actually understand language instructions? How do different instructions change behavior?
- How does the approach compare against methods using task embeddings/one hot embeddings?
- How do different (smaller) language encoders influence performance?
- How does the architecture perform on IL-only benchmarks (without constrastive alignment of language and trajectory)?

### Soundness
2

### Presentation
3

### Contribution
2
