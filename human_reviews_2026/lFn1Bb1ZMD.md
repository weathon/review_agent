# Adaptable Symbolic Music Infilling with MIDI-RWKV

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Existing work in automatic music generation has mostly focused on end-to-end systems that generate either entire compositions or continuations of pieces, which are difficult for composers to iterate on. The area of computer-assisted composition, where generative models integrate into existing creative workflows, remains comparatively underexplored. In this study, we address the tasks of model style adaptation and multi-track, long-context, and controllable symbolic music infilling to enhance the process of computer-assisted composition. We present MIDI-RWKV, a small foundation model based on the RWKV-7 linear architecture, to enable efficient and coherent musical cocreation on edge devices. We also demonstrate that MIDI-RWKV admits an effective method of finetuning its initial state for style adaptation in the very-low-sample regime. We evaluate MIDI-RWKV and its state tuning on several quantitative and qualitative metrics with respect to existing models, and release model weights and code in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces MIDI-RWKV, a 35M-parameter symbolic music infilling model based on the RWKV-7 linear attention architecture, trained on the GigaMIDI dataset for multi-track, long-context, and controllable music generation. It proposes "state tuning" as an efficient method for low-sample style adaptation by optimizing the model's initial hidden states. The authors evaluate the model on objective metrics (e.g., content preservation, groove similarity) and a subjective listening test, claiming it outperforms baselines like MIDI-GPT and Composer's Assistant in infilling tasks.

### Strengths
The paper is clearly written, with well-organized sections, informative figures (e.g., comparisons of encoding schemes and infilling representations), and a logical flow from motivation to experiments. 

It contextualizes the work adequately within related literature on symbolic music generation and linear architectures. In terms of quality, the experimental setup leverages established datasets (GigaMIDI and POP909) and metrics, and the release of code, models, and supplementary materials enhances reproducibility and potential for community use. 

The application of RWKV-7 to symbolic music infilling addresses practical limitations like long-context handling on edge devices, which could be significant for computer-assisted composition workflows. State tuning represents a modest originality in adapting linear models for low-data style transfer, building on prior ideas in RNN initialization but applying them to a music domain.

### Weaknesses
The core innovation is limited: the model primarily involves training an RWKV-7 backbone on GigaMIDI with existing REMI+ encoding and Bar-Fill infilling objectives (adapted from Pasquier et al., 2025), followed by fine-tuning on POP909. This lacks significant new scientific insights, as the infilling techniques and attribute controls are derived from prior works (e.g., von Rütte et al., 2023; Huang & Yang, 2020), and state tuning, while efficient, is an extension of established RNN initialization concepts (e.g., Gers et al., 2002) without deep theoretical novelty for ML. 

The baselines are insufficiently comprehensive; comparisons focus on models like MIDI-GPT, MMM, and Composer's Assistant, which are not the most mainstream or recent in symbolic music generation. Stronger baselines such as Text2MIDI (Rizzotti et al., 2025, which supports infilling via inference alignment), Text2MIDI-InferAlign, or the Multi-Track Music Transformer (Yu et al., 2022) could provide better context, especially for long-range dependencies and controllability. 

The subjective listening test is mentioned but lacks methodological details in both the main text and appendix (e.g., number of participants, their musical training, evaluation criteria like coherence or preference scales, inter-rater reliability), making it hard to assess its validity—adding these would strengthen claims of practical utility. 

Overall, the work offers few reusable ML insights (e.g., beyond music-specific applications) and seems better suited for music-focused venues like ISMIR rather than ICLR, where broader algorithmic or architectural advancements are emphasized.

### Questions
1. Could you elaborate on the subjective listening test methodology? For instance, how many participants were involved, what was their level of musical expertise (e.g., trained musicians vs. general listeners), what specific criteria were used for rating (e.g., musical coherence, style fidelity, overall preference), and were there measures for inter-rater agreement? Providing these details could bolster the qualitative claims.

2. Why were baselines limited to models like MIDI-GPT and Composer's Assistant? Comparing with more recent or versatile systems such as Text2MIDI (which can handle infilling through inference techniques), Text2MIDI-InferAlign, or the Multi-Track Music Transformer could better demonstrate MIDI-RWKV's advantages in long-context or multi-track scenarios. If these were considered but excluded, what were the reasons?

3. The paper emphasizes state tuning's efficiency for low-data adaptation, but how does it generalize beyond POP909 melodies? For example, have you tested it on other styles or datasets, and what theoretical justifications support its superiority over LoRA in RWKV specifically? Ablations on varying sample sizes (e.g., 10 vs. 99) could clarify its robustness.

4. The contributions seem tailored to music generation; what broader reusable insights does this offer the ICLR community, such as in other sequence modeling domains (e.g., text or code infilling with linear architectures)?

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
This paper introduces MIDI-RWKV, a foundation model for symbolic music infilling designed for computer-assisted composition. The model is based on the RWKV-7 architecture , a linear-time architecture that allows MIDI-RWKV to be relatively small (35M parameters ) and efficient, making it suitable for edge devices and long-context applications.
The system aims to be controllable, using attribute controls for features like note density and polyphony, and adaptable. For adaptability, the paper's main methodological contribution is the exploration of state tuning—finetuning the model's initial hidden state rather than its weights—for low-sample style adaptation.
The authors conduct a thorough evaluation:
- They compare the base MIDI-RWKV against other infilling models like Composer's Assistant and MIDI-GPT on several objective metrics (e.g., Content Preservation, Groove Similarity) , showing that their model is competitive or superior, particularly given its smaller size.
- They directly compare state tuning against LoRA finetuning on a style adaptation task using the POP909 dataset.
- The results, supported by both objective metrics and a subjective listening test , demonstrate that state tuning provides superior adaptation in a low-sample regime, being preferred by participants over both the base model and LoRA-tuned variants.

### Strengths
High Quality and Empirical Thoroughness: The paper's greatest strength is its high-quality, thorough, and well-conducted empirical evaluation. The authors compare their 35M parameter model against several relevant baselines, including a larger one (CA, 54M), and demonstrate its effectiveness. The validation of the state tuning method is particularly strong, as it includes objective metrics, a subjective listening test, and stability analysis.

Practical Significance: The paper tackles a significant and practical problem for musicians: creating controllable generative tools that are efficient enough to run on local, "edge" devices. By using the efficient RWKV-7 architecture and developing a small 35M parameter model, this work is a positive step towards democratizing such tools.

Novel Application of a Method: While "state tuning" is not a new concept (as the paper admits 34), its application to style adaptation in a modern, large-scale generative model is a key strength. The paper provides a strong demonstration that this
parameter-efficient method (training only $L \times d$ parameters) can be more effective than LoRA in a low-sample regime, which is a valuable finding for the community.

### Weaknesses
$\bullet$ Limited Conceptual Novelty: As detailed in the "Contribution" section, the paper's primary weakness is its lack of fundamental research novelty. The work is a clever and effective combination of existing components (RWKV-7 architecture , REMI+ encoding , Bar-Fill objective , and state tuning ). This makes the paper feel more like a strong technical report or an application paper rather than a new research contribution for a conference on learning representations.

$\bullet$ Limited Analysis of State Tuning: The paper shows that state tuning works well, but it doesn't deeply investigate why. The rationale in Section 3.4 is a hypothesis ("directs the hidden state evolution to operate within a new subspace" ). The paper would be significantly strengthened if it included experiments to verify this, for instance, by analyzing and visualizing the hidden states of the base vs. tuned models to show this "subspace" shift. Without this, the contribution remains a "black box" empirical finding.


$\bullet$ Infilling Objective Limitations: The "single-section infilling" objective  is a significant simplification of the arbitrary infilling problem. The authors rightly note this avoids the "exponentially" large space of masking patterns. However, this comes at the cost of inference-time latency. To emulate the arbitrary infilling (bottom of Fig 3), the system must perform "repeated single-section infilling". The authors concede this "currently takes somewhat longer than with Composer's Assistant". This serial, repetitive inference process could potentially negate the training and per-token efficiency gains of the RWKV architecture, especially for complex infilling requests from a user. This trade-off is not quantified.

### Questions
$\bullet$ On the Novelty of State Tuning: The paper correctly cites prior work on training the initial state of RNNs. Could you please clarify precisely what the novel contribution of your "state tuning" method is, beyond applying this existing concept to the RWKV architecture for style adaptation? Is there a methodological difference in how you optimize the state, or is the contribution purely the empirical demonstration of its effectiveness in this new context?

$\bullet$ On Inference Latency: The "single-section infilling" approach requires "repeated single-section infilling"  to handle arbitrary masking patterns. This seems to introduce a significant serial dependency at inference time. Could you quantify this latency? For example, for the "arbitrary masking pattern" shown in Figure 3 (bottom), how many sequential model calls would be required, and how does the total wall-clock time compare to a single call from a model that can handle arbitrary masking (like Composer's Assistant)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new training pipeline for symbolic music generation using RWKV and state tuning.

### Strengths
1. The application of RWKV to MIDI LM is an important work for the music AI community. 
2. The results on state tuning vs LoRA is pretty interesting.
3. Nice narrative and writing.

### Weaknesses
1. The major limitation is the novelty. The encoding scheme, model architectures are all well-defined, making it a good application paper but less ideal for a ICLR paper.
2. The setting of single-section infilling is limited compared to abitrary masking.
3. Missing comparison against some types of symbolic infilling models, like diffusion-based [1].
4. The fine-tuning datasaet is limited to only POP909. If only 99 songs are needed for training there are many other genres can be experimented on, including folk songs (Nottingham, BFDB), classical piano, pop piano (ailabs1k7), pop (rwc pop) etc. Should be  easy to at least get some demos and objective evaluation on them.
5. In table 2 & 3, it would be better to add the evaluation metrics for the ground-truth (human).
6. The subjective evaluation is not performed against existing baselines.

[1] Min, L., Jiang, J., Xia, G., & Zhao, J. (2023). Polyffusion: A diffusion model for polyphonic score generation with internal and external controls. arXiv preprint arXiv:2307.10304.

### Questions
1. Line 098: A real-world efficiency comparison between MIDI RWKV and transformer-based model would be cool. 
2. Line 228: Why "the analog would be the KV cache"? It looks more similar to (trainable) prefix tokens to me.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes MIDI-RWKV, a controllable, adaptable multi-track symbolic music infilling model based on RWKV-7. It trains a ~35M-parameter model on GigaMIDI and introduces state tuning (optimizing the model’s initial hidden state) for low-sample style adaptation. Objective evaluations compare against MIDI-GPT and Composer’s Assistant variants; a 28-participant listening test suggests state tuning beats base/LoRA on POP909 melody infillings. Authors say code and weights are included in the supplementary.

### Strengths
- Simple, efficient backbone + controllability: Using RWKV-7 for long-context infilling with numerical/categorical controls is well-motivated and clearly described.
- State tuning is a neat adaptation mechanism, parameter-efficient and conceptually distinct from LoRA; the paper positions it clearly. 
- Reasonable benchmarking on single-section and random infilling against CA and MIDI-GPT with transparent metrics.
- Human study present (28 participants) with statistical analysis, albeit limited in scope (see weaknesses).

### Weaknesses
- No accessible demo/audio page: For a music generation paper, the submission provides no public audio examples or interactive demo; only a claim that “code and weights [are] in the supplementary.” This makes it hard to independently judge musical quality, control fidelity, and usability.
- Evaluation scope undercuts the headline claim: State-tuning experiments and the listening test are confined to POP909 melody-only finetuning, not multi-track use, limiting evidence for the paper’s core “multi-track controllable infilling” pitch. 
- Missing stronger baselines in key settings: While the paper lists CA2 as a modern system, it does not compare against it in results; objective tables include CA/MIDI-GPT but not CA2. This weakens claims about competitiveness.
- Human study design is narrow: The user study (28 participants) truncates audio to 4 bars around the infill and compares only finetuned MIDI-RWKV variants (not cross-model comparisons), limiting what we can conclude about musicality and control in realistic multi-track contexts.
- Latency and workflow gaps acknowledged: The method cannot compose in real time, DAW integration is not yet available, and emulating arbitrary-mask infilling via repeated single-section calls is slower than Composer’s Assistant (partly due to under-optimized RWKV inference). This undermines practical impact for ICLR.

### Questions
- Audio/demo: Will you provide a public audio page (e.g., curated examples, ablations for control success vs. coherence) and/or a minimal interactive demo (Colab/Gradio)?
- Multi-track evidence: Can you add multi-track subjective tests (not only melody) and show how attribute controls interact across tracks?
- Baselines: Can you include CA2 (or explain why it’s infeasible) and, for POP909, stronger LoRA configurations under matched training budgets/parameter counts?
- Arbitrary-mask infilling: Do you have one-shot arbitrary-mask results/latency (not multi-call emulation), or can you quantify wall-clock latency vs. CA/MIDI-GPT with caching enabled?
- Real-time/DAW path: Any concrete plan or prototype for streaming/real-time token generation and DAW integration (e.g., a Calliope or VST bridge), with preliminary timings?
- State-tuning limits: Could you test extreme/out-of-distribution styles (the paper itself flags this risk) and provide ablations on state-vector size/training steps?

### Soundness
2

### Presentation
3

### Contribution
2
