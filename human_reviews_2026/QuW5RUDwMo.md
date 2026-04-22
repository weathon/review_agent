# From Evaluation to Defense: Advancing Safety in Video Large Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 4

## Abstract
While the safety risks of image-based large language models (Image LLMs) have been extensively studied, their video-based counterparts (Video LLMs) remain critically under-examined. To systematically study this problem, we introduce \textbf{VideoSafetyEval} -- a large-scale, real-world benchmark for Video LLM safety, which comprises 11.4k video-query pairs and spans 19 principal risk categories. Based on this, \textit{we reveal that integrating video modality degrades safety performance by an average of 34.2\%, thereby exposing systemic risks in multimodal attack exploitation.}
To address this vulnerability, we propose \textbf{VideoSafety-R1}, a dual-stage framework achieving unprecedented safety gains through three innovations: (1) VideoSafetyThinking dataset contains 46k video-query–thinking response triplets.  (2) Alarm Token-Guided Safety Fine-Tuning (AT-SFT) injects learnable alarm tokens into visual and textual sequences, enabling explicit harm perception across modalities via multitask objectives. (3) Safety-guided GRPO enhances defensive reasoning through dynamic policy optimization with rule-based rewards derived from dual-modality verification. These components synergize to shift safety alignment from harm perception to active reasoning. The framework achieves a 71.1\% improvement on VSE-HH, and improves by 59.1\%, 44.3\%, and 15.0\% on the image safety datasets MMBench, VLGuard, and FigStep, respectively. Our code and dataset are  available at \url{https://github.com/Emiya-syw/VideoSafety-R1.git}.
\textcolor{red}{Note: This paper contains harmful language and image examples, and reader discretion is recommended.}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VideoSafetyEval — a large-scale, real-world benchmark for video LLM safety (11.4k video–query pairs, 19 categories, 10 languages). Building on VideoSafetyEval, the authors propose a post-training defense framework, VideoSafety-R1, 
which integrates: (1) VideoSafetyThinking (VST) for chain-of-thought safety reasoning (46k samples), (2) Alarm Token-Guided Safety Fine-Tuning (AT-SFT), and (3) Safety-Guided GRPO with rule-based rewards.

### Strengths
1. To address the safety vulnerabilities of video LLMs, the authors introduce the VideoSafetyThinking dataset, which includes 46K video–query–reasoning triples and provides a valuable resource for the research community.
2. The defense strategy is well-designed: it follows a coherent end-to-end pipeline — from perception (alarm token) to reasoning (CoT) to reinforcement learning (GRPO with dynamic rewards) — offering a clear approach that convincingly shifts from passive refusal to safety-driven reasoning.

### Weaknesses
1. Overstatement: The authors claim that VideoSafetyEval is the first real-world safety evaluation benchmark for video LLMs. However, prior works such as VideoSafetyBench, SafeVid, and Trust-VideoLLMs have already explored safety evaluation in this domain.
2. Data collection, query generation, initial harmful labeling, and evaluation heavily rely on Qwen-Max, Qwen-Long, and GPT-4o, with limited enough human verification. Since these models have inherent reliability issues in video understanding, the resulting dataset may share similar distributions and biases with them, undermining the credibility of the evaluation.

### Questions
1. How are the hyperparameters, such as $\lambda_1$ and $\lambda_2$, determined? Do their values significantly affect the performance of the alarm token and the GRPO strategy?

2. Does fine-tuning video LLMs on safety dimensions impact their performance in other aspects, such as hallucination or overall video understanding ability (e.g., on Video-MME)?

3. Could the authors further analyze the diversity of reasoning chains in VST (e.g., average number of logical steps, template proportion) and the model’s performance across different languages?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a benchmark for Video-LLMs safety and two modifications to the post-training procedures of Video-LLMs to produce improved performance. The first improvement is Alarm Token-Guided Safety Finetuning, which learns learns alarm tokens that allegedly activate harm detection mechanisms. The second improvement is a safety-guided GRPO procedure that tweaks the RL procedure by introducing rule-based rewards based on policies.

### Strengths
S1: The VideoSafetyThinking dataset could be a nice contribution to the literature. If the data is of sufficient quality and interesting analyses about it can be performed, it could become a key aspect of discussion in the paper. 

S2: The proposed modifications are simple and, although not particularly innovative, seem effective.

S3: In general, the idea of using reasoning data and applying RL/tokenization can be interesting

### Weaknesses
W1: The claim to be the 'first large-scale, real-world benchmark for Video LLM safety' is a bit of an overstatement. The authors themselves, do cite competing (allegedly concurrent) benchmarks, besides the existence of [1,2,3]. Similarly the claim that the video modality introduces a vulnerability is not particularly novel: any form of finetuning (even on few benign samples in the same modality, even worse when a new modality is added) reduces the effectiveness of safety finetuning procedures. 

W2: Both the idea of adjusting the tokenization mechanism and of performing safe variants of RL are well known (e.g. besides [1,5], plenty are available). Maybe it could help having a detailed comparison between the various safety finetuning methods of RL to point out aspects of novelty or greater efficiency/effectiveness.

W3: The authors focus extensively on the impact of FPS on the results. However:

W3.1: In the section 'Effect of FPS' (line 373) they claim 'increased visual content introduces more safety risk'. The claim is handwavy: since their dataset does not focus on safety harms caused by short-length unsafe contents or malicious frames, it's not clear why the statement should be true. Presumably, the authors observed the correlation between higher number of frames in input and lower performance. However this is likely more related to the inability to handle long context, possibly high resolution, context windows or implicit downsampling mechanisms. Could the authors clarify this? 

W3.2: Both previusly indicated section and lines 409-414: RL is framed as a silver bullet for safety, while it is well known not to be. Further experiments are required to support the claim that some classes of models are more safe due to RL with respect to others. In particular, it would be necessary to ensure models are pre-trained and fine-tuned on the same distributions using different finetuning approaches. 

W3.3: The real purpose and meaning of the experiment in Sec 5.5 is unclear, especially without a clear description of how dynamic the videos produced are (indeed many image to video models produce minimal movements). 

W4: The data collection procedure is not particularly novel, and could be moved to appendix to leave more space to discuss and analyse the strengths of the work. 


[1] https://arxiv.org/pdf/2412.06878

[2] https://arxiv.org/abs/2409.19734 

[3] https://arxiv.org/pdf/2406.12235

[4] https://arxiv.org/pdf/2505.20065

[5] https://arxiv.org/abs/2412.10493

Minor: several sentences are verbose and unclear (e.g. .g. line 045: dynamic content interaction and temporal threat propagation; or not grammatically correct ‘then extend to 10 languages’ line 165. Or inappropriate terms, e.g. line 173 ‘surviving’ videos for videos that pass the filtering stage). These are just few examples.

### Questions
Q1: Could the authors discuss the similarities and dissimilarities of their token-guided finetuning approach with respect to [1]?

Q2: Could the authors produce a comparison in terms of performance with respect to SafeWatch and Holmes-VAD?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies safety in Video LLMs and introduces a new benchmark and defense pipeline. The authors build VideoSafetyBench (VSB-77k), a large, policy-grounded dataset that contains 77,646 video–query pairs, 19 subcategories across 6 risk categories and 10 language communities, and derive an 11.4k evaluation split VSB-Eval and a 46k post-training set with CoT (VSB-R1-46k). The author shows that adding the video modality substantially degrades safety, highlighting systemic vulnerabilities unique to video inputs. To address this, they propose VideoSafety-R1, combining Alarm Token-Guided Safety Fine-Tuning (AT-SFT) with Safety-Guided GRPO to inject modality-aware “alarm” tokens and then reinforce defensive reasoning with rule-based rewards tied to dual-modality verification. Extensive experiments validate VideoSafety-R1’s effectiveness on the adversarial VSE-HH benchmark, it improves VideoLLaMA3-2B’s DSR by 71.1% (from 18.4% to 89.5%). This approach also generalizes well to image safety benchmarks, boosting DSR by 59.1% on MMBench, 44.3% on VLGuard, and 15.0% on FigStep. It maintains strong utility and avoids over-defense while outperforming existing methods like SPA-VL and VLGuard.

### Strengths
Clear evidence that video hurts safety: The paper provides quantitative analyses showing that integrating video can sharply reduce DSR relative to text-only scenes (e.g., −79.4% for VideoLLaMA3-2B), and that harmful videos amplify adversarial effectiveness. This is an important and under-explored finding that justifies a video-specific safety agenda.

Well-designed benchmark and pipeline: VSB-77k is large, multilingual, and aligned to platform policies. The construction pipeline that consists of filtered video collection, multi-agent LLM annotation, and template-driven query generation is also clear. For metrics, the paper also sets DSR, Helpfulness score and FRR to systematically evaluate.

Effective two-stage defense with clear ablations: AT-SFT (learnable modality-specific alarm tokens + multitask alarm-classification objectives) and Safety-Guided GRPO are proved to be complementary. Ablations show each component helps and the combination yields the best overall safety, while keeping original video understanding essentially unchanged. The method substantially boosts DSR on VSB-Eval-HH and transfers across external safety benchmarks.

### Weaknesses
No coverage of “dynamic adversarial attacks”: Although the task is video, the method/evaluation effectively treats it as a set of sampled frames, rather than modeling sequence-level risk. The benchmark’s “harmful video” cases are largely explicitly harmful (e.g., direct fights, weapon displays). It does not include more realistic implicit dynamic attacks, for example, first ~10 frames benign, later ~20 frames contain fragmented harmful content or the video shows a seemingly legal tool but frame order implies harmful use. These higher-frequency, real-world attack patterns are absent, so the reported safety may overestimate robustness in deployment.

Token-load confound (not controlled): The paper attributes safety degradation to the video modality, but it does not rule out a simpler cause "large visual token loads". If one constructs a “pseudo-video” by repeating the same malicious image across many frames, the model receives far more visual tokens without adding temporal information. If defenses fail primarily because of attention/kv-cache saturation or context dilution under high visual token counts, then the reported vulnerability is not uniquely video-specific.

### Questions
1. Could the current “dual-modality verification” be extended to a temporal–text–vision triad (e.g., segment-level harmful annotations/rewards, time-aware alarm tokens, or temporal grounding constraints)?

2. Could you evaluate a control set where a single malicious image is duplicated to N frames (no temporal variation) and check if safety still collapses as N grows, that implicates token load, not temporal content.

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
4

### Summary
The paper shows that current video LLMs that look safe in text or image settings often fail once real video is involved, especially when the video itself contains harmful content. To measure this, the authors build VideoSafetyEval, an 11.4k sample benchmark covering 6 risk categories and 19 subcategories in 10 languages, with paired harmful and safe queries so they can test both under defense and over defense. To improve safety they introduce VideoSafetyThinking, a 46k training set with modality level tags, and VideoSafety R1, a two stage alignment method that first trains visual and textual “alarm tokens” to detect harmful signals, then uses safety guided GRPO style RL to reward correct detection and well formatted safe answers.

### Strengths
1. This paper makes substantial experiments. Authors evaluate a wide range of publicly available video LLMs in exactly the challenging setting they care about and they do so on a balanced benchmark that separates harmful video plus harmful query (VSE HH), safe video plus harmful query (VSE SH) and safe query for false refusals.
2. This paper is well-written and organized. The presentation of this paper is clear. 
3.  The task in this paper focus on safety issue of video LLM, which is important as a large share of the community is now training video LLMs without having a serious safety diagnostic. This paper hands them exactly that diagnostic together with a concrete way to fix at least the current class of failures.

### Weaknesses
I'd like to increase my rate if author can address my following concerns:

1. A large part of the safety gains is established with a model in the loop evaluator (Qwen based). The policy which indicates the definition of harmfulness is missing and so the definition of harmfulness is fully dependent on Qwen model, which may limit the future research. 
2. The base model used in ablation is unknown. It's possible gain of each component is different across different base model. 
3. The paper claims that harmful video semantics, not just more frames, cause the safety drop. But experiments do not systematically sweep frame numbers for the same clip across several models. A controlled frame sweep would make that claim firmer.

### Questions
Proposed model may have limit addressing long videos (> 60 sec), as the length of training video mostly are less than 60 sec. It's better to add this in limitation part.

### Soundness
3

### Presentation
2

### Contribution
2
