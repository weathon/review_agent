# T2G-Reasoner: Deep Reasoning for Text-to-Gloss Translation

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
In this work, we present T2G-Reasoner, a framework equipped with a reasoning mechanism to improve text-to-gloss translation (T2G), where gloss is a written record of sign language. 
The reasoning LLMs have achieved remarkable success in a range of NLP tasks, benefiting from their strong generalization capability stemming from pretraining on massive data.
However, incentivizing the reasoning capabilities for the T2G task is challenging due to the absence of gloss information in LLMs' pretraining. 
Considering shared lexical concepts between two languages, we leverage an advanced LLM to extract word-level alignments as the T2G reasoning process.
Instead of directly generating sign language gloss, the proposed method structures the model's output into two distinct components, \emph{i.e.}, the word-level alignments and the final gloss translation.
T2G-Reasoner adopts a two-stage training strategy, \emph{i.e.}, SFT-based imitation and RL-based exploration.
The T2G-Reasoner model is first fine-tuned on the synthetic reasoning data, which establishes a foundational layer of reasoning capability.
As the synthetic reasoning data may be of lower quality, the T2G-Reasoner model further leverages the RL algorithm to autonomously discover optimal word-level alignments.
Extensive experiments on two benchmark datasets show that the proposed T2G-Reasoner achieves significant performance improvements.
Additionally, our T2G-Reasoner exhibits great potential to address out-of-vocabulary (OOV) challenges in T2G.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper approaches part of a common sign language generation pipeline, text to gloss translation, by applying SFT-based distillation of GPT-4o and then outcome-based RL with GRPO with format and BLEU reward. This achieves some gains (2-3 BLEU over baseline, <1 BLEU over prior work).

### Strengths
I've seen works that do in-context learning for text to gloss before, but never chain of thought outcome-based RL as far as I can remember. It's a good strategy since it's known to be more data-efficient.

The paper is written clearly. Related work is a good summary without going into unnecessary depth.

### Weaknesses
The main issue I have with that paper is that the novelty/contribution isn't large enough. It applies distillation from an API model (GPT-4o) and then GRPO to text to gloss, and it works a little bit (very little compared to other works, more compared to baselines). Most of the improvement in the baseline, in light of Table 3, seems to be from distilling GPT-4o. I'm not sure that I've seen text to gloss distillation per se, but I've seen gloss translation zero-shot or from in-context examples which isn't that different. There is some contribution from confirming that GRPO helps for this task but I don't think it suits the ICLR venue.

I'm especially inclined to think this because the PHOENIX/CSL-Daily datasets are a bit of a trope, where the numbers in these big comparison tables tend to creep up but nothing meaningfully changes. I'm sure the experiments are sound and well-executed but the broader significance is limited (especially when text to gloss is uncritically assumed as a given, and when glosses don't really make sense when you get out of these toy benchmarks https://arxiv.org/abs/2403.02563).


A bunch of the ways things are described in the paper also aren't really technically correct / best practice for the field, but these are minor/fixable points.

For example:
* The paper acts as if "gloss" is only a term for sign language, e.g. "where gloss is a written record of sign language" and the title of the paper not mentioning sign language. But glosses are used for spoken languages too. (Maybe the methods would apply there too!)
* "Sign language is the primary means of communication for the deaf": This would be more correct as "many deaf people", or "Deaf" to signify culturally Deaf. There are a lot of means of communication and a lot of diversity among deaf/hard of hearing people, e.g. if you're talking about age-related deafness then sign language is not the primary means of communication, and for a statement like this to be true then you have to start looking at the rates of different causes of deafness etc.
* "Translation between sign language and spoken language is an important research topic": This should really be "signed language" if you're using it in contrast to "spoken language", or "sign languages" and "spoken languages" plural.
* "the first step of the sign language generation task, named text-to-gloss translation": This isn't the first step of the task; it's the first step of a common approach to the task that uses a cascaded pipeline.

### Questions
I'm open to having my opinion changed about the magnitude of contribution in the paper.

### Soundness
4

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
4

### Summary
This manuscript addresses the text-to-gloss (T2G) translation task, which aims to generate high-quality gloss sequences with the assistance of sign language experts. To achieve this, it explore sthe reasoning capabilities of large language models (LLMs) for the T2G task. Specifically, this manuscript decomposes the T2G task into two stages: a word alignment process and a translation process. In the first stage, it utilizes advanced LLMs (e.g., GPT-4o) to generate word-level alignments, which serve as a reasoning-augmented dataset for guiding the supervised fine-tuning (SFT)-based imitation process. Next, it designs a reward function that accounts for both format and translation quality, which is integrated with a gradient-based reward optimization (GRPO) method. Experimental results on two public datasets demonstrate the effectiveness of the proposed approach.

### Strengths
- S1. This manuscript is well-organized and easy to follow, providing a detailed description of the dataset construction process as well as the method designs.
- S2. The manuscript offers an intriguing perspective on T2G tasks by leveraging reasoning capabilities to guide the translation model’s learning. The proposed method is well-designed and effectively illustrates the reasoning process in T2G, enhancing the model's interpretability.
- S3. Experimental results demonstrate the effectiveness of the proposed method, showing improved translation performance over previous approaches and offering valuable insights into out-of-vocabulary (OOV) behavior.

### Weaknesses
- W1. The proposed method outperforms the previous state-of-the-art (SOTA), but it uses a significantly larger translation model (Qwen 2.5-3B) compared to the previous SOTA [Yao et al., 2024], which utilizes only 3-5 transformer layers. This makes the contribution somewhat incremental.
- W2. The manuscript evaluates T2G quality using two public datasets, focusing solely on translation performance. However, it remains unclear how word alignment quality impacts translation accuracy, and how well the proposed method performs in more challenging scenarios. For instance, it is not addressed whether the generated glosses can assist other sign language processing tasks, or how the introduced inference stage improves related tasks.
- W3. The proposed method appears less relevant to the direction of translation, and I am more interested in the G2T (Gloss-to-Text) translation performance to explore its potential in sign language translation.
- W4. The writing needs improvement. For example, the mapping $f(\theta)=X\rightarrow Y$ is inconsistent with Equation (1), and it is unclear why the final award range in Equation (6) is from 1 to 3.

### Questions
- In Table 7, the manuscript demonstrates that the proposed method can generate low-frequency glosses. I am curious whether this also results in the generation of non-existing glosses, and how the ratio of these changes after introducing the reasoning process.
- As noted in the weaknesses, I still believe the proposed method is not directly related to the translation direction. Therefore, I am particularly interested in its effectiveness for the translation direction, which has garnered more attention recently.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
T2G-Reasoner applies GRPO (SFT + RL) in a two-stage pipeline—first fine-tuning on synthetic word-level alignments, then RL to refine gloss output—effectively directly adapting GRPO to T2G translation. The core idea is intuitive: break gloss generation into alignment + translation, using RL to correct noisy synthetic supervision.

### Strengths
The primary value of T2G-Reasoner lies in empirically validating GRPO’s feasibility for structured sign language translation (SLT/T2G). By decomposing gloss generation into word-level alignment and final translation, then applying SFT on synthetic reasoning traces + RL for refinement, the work provides a practical proof-of-concept that GRPO—originally designed for open-ended LLM reasoning—can be effectively adapted to sequence-to-sequence tasks with lexical alignment constraints.

### Weaknesses
While the decomposition is natural and GRPO integration straightforward, novelty is limited—it largely repurposes standard LLM reasoning + RL techniques to a structured output task. Gains are real but expected given the supervision quality. No deep methodological innovation; strong execution of an obvious strategy.

### Questions
- Lack of Core Innovation: The method is essentially LLM + SFT on synthetic alignments + GRPO for refinement—a direct transplantation of GRPO to T2G. What is the essential technical departure from applying GRPO to any structured prediction task? Without a novel objective, architecture, or alignment mechanism, the contribution reduces to application, not advancement.
- Gloss-Dependent Generalization: The pipeline heavily relies on gloss-annotated data for both synthetic alignment generation and evaluation. On gloss-free benchmarks like OpenASL, the entire reasoning trace construction fails. How does the method adapt to real zero-shot or gloss-free SLT? This severely limits claimed generalizability.
- Sub-SOTA Performance & Weak Validation: On P14T-Dev, T2G-Reasoner (29.05) underperforms CV-SLT (29.27)—hardly a compelling result. No ablation across baselines (e.g., applying GRPO to CV-SLT, C2RL, etc.) leaves effectiveness unverified beyond the authors’ own setup.
- Unexplained RL Collapse (Fig. 3C): Training with RL only (no SFT) yields drastically shorter thought chains. This suggests reward hacking or mode collapse, not efficient reasoning. Why does GRPO fail without SFT pre-alignment on SLT? Is KL control or reward shaping insufficient? This behavior undermines the robustness of the RL component.

### Soundness
3

### Presentation
3

### Contribution
2
