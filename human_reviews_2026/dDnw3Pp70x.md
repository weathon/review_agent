# TIPO: Text to Image with Text Pre-sampling for Prompt Optimization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
TIPO (Text-to-Image Prompt Optimization) introduces an efficient approach for automatic prompt refinement in text-to-image (T2I) generation. Starting from simple user prompts, TIPO leverages a lightweight pre-trained model to expand these prompts into richer, detailed versions. Conceptually, TIPO samples refined prompts from a targeted sub-distribution within the broader semantic space, preserving the original intent while significantly improving visual quality, coherence, and detail. Unlike resource-intensive methods based on large language models (LLMs) or reinforcement learning (RL), TIPO provides computational efficiency and scalability, opening new possibilities for effective, automated prompt engineering in T2I tasks.

We provide visual results, human preference report to investigate TIPO's effectiveness. Experimental evaluations on benchmark datasets demonstrate substantial improvements in aesthetic quality, significant reduction of visual artifacts, and enhanced alignment with target distributions along with significant human preference proficiency. These results highlight the importance of targeted prompt engineering in text-to-image tasks and indicate broader opportunities for automated prompt refinement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
TIPO (Text-to-Image Prompt Optimization) is an efficient framework designed for the automatic refinement of simple user inputs into rich, detailed prompts for text-to-image generation. The core intuition is to align the optimized prompt with the text distribution used in text-to-image model training. Compare with using expensive LLM rewriting or reinforcement learning approach, TIPO trains a lightweight multi-task language model, employing a pre-sampling mechanism to expand the original user prompt. This multi-task training includes pretext tasks for handling both tag-based and natural language inputs. Experiments demonstrate that TIPO achieves superior image quality, stronger text alignment and a 62.8% human preference win rate against strong baselines, while also providing up to a 29.4% runtime efficiency improvemen.

### Strengths
1. The core intuition is that optimal prompts must align with the large-scale text distributions used to train T2I models, which can reduce the mismatch between the training and inference. It is novel and straightforward. Moreover, the method is designed to be universal, leveraging its large-scale curated corpus of over 30 million text descriptions to ensure compatibility across various T2I models.

2. The method employs a lightweight language model to expand the use prompts. This computational efficiency is a major practical advantage over resource-intensive LLM-based or RL-based methods.

3. The experimental results are solid. It is conducted on both in-domain and out-of-domain tasks, which shows that the method is general.

### Weaknesses
1. The limitation of this work is that it needs the large-scale open-sourced text-to-image training data to make sure that prompt expanding is robust. But some current models are trained use higher-quality close-sourced data, often containing some synthetic data. It is impossible  to get the reasonable data distribution of these models, making the proposed method ineffective.

2. The framework does not consider any feedback from the generated images, which means that it cannot adaptively improve the prompts according to user feedback or model feedback.

### Questions
What sampling method do you use, top-p or greedy decoding?

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
The proposed TIPO is a prompt optimization method for text-to-image (T2I) models that leverages a lightweight pre-trained model to automatically expand users' simple inputs into detailed and semantically rich prompts. The core idea is to transform user input prompts into prompts that align with the text distribution of T2I model training data, thereby significantly improving visual quality, coherence, and detail while preserving the original intent. The experiments report that prompt refinement can be performed in a lightweight, fast, and general manner across various evaluation metrics such as FDD and Aesthetic Score.

### Strengths
- The implementation approach, which uses a LLaMA-based 200M parameter model for pre-training and does not depend on large-scale LLMs or reinforcement learning, is attractive from a practical standpoint. Additionally, it does not involve fine-tuning of T2I models.

- For tag-based prompts, TIPO achieves the best FDD (0.2282), and for natural language-based (short) and truncated long prompts, it demonstrates the best or second-best performance in Aesthetic Score and AI Corrupt Score with stable performance.

- Beyond Stable Diffusion family models, performance improvements are confirmed on diverse T2I models including FLUX.1-dev, Omnigen2, Lumina-2, and HiDream-I1. The fact that TIPO is effective even for models with undisclosed training data demonstrates the high generalizability and robustness of the proposed method.

- While conventional methods indirectly optimize prompts using reward models, the finding that better prompts can be created simply by directly aligning with the data distribution learned by T2I models, as in the proposed method, provides an interesting insight.

### Weaknesses
- The verification of "distribution-aligned" is indirect. While claiming to "align with T2I training distributions," the paper does not directly measure the distance of text distributions themselves (e.g., KL divergence of embedding distributions or perplexity differences). The alignment is primarily inferred from image-side metrics (such as FDD) and human comparisons, lacking direct measurement of text distribution.

- In OOD settings, while diversity (Vendi Score) improves significantly, there is an issue that GPT-generated prompts are accurate but lack diversity. Although TIPO optimization adds additional details that harmonize with the original themes and significantly improves the diversity of generated images, the relative positioning of aesthetic scores and corruption rates compared to some baselines should be explicitly stated as a limitation of generalizability.

- While the paper presents inference time cost comparisons (Table 4 showing up to 29.4% speedup), and Table 6 shows training settings for each TIPO model (GPU types, training time, datasets), there is insufficient direct quantitative comparison with RL-based methods such as Promptist and PAE regarding training costs. Explicit comparison is desired for fair evaluation.

- The adopted metrics are primarily image-centric, such as Aesthetic Score, AI Corrupt Score, Vendi Score, and FDD, and do not include automatic metrics that measure text-image alignment such as CLIPScore or GenEval. There is insufficient quantitative evidence to support the claimed "stronger text alignment."

### Questions
1. Can you add metrics that directly measure text distribution alignment (e.g., embedding distance from training data or perplexity)?

2. While diversity improves in OOD environments, do you have any causal analysis or improvement strategies regarding the relative performance differences with other baselines?

3. Can you provide a quantitative comparison of training computational costs and resource requirements (GPU memory, training time, etc.) with RL-based methods such as Promptist and PAE?

4. Can you add evaluation results using automatic metrics that directly measure text-image alignment, such as CLIPScore or GenEval?# Review of "TIPO: Text to Image with Text Presampling for Prompt Optimization"

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
TIPO (Text-to-Image Prompt Optimization) proposes an efficient approach to automatic prompt enhancement for T2I models. Instead of fully rewriting user inputs with a large LLM, TIPO uses a lightweight, multi-task pretrained language model to perform text presampling before image generation: it retains the original user prompt and appends a structured, detail-rich continuation that better matches the text distribution T2I models are actually trained on.

### Strengths
The paper’s key strength is conceptual clarity: it treats “bringing the prompt back to the T2I training text distribution” as the central objective. To achieve this, it adopts a lightweight model that does prefix-preserving, suffix-expansion prompt optimization, which (i) avoids the off-topic drift often seen in full LLM rewrites, and (ii) is cheaper, faster, and more deployable than RL-based prompt optimization. The experimental section is also solid: it covers in-domain and out-of-domain settings, multiple T2I backbones, and evaluates with FDD, Aesthetic Score, AI Corrupt Score, Vendi, plus human preference, so the claimed gains are reasonably well supported.

### Weaknesses
The method is still distribution-dependent: if the user prompt is very niche, domain-specific, or stylistically unusual, the expanded prompt may not be reliable. Once the prompt is made more detailed, the generation space narrows, and Vendi indeed drops in some settings. For closed-source or stylistically distant T2I models, the “distribution-aligned” expansion can sometimes misfire, leading to slight aesthetic regressions. Finally, the current TIPO is a generic optimizer — it doesn’t condition on user profile, project domain, or target style, so it cannot precisely do “expand this, but keep my style.”

### Questions
- Since a single TIPO model is used for multiple T2I backbones (rather than training one optimizer per T2I model), does this create subtle distribution-mismatch issues for models whose caption/style format differs from the main training corpus?
- Does the method exhibit a scaling law? In other words, if we move from the 200M model to larger variants, do we see monotonic gains in fidelity, aesthetic score, and OOD robustness — and where is the efficiency/quality turning point?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose a new method for prompt rewriting for text-to-image generation. They design a specific format to structurally combine natural language and tag-based prompts and implement a pre-sampling algorithm that progressively refines arbitrary, coarse user input into organized, fine-grained prompts by training LLMs. They evaluate their models against prior baselines as well as SOTA LLMs on a large variety of text-to-image models as well as native multimodal models using standardized metrics and human preference survey.

### Strengths
1. This paper has very comprehensive experiments, especially the range of text-to-image models and native multimodal models that they test with. And the fact that it is effective even for models with self-refinement capabilities also shows good practical applicability of their method.
2. ELO rating is an interesting and creative way to showcase human evaluation comparison results among multiple models. I don’t think this is a standard metric for image generation evaluation yet, but I think it should become one.
3. The description of their training recipe is also very clear and easy to follow, making the method also very adoptable.
4. The inference speed is very fast.
5. The models are trained with relatively small GPUs in a relatively short amount of time, which makes the method resource efficient.

### Weaknesses
1. The improvement on benchmarks that TIPO brings is not very consistent across the board and is sometimes very marginal.
2. Mild writing suggestions: Section 3 does not really provide much information and is not very well connected to the rest of the paper. Given how packed the remainder of the paper is, I would suggest either shortening that section or removing it entirely and just explaining the notations when using them later on.
3. Minor: there are some inconsistencies in citation styles in the first paragraph of the introduction.

### Questions
From the qualitative examples, it seems like the diversity of the images should be improved after rewriting. However, this is not reflected with the Vendi scores. Do the authors have an explanation of why this may be the case?

### Soundness
4

### Presentation
3

### Contribution
3
