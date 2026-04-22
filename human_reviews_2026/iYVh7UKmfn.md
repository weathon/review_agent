# MMA-ASIA: A Multilingual and Multimodal Alignment Framework for Culturally Grounded Evaluation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 6

## Abstract
Large language models (LLMs) are now used worldwide, yet their multimodal understanding and reasoning often degrade outside Western, high-resource settings. We propose MMA-ASIA, a comprehensive framework to evaluate LLMs’ cultural awareness with a focus on Asian contexts. MMA-ASIA centers on a human-curated, multilingual, and multimodally aligned multiple-choice benchmark covering 8 Asian countries and 10 languages, comprising 27,000 questions; over 79% require multi-step reasoning grounded in cultural context, moving beyond simple memorization. To our knowledge, this is the first dataset aligned at the input level across three modalities: text, image (visual question answering), and speech. This enables direct tests of cross-modal transfer. Building on this benchmark, we propose a five-dimensional evaluation protocol that measures – (i) cultural-awareness disparities across countries, (ii) cross-lingual consistency, (iii) cross-modal consistency, (iv) cultural knowledge generalization, and (v) grounding validity. To ensure rigorous assessment, a Cultural Awareness Grounding Validation Module detects “shortcut learning” by checking whether the requisite cultural knowledge supports correct answers. Finally, through comparative model analysis, attention tracing, and an innovative Vision-ablated Prefix Replay (VPR) method, we probe why models diverge across languages and modalities, offering actionable insights for building culturally reliable multimodal LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MMA-ASIA, a comprehensive benchmark designed to evaluate the cultural awareness and multimodal reasoning capabilities of large language models (LLMs) in Asian contexts. It includes 27,000 human-curated, multilingual, tri-modal (text, image, speech) questions across 8 countries and 10 languages, with most requiring multi-step, culturally grounded reasoning. The authors propose a five-dimensional evaluation protocol assessing cross-lingual, cross-modal, and cultural consistency, along with grounding validity. They also introduce a Cultural Awareness Grounding Validation Module to detect shortcut learning and a Vision-ablated Prefix Replay (VPR) method to analyze model divergence. Extensive baseline evaluations across 14 model families provide insights into current models’ strengths and weaknesses in culturally aligned multimodal understanding.

I find this work timely and relevant, especially as multimodal LLMs expand globally but remain under-evaluated in non-Western contexts. The benchmark’s tri-modal alignment and five-dimensional evaluation are impressive contributions, though I would have appreciated broader linguistic coverage, detailed study of literature review and deeper analyses of cultural-topic interactions.

### Strengths
1. Introduced the first tri-modal, multilingual benchmark explicitly designed for Asian cultural contexts, covering multiple modalities (text, image, speech) aligned at the input level for direct cross-modal evaluation.
2. Proposed a comprehensive five-dimensional evaluation protocol capturing diverse aspects of cultural awareness, which incorporates a Cultural Awareness Grounding Validation Module to detect shortcut learning and ensure true reasoning.
3. Provided extensive baseline comparisons across some model families.
4. Provided detailed analysis and ablation studies.

### Weaknesses
1. The benchmark covers *only 8 Asian countries and 10 languages*, which is quite limited considering that *Asia has around 50 countries*. This represents roughly *15–20%* of the region’s linguistic diversity. Moreover, the *country selection appears heavily skewed toward Southeast Asia*, with most *Central and Southwest Asian* countries missing. Notably, *Arabic*, one of the most widely spoken Asian languages, is also absent.
2. Lines 46–47 state that the benchmark aims to address *(ii) inadequate representation of low-resource Asian languages)*. However, given the limited geographical and linguistic scope mentioned above, this issue *still persists* and undermines the benchmark’s stated objective.
3. The related works section overlooks several key studies relevant to cross-cultural and multilingual benchmarks. The following papers should be reviewed and cited appropriately: 10.18653/v1/2024.emnlp-main.671, openreview.net/forum?id=DFr5hteojx, openreview.net/forum?id=k3gCieTXeY, 10.48550/arXiv.2505.24119, 10.1145/3701716.371546, openreview.net/forum?id=yxzVanFoij, 10.18653/v1/2025.acl-long.919, 10.48550/arXiv.2412.04261, 10.18653/v1/2025.findings-naacl.341, openreview.net/forum?id=N38Sny0XJi, 10.48550/arXiv.2504.05747
4. Lines 51–52 mention *tri-modal items (textual question, image+question, and Text-to-Speech (TTS)-spoken question)*, but *no information is provided on the distribution of samples across modalities*. Additionally, *there are no evaluation results reported per modality*, which makes it difficult to assess whether multimodal balance was achieved or if one modality dominates.
5. The benchmark explores only *multiple-choice question (MCQ)* formats. Incorporating *open-ended or generative evaluation* tasks could provide a more comprehensive understanding of model reasoning and cultural grounding.
6. In Figure 2, *performance values appear nearly identical* and are *difficult to interpret*. It would be clearer and more informative to present this data in *tabular format* instead of using the current visualization.
7. Claims to provides extensive baseline comparisons across 14 major multilingual and multimodal model families, but very unclear from curent state of the paper.

### Questions
1. Why were only 8 Asian countries and 10 languages chosen, given Asia’s vast linguistic and cultural diversity?
2. How does the benchmark justify calling itself “Asian” when Central and Southwest Asian regions are largely excluded? Was Arabic intentionally omitted, and if so, what was the rationale?
3. How does the benchmark address its stated goal of improving representation for low-resource Asian languages despite this limited coverage?
4. Could you clarify the distribution of samples across the three modalities (text, image+text, TTS)?
5. Are there separate evaluation results for each modality to assess cross-modal consistency?
6. Why did you focus solely on multiple-choice questions instead of including open-ended or generative evaluation tasks?
7. Are there plans to extend the benchmark to more countries, languages, or modalities in future work?
8. Could you provide a more detailed analysis of cultural topic effects, how performance varies across themes like religion, tradition, or social norms?
9. Can you include language-level analyses, showing how performance differs across languages and model-language familiarity?
10. Could you add a language–topicperformance analysis to examine whether certain cultural topics are harder for specific languages?
11. Please improve Figure 2 by presenting the results in a tabular format with clear numerical values.
12. Will you update the related work section to include missing key studies (e.g., EMNLP 2024, ACL 2025, and OpenReview papers listed)?
13. How do the benchmark results compare when evaluating cross-cultural generalization versus within-culture accuracy?
14. Did you observe any bias patterns in model outputs linked to specific cultural or linguistic groups, and if so, how were they handled?
15. Check if model size has any impact of the performance and bias?
16. Add detailed results on 14 model fmailies as claimed. Currently, it is very unclear.


Minor: L44: Fix citation formatting → “Balepur et al. (2024); Molfese et al. (2025); Zheng et al. (2023).” 
and some more such error across the paper.

### Soundness
3

### Presentation
2

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
MMA-ASIA introduces a tri-modal (text, image, speech), multilingual benchmark for evaluating “cultural awareness” across 8 Asian countries, 10 languages, 27k MCQ items with aligned text/VQA/spoken versions and a five-axis protocol (disparity, cross-lingual, cross-modal, generalization, grounding). The paper claims 79%+ multi-step reasoning, human-curated knowledge points, and an LLM-as-judge module to detect shortcut reasoning. Main findings: accuracy is generally modest (<80% for most SOTA models), English often outperforms low-resource languages, cross-lingual and cross-modal consistency is low, and spoken accents sometimes help.

### Strengths
- Careful alignment and scope: same items across three modalities and languages (including English + local language) with a clear five-dimension protocol; helpful comparison table vs prior datasets (Table 1).
- Grounding checks: knowledge points per item and Rationale Unfaithfulness Rate (RUR) expose shortcut-driven wins; the paper shows non-trivial RUR even for strong models (Fig. 3).
- Analyses beyond accuracy: cross-lingual and cross-modal consistency plots (Fig. 4), attention heatmaps and a Vision-ablated Prefix Replay diagnostic to probe visual-token effects.

### Weaknesses
W1. Impact & new insight are underdeveloped.
- The headline observations (English vs. low-resource gaps; text > VQA > speech; low cross-lingual/modal consistency) are valuable but mostly reconfirm known trends (resource asymmetry, multimodal transfer difficulty) without a strong new causal or design insight that changes how we build or evaluate M(L)LMs. The paper’s own figures support the descriptive nature (accuracy bars in Fig. 2, consistency radars in Fig. 4), but the analysis often stops short of actionable, general lessons (e.g., ablations that isolate dataset artifacts vs. model architecture effects).

W2. Extensiveness: breadth is good, but depth of evaluation is uneven.
- The benchmark spans 8 countries/10 languages and modalities, yet downstream corroboration is limited to MCQ-style accuracy/consistency; there’s no transfer to real-world tasks (retrieval-augmented QA, interactive grounding) to show external validity. The “generalization” analysis is interesting but narrow (two models, English only).

W3. Comparative baselines and protocol clarity need to be crisper in the main text.
- A lot of the critical experimental protocols (prompts, decoding, seeds, hardware) live in the appendix; given the cross-model comparisons, a concise main-paper protocol box would improve trust in compute-matched fairness (prompts shown in Table 7; settings summarized in A.6).

W4. Grounding validator depends on an LLM judge; robustness discussion is brief.
- RUR uses a single LLM-as-judge with a small agreement study; while pragmatic, the approach could shift with judge choice and language. The paper notes high agreement and chooses a single judge for cost, but more sensitivity analysis (judge swaps, multilingual judging) would bolster confidence.

### Questions
- What’s the new actionable takeaway? Beyond reporting gaps (Fig. 2, Fig. 4), can you show design interventions a model developer could adopt (e.g., accent-aware speech preconditioning, cross-modal consistency losses) and quantify improvements on MMA-ASIA?
- Protocol box in main paper. Please surface a one-page box (prompts, seeds, decoding, image resizing, hardware) so cross-model comparisons are evidently compute-matched without digging through the appendix. (Table 7, A.6 are close—bring them up front.)
- Judge robustness. How sensitive is RUR to the choice of judge (Claude vs. GPT-4o vs. Gemini) and to judging in local languages rather than English rationales? Consider a small triangulation table.
- Broaden generalization analysis. The step-decomposition study is run on two models in English; please extend to additional models/languages and report where “knowledge is present but integration fails” vs. “knowledge missing”.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces MMA-ASIA which is a comprehensive evaluation framework for evaluating cultural awareness of LLMs. The newly proposed benchmark that is a part of this framework consists of 27k QA pairs in 10 languages across 8 Asian countries. The benchmark consists of data across 3 modalities of image, text and audio. The proposed 5-dimensional evaluation protocol consists of cultural-awareness disparities across countries, cross-lingual consistency, cross-modal consistency, cultural knowledge generalization, and grounding validity. The paper investigates multiple LLMs. VLMs and audio models to understand and compare their capabilities regarding cultural awareness in this context.

### Strengths
1.	Interesting direction: The proposed multilingual multimodal benchmark explores an interesting direction of analyzing the capacity of modern multimodal models in handling data from Asian languages. 
2.	Inclusion of audio: The inclusion of audio in the proposed benchmark is a novel contribution as the existing multilingual QA datasets mostly focus on visual and text data.
3.	Clear writing: The writing of the paper is clear, and the concepts explained and performed analysis is easy to understand for the most part.
4.	Detailed supplementary material: The supplementary material is extensive and detailed, as it includes additional key contributions and other analyses.

### Weaknesses
Major:
1. Limited question types (MCQ only): The questions included in the proposed benchmark comprise Multiple Choice Questions only. However previous related work has included other type of questions like Short Question Answers, Long Question Answers (Captions) and True/False Question Answers. MCQ has Implicit bias and randomness that can often lead to issues in analysis. Previous works have tackled this by shifting the order of options in multiple evaluation iterations and converting the MCQs to open-ended questions.
2. Issues in Data Quality: The supplementary material shows a few examples of questions in different languages. However, there are grammatical/structural issues in the provided examples. For example, in the Hindi question example, the last 3 lines of the shown question do not make sense in sentence structure and grammatical context. 
3. Limited scope in terms of number of languages: The authors propose a benchmark to represent Asian languages but only include 10 languages which are not enough to represent Asia in general. Benchmarks that are not Asia-specific (CVQA [1] and ALM-Bench [2]) include more Asian languages. Also the proposed benchmark does not include languages like Bengali, Arabic, Punjabi and Urdu which are among the most spoken languages in Asia.  
4. Inconsistency in model selection: The authors choose an inconsistent array of models which can hurt the overall performance analysis. The models shown vary in scale (3B v/s 11B v/s 32B...) as well as use of reasoning/non-reasoning models in the same analysis. 
5. Lack of model scaling analysis: The authors do not perform a model scaling analysis, i.e., using different sizes of the same model (for example, Qwen2.5-VL 3B v/s 7B vs 32B), which is a very helpful study included in benchmark papers.
6. Language support for various included VLMs: The authors do not include KIMI, DeepSeek and InternVL in any VQA analysis for non-English languages. However, as shown in ALM-Bench [2], they can accept these tokens. The performance might not be as good, however it does give some output.
7. Missing analysis of “cultural themes”: The authors mention the use of 9 cultural themes. However, there is no analysis of said themes or how they impact the performance of models.
8. Audio comprising 100% TTS: A component of audio QA should be live recorded, instead of having 100% TTS.

Minor:
1. The term “RI” used in Table 1 has not been explained/addressed/elaborated.
2. The captions for Fig. 1 and Table 1 are incomplete and short
3. More details are required regarding the selection of cultural categories and languages. Sec. 3.2 only states that they were decided through “collaborative discussions”
4. line 309: what is "itsemphknowledge"

### Questions
1.	The authors state that issues encountered in machine translation were addressed by human reverification. What types of issues were faced and how often? A statistical analysis would be helpful. How were the issues addressed?
2.	The authors show results on more VLMs as compared to LLMs. However, every VLM has a base LLM, some of which were excluded. What is the reason for that?
3.	What is the authors’ hypothesis behind the low performance of GPT on the Chinese language compared to other models (especially open-source models) when previous works like CVQA [1] and ALM-Bench [2] show otherwise?
4.	Why are models like GPT which have the capacity to accept audio tokes excluded from the audio QA performance analysis?
5.	The authors provide analysis of performance of models based only on MCQ final answer, however the reasoning output by the models is not evaluated/analyzed. Are there any experiments/evaluations that show the efficacy and quality of generated reasoning?
6.	How did you make sure that the VQA questions require visual input and cannot be answered without it (same for audio QA)?
7.	How were the 56 key words mentioned in Sec 3.3 obtained?
[1] Romero, David, et al. "CVQA: culturally-diverse multilingual visual question answering benchmark." Proceedings of the 38th International Conference on Neural Information Processing Systems. 2024.
[2] Vayani, Ashmal, et al. "All languages matter: Evaluating lmms on culturally diverse 100 languages." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MMA-ASIA, a benchmark designed to evaluate cultural awareness across Asian languages and cultural contexts. The benchmark spans 8 Asian countries and 10 languages, and covers three input modalities: text, image, and speech. In addition to benchmarking model performance, the authors also analyze why models behave inconsistently across languages and modalities, using a vision-ablated prefix replay method to reveal modality-specific biases and divergence patterns.

### Strengths
1. Multilingual and multicultural evaluation is increasingly essential as foundation models are deployed globally, and South & Southeast Asian languages remain significantly underserved in existing benchmarks.
2. The benchmark thoughtfully includes diverse bias categories, capturing cultural, demographic, and contextual nuances relevant to Asian societies.
3. The experimental evaluation is extensive, covering multiple languages, modalities, and model families, which strengthens the validity of the findings.

### Weaknesses
1. The contribution is primarily benchmark-focused, with limited methodological novelty beyond the dataset and evaluation setup.
2. The analysis would benefit from a deeper investigation into failure cases and model behavior across languages and modalities.
3. The writing and organization could be improved for clarity and flow, particularly in framing the motivation and guiding the reader through the core contributions.

### Questions
1. Have you evaluated the stability of the LLM outputs across repeated queries? In other words, how sensitive are the results to sampling variability, and what level of uncertainty do the models exhibit across trials?
2. Could you elaborate on the significance of the proposed metrics? In particular, how do they differ from metrics used in prior work, and what specific advantages or insights do they provide in the context of cultural awareness evaluation?

### Soundness
3

### Presentation
3

### Contribution
3
