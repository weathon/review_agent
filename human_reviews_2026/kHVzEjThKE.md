# What Do Large Language Models Know About Opinions?

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
What large language models (LLMs) know about human opinions has important implications for aligning LLMs with human values, simulating humans with LLMs, and understanding what LLMs learn during training.  While prior works have tested LLMs' knowledge of opinions via their next-token outputs, we present the first study to probe LLMs' internal knowledge of opinions, evaluating LLMs across 22 demographic groups on a wide range of topics.  First, we show that LLMs' internal knowledge of opinions far exceeds what is revealed by their outputs, with a 52-66\% improvement in alignment with the human answer distribution; this improvement is competitive with fine-tuning but nearly 300$\times$ less computationally expensive.  Second, we find that knowledge of opinions emerges rapidly in the middle layers of the LLM and identify the final unembeddings as the source of the discrepancy between internal knowledge and outputs.  Third, using sparse autoencoders, we trace the knowledge of opinions in the LLM's residual stream back to attention heads, and we identify specific attention head features that selectively encode different demographic groups. Through steerability experiments, we show that manipulating these features causally alters the LLM's outputs, aligning them more or less closely with different groups. These findings open new avenues for building value-aligned and computationally efficient LLMs, with applications in survey research, social simulation, and human-centered AI.  Our code is available at https://github.com/schang-lab/llm-opinions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the internal representation of human opinions in LLMs, studying how these models align with human values and learn during training. Unlike previous studies that focus on output-based evaluations, the authors probe internal model states across 22 demographic groups and a broad range of topics. They find that LLMs possess significantly more accurate internal knowledge of opinions than what is reflected in their outputs, achieving up to 59% better alignment with human responses at a fraction of the computational cost of fine-tuning. The study identifies the emergence of opinion knowledge in middle layers and attributes output discrepancies to final unembedding layers. Using sparse autoencoders, the authors trace opinion-related features to specific attention heads linked to demographic distinctions. 

This is a clearly written paper with sound experiments and clear contribution points.

### Strengths
- This work addresses the important topic of how investigating internal representations of human opinions across 22 demographic groups.
- Though methodologies used, such as linear or multinomial probing, aren’t original, they were appropriately applied to derive results that are reliable.
- The insights gained through the experiments, such as knowledge of opinions emerge in the middle layers, are useful for the research community.
- The writing is clear.

### Weaknesses
- The analysis is conducted on US survey data only.

### Questions
- Given the emphasis on the need for pluralistic AI systems, why were the experiments conducted on US survey data only?

### Soundness
4

### Presentation
4

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
The paper studies what LLMs “know” about human opinions beyond their surface outputs. Using OpinionQA and SubPOP (US public-opinion surveys) across 22 demographic groups, the authors probe residual-stream activations layer-wise (multinomial logistic vs. MLP probes) to predict full answer distributions to survey questions. Key findings: (i) probes extract substantially more opinion knowledge than next-token probabilities (≈50–60% lower KL), (ii) this knowledge emerges sharply in middle layers (≈layers 10–15), and (iii) the unembedding layer explains much of the gap between internal knowledge and outputs; finetuning only the final linear head recovers probe-level gains and achieves a large fraction of LoRA’s improvements with far fewer parameters. They further train SAEs over attention-head activations and claim group-selective features concentrated in middle layers.

### Strengths
- Clear research question & careful operationalization. Evaluating distributional alignment (KL over options) rather than single labels is appropriate for opinions and more informative than argmax accuracy.

- Layerwise analysis identifies where knowledge concentrates. The middle-layer emergence is consistent and useful for targeted interventions.

### Weaknesses
I am unconvinced by the novelty and significance of this work. I personally found that the paper largely re-applies established probing methodology to a new label space (opinion/poll distributions) and reports patterns that are unsurprising in light of prior probing literature.

- Major Concerns

   - Incremental Contribution: Most findings (e.g., information concentration in mid layers; linear vs. MLP probe behavior) closely mirror prior probing results. 

   - Last-Layer Adaptation Claim: The observation that finetuning or adapting only the output head (unembedding/last layer) recovers most gains is well known from adaptation/calibration lines of work. Presenting this as a key result does not constitute novelty.

   - Probe Design/Analysis: The linear–MLP comparison and the claim that hidden states encode answer distributions are not surprising; no new probing technique, control, or causal intervention is introduced to move beyond correlational readouts.

- Target Definition: Finally, this task predicts polling distributions (as in OpinionQA), which should not be conflated with underlying human opinions. The paper should be explicit about this limitation and avoid over-claiming.

### Questions
See above.

### Soundness
3

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
3

### Summary
This paper investigates what large language models (LLMs) internally know about human opinions and how such knowledge is represented in their layers. Using two large-scale U.S. opinion datasets, OpinionQA and SubPOP, the authors probe the residual streams of models such as Llama-3.1-8B, Mistral-7B, and Vicuna-7B to assess alignment with real human opinion distributions across 22 demographic groups. They find that LLMs encode substantially more knowledge about opinions than is evident from their next-token outputs, with probing achieving over 50% lower KL divergence compared to direct prompting. The study contributes to understanding latent representations of social knowledge in LLMs and suggests computationally efficient methods for probing or fine-tuning models for opinion prediction and value alignment.

### Strengths
1. Novel methodology and insight
The paper presents a compelling approach to measuring LLMs’ internal representations of opinions rather than relying on surface-level outputs. This perspective offers new insights into how LLMs encode multidimensional social information.

2. Strong empirical and interpretability contribution
By combining probing with sparse autoencoders, the study identifies where and how demographic and opinion-related features emerge in model layers. This combination of layer-wise probing and interpretability analysis is novel and methodologically sound.

3. Computational efficiency and practical implications
Demonstrating that probing achieves comparable gains to fine-tuning at a fraction of the cost provides a practical contribution relevant to alignment, social modeling, and interpretability.

### Weaknesses
1. Limited generalizability
The datasets used (OpinionQA and SubPOP) are limited to U.S. opinions. The authors acknowledge this, but cross-cultural validation would be critical to claim broader generalization of “LLMs’ knowledge of opinions.”

2. Unclear link between internal knowledge and actual reasoning
While the paper shows that LLMs encode information internally, it is less clear whether this knowledge can be effectively surfaced during generation. The distinction between “knowing” and “using” opinions could be discussed more deeply.

3. Interpretability conclusions may overreach
Although the use of sparse autoencoders reveals correlations between attention heads and demographic groups, the causal interpretation of these findings remains speculative. Prior work has shown that attention heads often exhibit polysemantic behavior and that apparent interpretability can arise from correlations rather than direct causal encoding (Kissane et al., 2024; O’Neill & Bui, 2024). Similarly, ablation studies such as Baan et al. (2019) demonstrate that removing seemingly interpretable heads often has limited impact on model performance, suggesting that correlation of head activity with a concept does not necessarily imply that the head causally encodes it. The paper would benefit from acknowledging these interpretability limitations and framing its conclusions accordingly.

### Questions
1. Could the authors elaborate on how “internal knowledge” might be made actionable? For example, can probing insights be used to steer or align model outputs at inference time?

2. How robust are the findings to prompt variations or to LLMs trained on non-English or cross-cultural corpora?

3. How does the framework compare to retrieval-augmented or social simulation approaches that integrate external evidence (e.g., Group Preference Optimization: Few-Shot Alignment of Large Language Models)?

### Soundness
2

### Presentation
3

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
This paper investigates opinion-related knowledge encoded within large language models (LLMs). To identify latent knowledge in LLMs, the authors employ probing and sparse autoencoder techniques. Specifically, they extract predicted probabilities from a trained probe given opinion-eliciting prompts and compare these against next-token probabilities, measuring divergence from ground-truth human opinion distributions. The experimental evaluation utilizes two opinion datasets containing demographic information. Results demonstrate that probing-based predictions align more closely with ground-truth distributions than standard LLM outputs. Through comprehensive layer-wise and option-wise analyses, the authors reveal that middle layers encode the richest opinion-related knowledge. The authors conclude that LLMs encode substantially more internal knowledge about opinions than their generated outputs suggest.

### Strengths
The primary strength of this work lies in its revealing findings about hidden knowledge in LLMs for opinion generation tasks. The results indicate that LLMs possess greater capacity to generate demographically appropriate opinions than their outputs reflect, suggesting interesting limitations in the generation process. The methodology, while conceptually straightforward, is sound and reproducible, providing a template that other researchers can adapt for similar investigations.

### Weaknesses
The paper would benefit from deeper analysis in several areas. The observation in line 321 “This suggests that the drop from probe to prompting arises at the unembedding stage” is particularly intriguing but underexplored. While the authors provide additional investigation, the presentation consists primarily of reported numbers without sufficient interpretation. Visualizing these results in a format similar to Figure 4 would help highlight the pattern where final layers underperform middle layers. Without such emphasis and thorough analysis, it remains unclear how this work's findings differ substantively from prior research examining layer-specific roles in other tasks, such as multilingualism [1]. 

[1] Zhao, Yiran, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, and Lidong Bing. "How do large language models handle multilingualism?" Advances in Neural Information Processing Systems 37 (2024): 15296-15319.

### Questions
The nature of the captured knowledge requires clarification. The authors frame their research question as "what do LLMs know about human opinions?" and examine differences across demographic groups. However, it is ambiguous whether the results reveal knowledge specifically about demographics, opinions, or their interaction. To strengthen the claims, the authors should conduct control experiments with non-opinion tasks (e.g., cultural knowledge or mathematical reasoning) using the same methodology. Do similar layer-wise patterns emerge? Are the demographic effects opinion-specific or more general? Such investigations are essential to establish whether the findings are unique to opinion modeling or reflect broader properties of LLM internal representations.

### Soundness
3

### Presentation
2

### Contribution
2
