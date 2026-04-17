# SEA-SafeguardBench: Evaluating AI Safety in SEA Languages and Cultures

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Safeguard models help large language models (LLMs) detect and block harmful content, but most evaluations remain English-centric and overlook linguistic and cultural diversity. Existing multilingual safety benchmarks often rely on machine-translated English data, which fails to capture nuances in low-resource languages. Southeast Asian (SEA) languages are particularly underrepresented despite the region’s linguistic diversity and unique safety concerns, from culturally sensitive political speech to region-specific misinformation. Addressing these gaps requires benchmarks that are natively authored to reflect local norms and harm scenarios. We introduce SEA-SafeguardBench, the first human-verified safety benchmark for SEA, covering eight languages, 21,640 samples, across three subsets: general, in-the-wild, and content generation. The experimental results from our benchmark demonstrate that even state-of-the-art LLMs and guardrails are challenged by SEA cultural and harm scenarios and underperform when compared to English texts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SEA-SafeguardBench, a multilingual, culturally grounded safety benchmark for Southeast Asia (SEA). It comprises three subsets—General (translated/adapted from English safety data), In-the-Wild (ITW) (native prompts reflecting locally salient topics), and Content Generation (CG) (prompts meant to elicit culturally unsafe outputs). The study evaluates a range of safeguard models and LLMs using AUPRC and threshold analyses, reporting notable performance drops in SEA languages and on culturally nuanced cases.

### Strengths
1. Shifts safety evaluation toward under-served SEA languages/cultures, beyond English.
2. Blends translated general safety data with native ITW and CG subsets to probe culture-specific failures.
3. Compares various guardrails/LLMs with clear metrics (AUPRC, threshold sensitivity) and qualitative failure analyses.
4. Results highlight major cross-lingual and cultural performance gaps, reinforcing the need for region-aware safety alignment.

### Weaknesses
1. Limited novelty beyond prior work (SafeWorld).
- The benchmark design—particularly the ITW and CG subsets—closely mirrors SafeWorld: Geo-Diverse Safety Alignment (NeurIPS 2024), which already proposed:
  - human-verified prompts grounded in local cultural/legal norms,
  - evaluation of cultural safety across multiple regions, and
  - analyses of culturally conditioned “unsafe” behaviors.
- SEA-SafeguardBench mainly adapts this paradigm to SEA locales. The paper should explicitly position itself relative to SafeWorld and clarify concrete methodological differences (e.g., data collection scale, validation pipeline, or regional focus) rather than presenting it as a wholly novel contribution.
- Related omissions: The paper should also reference CARE (Cultural Awareness Alignment, 2025) and CROSS (Multimodal Cultural Safety, 2025), which similarly extend beyond English-centric evaluation.

2. Ethical and IRB concerns.
- Annotators are explicitly exposed to harmful or taboo content during data creation and translation. Yet, the paper lacks an IRB or equivalent ethical review statement, risk disclosure, or worker-protection measures (e.g., content warnings, opt-out options, debrief).

3. Lack of formal definitions.
- The main text never clearly defines key terms such as safety, harmful, or sensitive. The authors should formally specify their operational definition of “safety”—and how it maps onto the hazard taxonomy used by prior safety benchmarks.

4. Unclear provenance of “things-not-to-do.”
- Section 2.4 states that 120 “things-not-to-do” per country were collected but omits how these were sourced (expert panels, web scraping, crowd annotations, or LLM-based drafts) and validated.
- The paper should detail:
  - data sources and authority,
  - filtering and validation criteria,
  - annotator instruction and adjudication protocols, and
  - handling of contested norms or ambiguous cases.
- Without this, the construct validity of the CG subset remains weak.

5. Labeling and disagreement handling ambiguities.
- How are disagreements between English and SEA-language labels resolved (kept separate, canonicalized, or marked “sensitive”)?
- What is the fallback for no-majority cases—automatic “sensitive” or re-adjudication?

6. Prompt template and quality-control opacity.
- Section 2.4 introduces three CG templates  but provides no concrete examples. Readers cannot assess whether these prompts truly capture cultural nuances.
- The authors should include representative examples from at least two countries and describe the verification process when annotators were uncertain about correctness or cultural appropriateness.

7. Evaluation coverage.
- The paper excludes frontier API models such as GPT-4o from the main results table. Including these baselines would clarify how general-purpose models compare with dedicated safeguards.

8. Missing ablation on culturally aware prompting.
- Figures 10–11 hint at cultural sensitivity evaluation, but the paper does not report whether zero-shot prompts explicitly instructing models to consider the target country’s cultural norms were tested.
- Such an ablation could show whether “culturally conditioned” instructions improve accuracy, calibration, or recall on sensitive cases—an important finding for future safety alignment.

### Questions
See above.

### Soundness
2

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
4

### Summary
The paper presents SEA-SafeguardBench, a human-verified safety benchmark spanning eight languages (seven SEA) and 21,640 samples across general, in-the-wild (ITW), and content-generation (CG) subsets, and shows that both guard models and LLMs that look strong in English degrade on SEA languages and culture-specific cases. The samples are generated by native prompts from templates + model responses, with “Safe/Sensitive/Harmful” labels. "Sensitive" labels represent ambiguity where no clear majority is reached by labelers. It further diagnoses threshold sensitivity, prompt-context effects, and reports LLM safety/response trade-offs in SEA settings.

### Strengths
1. The authors assemble culturally grounded SEA data across 7 SEA languages + English, with native SEA speakers writing/validating content and details on annotator hiring, pay, QA, which is important for ethics and authenticity. This fills a gap left by English-centric or translation-only datasets, which were done by prior work. The authors also show that current models cannot cover SEA-language effectively by validating performance degradation on guard models and language models. 
2. The paper covers a rich analysis: beyond guard model and language model performance, it contains comparison with other benchmark datasets, impact of prompt in response classification showing shortcut reasoning in models, threshold sensitivity for F1 scores, and model behavior in sensitive labels.

### Weaknesses
1. Sensitive response are assigned label "safe" in prompts but "unsafe" in responses, which seems like an arbitrary choice, especially given that the sensitive samples represent ambiguous cases where no clear majority has been reached. The experiment setting would have been more persuasive if you have excluded sensitive samples, or treated them as separate classes. 
2. There exists severe class imbalance in the Content-Generation subset (view A.6). Generating harmful responses with jailbreaking LLMs, such as the data augmentation technique in HarmAug[1] would be helpful.
3. The paper uses template-based LLM generation for constructing the dataset, but lacks approaches like detecting and removing overlaps in the generated samples. You analyze word overlap, but do not describe near-duplicate detection across generations/translations. Please include MinHash or embedding-based deduplication (and publish duplicate rates), a standard step for modern corpora.[2]

[1] Lee, Seanie, Haebin Seong, Dong Bok Lee, Minki Kang, Xiaoyin Chen, Dominik Wagner, Yoshua Bengio, Juho Lee, and Sung Ju Hwang. "Harmaug: Effective data augmentation for knowledge distillation of safety guard models." ICLR 2025

[2] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. "Deduplicating Training Data Makes Language Models Better." ACL 2022

### Questions
1. Though the whole point of this paper is about multilingual safetyness, the t-SNE visualization is made with only english samples without further justification, even though it uses a multilingual embedding model. Is there any particular reason for this decision?

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
SEA‑SafeguardBench introduces a human‑verified benchmark spanning seven Southeast‑Asian languages with over 21,000 samples that evaluates how well large language models and guardrails avoid harmful or culturally inappropriate outputs in general, in‑the‑wild and content‑generation scenarios, revealing significant performance gaps compared with English evaluations.

### Strengths
1. The benchmark is the first to focus on safety evaluation in Southeast‑Asian languages, filling a void in existing multilingual benchmarks that rely on machine‑translated English data.
2. The dataset includes 21,640 samples across three carefully defined subsets (General, In-the-Wild, and Content Generation), each serving a different safety dimension, with strong annotation protocols (multiple annotators, majority voting, sensitive-label handling).
3. The experiments on 20 safeguard models reveal consistent performance degradation in SEA languages, demonstrating both language-specific and culturally driven failure modes, which provide valuable diagnostic evidence for future multilingual safety alignment.

### Weaknesses
1. Although human-verified, both the General and CG cultural subsets originate from Google NMT translations, which may imprint English framing and safety priors and weakens the paper's "native authoring" claim.
2. The paper reveals gaps but stops short of giving clear design principles for culturally-aligned safeguard modeling (beyond needing more data).
3. No systematic taxonomy of “cultural topics” is presented. The notion of “culture” as used in the benchmark stays conceptually vague.
4. The benchmark does not explicitly disentangle language-level effects from culture-specific phenomena, especially in the cultural subset, where cultural and linguistic dimensions co-vary.
5. In the “Diversity of Our Datasets” section, the t-SNE-referenced figure should be Figure 3 instead of Figure 2.

### Questions
NA

### Soundness
3

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
4

### Summary
The paper introduces SEA-SAFEGUARDBENCH, a LLM safety benchmark for 7 south east asian languages. The benchmark consists of a general subset that is derived through manual curation of auto-translated English benchmarks. In addition, the authors add two cultural subsets, with the in-the-wild one being prompts curated by native speakers and a culturally sensitive prompt set with respective GPT4o responses.

### Strengths
- fills an important gap in LLM safety evaluation beyond English
- well constructed benchmark with high-quality human annotations 
- focus on the difference between generic (Western-centric) and culturally sensitive evaluation 
- meaningful annotation setup where safety vs sensitive content is measured through annotator agreement
- inclusion of input and output safety 
- The analysis conducted beyond the benchmark contribution reveals some interesting insights into the differences between sensitive SEA and English topics. 
- Safeguard evaluation also provided interesting insights on model performance and multilingual trade-offs

### Weaknesses
# Major Weaknesses 

**Word Overlap Analysis**
- the usage of tokens instead of lemmata here makes little sense, since tokens are subwords as well as case and whitepace sensitive 
- additionally there are duplicate tokens for conjugations/declinations with significant semantic overlap
- a more standard NLP analysis that eliminates stop-words and considering lemmata would be more useful 
- in addition the existence of new words alone is a decent indicator of a signficant change but a more detailed analysis on where the cultural difference actual lie is missing from the paper 

**Metrics**
The setup to compute confidence scores seems highly flawed. 
For one as your own analysis shows LLMs generally yield overconfident token probabilities with little to no correlation of likelikhood or confidence. Conversely, a setting using ordinal categories is generally much more robust and crucially not comparable to token probabilities. 
I do not understand the need for two different evaluation setting wrt. to closed vs. open models. Using ordinal categories for all would result in a much more robust and fair comparison. 

this also translates to the analysis in 5.2 dedicated safeguard models are specifically trained to give "safe" and "unsafe" outputs which leads to much better calibration of these respective token probabilies compared to a general-purpose model like Gemma. While the findings in Fig. 5 are interesting the may be heavily biased by the used metric itself. 

**Evaluation**
- only evaluating outputs from a single model (GPT4o) is definitely a limiting factor


**Larger Context**
The paper is missing a more general discussion on the ethics and implications of cultur specific safety notions. 
Currently, the setup in the paper simply aims to reflect cultural conventions for specific countries and societies. While this is one valid approach to safety an important reflection is missing here. 
For example, many cultures or countries will deem things as 'unsafe' that might violate general human rights, inlcuding culturally enshrined views on racism, sexis, homophobia limiting religious freedom. Further, as mentioned by the authors themselves some countries might have de facto or even de jure restrictions on free speech and critisim of the ruling party. When developing and assessing LLMs there is a certain moral and ethical obligation to transparently handle related issues and decisions. 

**Datasheet*
The paper should provide a datasheet for the newly introduced dataset (https://arxiv.org/abs/1803.09010) to adhere with standardized documentation practices, especially in safety related fields. 

## Minor Comments
The paper is missing some citations for important related work. 
Global-MMLU [1] also tackles cultural biases in multilingual LLM evaluation. Other works [2,3] have focused  on developing safe LLMs with Vietnamese support. Other evaluation work for SEA languages has also considered cultural biases [4]

[1] https://arxiv.org/abs/2412.03304

[2] https://aclanthology.org/2025.coling-industry.56/

[3] https://huggingface.co/Viet-Mistral/Vistral-7B-Chat

[4] https://arxiv.org/abs/2403.02715

### Questions
- Q1) For the world overlap analysis, can you provide some examples of new words that appear in the ITW set? Especially, those that give an  insight into the actual differences here 

- Q2) Can you provide furthter insights into the failure cases of cultural content generation? Table 2 shows that almost all safeguards exhibit a significant perfromance drop on the CG Cultural set. But what actually are those culturally sensitive  topics that models fail to asses correctly. Are there common patterns/failure cases? 

- Q3) Can you elaborate on your conclusions for Fig. 4. You claim that removing prompt access increases a models misclassification rate. From the Figure itself this seems evident but to me there is a larger question here. Should we generally evaluate the safety/unsafety of outputs in isolation without considering the larger context of the user prompt? Unsafe behavior may only emerge in certain combinations of input and output and seeminlgy begning responses can be problematic in certain context only and vice versa.

### Soundness
3

### Presentation
2

### Contribution
3
