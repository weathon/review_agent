# Culturally Grounded Real-World Evaluation of Korean Vision–Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
VLMs perform well on standard benchmarks, yet their performance on authentic, culturally grounded tasks remains underexplored. 
We introduce HAERAE-Vision, a Korean real-world benchmark built from 86,052 question–image pairs across nine online platforms. 
Through a six-stage pipeline that applies appropriateness filtering, difficulty calibration, image dependency verification, checklist-based decomposition, and multi-phase human validation, we curate 653 rigorously validated items across 13 domains (0.76% survival). 
Each item is paired with a structured checklist rubric, enabling fine-grained evaluation beyond single-point correctness. We evaluate 39 VLMs spanning proprietary, open-weight, and Korean-specialized families under a unified protocol, and scoring with LLM judges demonstrates high reliability (Krippendorff’s α = 0.867). Even the strongest systems (Gemini 2.5 Pro, GPT-5) remain below 50% accuracy, with errors concentrated in explicitness and procedural reasoning, while Korean-specialized models show no clear advantage over multilingual counterparts. These findings highlight persistent gaps in real-world multimodal reasoning. Our work further offers a reproducible methodology for constructing robust, culturally grounded benchmarks across languages.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents HAERAE-Vision, a Korean real-world benchmark designed to evaluate vision–language models (VLMs) on culturally grounded tasks. It compiles 653 validated question–image pairs from an initial pool of 86,052 samples using a six-stage data curation pipeline and structured checklist-based evaluation. The authors assess 39 VLMs and find that even top-performing models achieve below 50% accuracy, indicating persistent challenges in multimodal reasoning and cultural understanding.
I find the benchmark’s motivation valuable, but the paper lacks methodological transparency in several pipeline stages and offers limited evidence connecting its cultural grounding claim to the presented data.

### Strengths
1. Addresses an underexplored gap by focusing on culturally grounded, real-world VLM evaluation in the Korean context.
2. Introduces a systematic multi-stage data curation pipeline that aims to ensure quality and difficulty calibration.
3. Did some interesting improvements from existing works, like structured checklist-based rubrics, allowing for more granular analysis of model reasoning beyond simple accuracy scores.

### Weaknesses
1. The core argument of the paper is cultural grounding (L90: Tasks require Korean-specific knowledge). But, in Figure 1, in the first (Natural Objects (Animals/Plants/Insects)), fourth (Science), and sixth (Coding) questions, I don't see any aspect of 'cultural' here. Needs clarification on how and why. It is not possible for me to check the whole dataset, but I expect more details on this.
2. Following the concerns of cultural grounding, 653 sample size of 653 is really low.
3. No mention of ethical aspects of data collection, biases and mitigation.
4. In Stage 2: Appropriateness Assessment of Data Construction Pipeline, if any post has PII data or sensitive info, I'm not sure if it is OK to use closed-source models to judge that, as it may leak sensitive data; closed-source model providers are known to use those data for training and more. Thirdly, there is no evaluation on if this approach is correct and performs on par with human annotators. Typically, human-AI annotation agreement or such is measured to show that LLM is capable enough for this task, but unfortunately, it is not present in this paper.
5. In Stage 3: Difficulty Calibration, the authors removed easy questions by simply filtering out the correct ones by GPT-4o, Gemini-1.5-Flash, and Claude-3.5. If that's so, the statement in lines 51-53 (*earlier Korean benchmarks report notably higher scores with the older generation model*) doesn't make sense, as you already removed the ones they got correct. Also, is it OK to do so? Is there any prior work that did that?
6. In Stage 4: Image Dependency Verification, again, details are missing. No human annotator agreement study was done.
7. In Stage 5: Checklist Generation, there is no human study to see if the LLM is as capable of generating relevant checklists as a human.
8. In Stage 6: Human Validation, again, a lot of details are missing. Need information on annotator recruitment, training, and process details. As per Figure 2, after Image Dependency, it had 1,040 samples. Considering the final sample size of 653, 31.4% are in Line 150/151, which doesn't add up.
9. Line 261/262 say, *We evaluated 39 vision-language model.*, but Table 1 has only 18.
10. Line 350/351 say, *We analyzed 59k checklist items across six rule types*. How? No details are available. More detailed error analysis with examples is expected. Currently, the level of detail is very low, and no proof is presented.
11. No error analysis connecting the VQAs to the wrong answer, which is expected in benchmarks. Is there any particular pattern? For example, if the image has X, then the model usually gets it wrong by assuming Y and more. 
11. No detailed analysis on performance deviation and variation across 13 categories by model and category combination. It is needed to understand more complex issues and underlying problems.

### Questions
1. What specific criteria were used to determine whether a question is “culturally grounded”? Add more details and some statistics from the dataset too on this possible error/confusing issue. Please provide representative examples from each category that demonstrate Korean-specific reasoning. Also, add examples of potential ambiguous issues, with reasoning/justification of why it is correct.
2. How was this size determined to be sufficient for robust benchmarking? Did you perform any reliability or statistical representativeness analysis (e.g., variance or confidence intervals across categories)? How can you confirm that this dataset represents Korean culture well?
3. What was the source of the images and text data, and how was consent handled?  How did the authors ensure no PII or sensitive information was included? Were any bias audits (e.g., gender, regional, socio-economic) performed? Check ethics issues and clarify those too.
4. In Stage 2: Appropriateness Assessment, could you provide this prompt structure and rationale? How was model leakage of sensitive content prevented, given the use of closed-source LLMs? Was there any human verification or inter-rater agreement analysis comparing GPT-4o assessments with human annotators? Without such validation, how do the authors justify the reliability of automated appropriateness checks?
5. In Stage 3: Difficulty Calibration, how will you justify the issue of removing these items, mentioned in weakness, or are there precedents in the literature for difficulty calibration by exclusion of correct responses? How does this decision align with the claim that older-generation models perform better on Korean benchmarks (Lines 51–53)? Could this filtering introduce selection bias toward items that current models systematically fail?
6. In Stage 4: Image Dependency Verification, what exact criteria were used to determine image dependency? How is it determined? Were human raters involved, and if so, what was the inter-annotator agreement? If this stage relied solely on LLM judgment, how was reliability validated?
7. In Stage 5: Checklist Generation, could the authors include example prompts and outputs for clarity? Was any human comparison done to verify that the LLM-generated checklists align with human expectations and reasoning quality? Without such validation, how do the authors ensure that checklist-based evaluation is trustworthy?
8. In Stage 6: Human Validation, how were annotators recruited and trained? Please clarify the apparent mismatch between the number of samples after Stage 4 (1,040) and the final dataset (653). What explains "31.4%" in Lines 150–151? What exactly does it represent?
9. In evaluation details (Lines 261–262, Table 1), were additional models excluded from reporting? If yes, why? Please include category-level (13 categories)) results for all models.
10. The paper reports “*59k checklist items across six rule types*,” but no methodology is given. How were these checklists analyzed and categorized? What tools or coding schemes were used? Please include representative examples of common checklist errors (dataset development phase). 
11. Did the authors identify recurring failure patterns (e.g., object recognition errors, language misunderstanding, culturally grounded terms)? Are there any examples showing how visual or cultural factors contributed to model failure? If yes, please categorize and add details on this. Such insights are crucial to validate the benchmark’s diagnostic value. 
12. No detailed analysis is provided for model performance across the 13 categories. Could the authors include a per-model, per-category breakdown? Perform a detailed analysis. Such analysis could clarify whether the benchmark’s difficulty stems from cultural knowledge, reasoning type, or visual complexity.
13. Could the authors provide full prompts, annotation instructions, and some changelog examples?


**Minor errors and suggestions:**
- L077:  four of the 13 domains >  *six* of the 13 domains

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HAERAE-Vision, a culturally grounded benchmark for evaluating Korean vision-language models (VLMs) on authentic real-world tasks. The benchmark is constructed from 86,052 raw question–image pairs sourced from nine Korean online platforms, refined through a rigorous six-stage pipeline: data collection, appropriateness assessment (filtering for safety, objectivity, and temporal stability), difficulty calibration (removing trivially easy items), image dependency verification (ensuring visual reasoning is required), checklist generation (creating structured evaluation criteria), and multi-phase human validation. This process yields 653 high-quality items across 13 domains (0.76% survival rate), each paired with a structured checklist for fine-grained evaluation beyond binary correctness.

The authors evaluate 39 VLMs spanning proprietary (e.g., Gemini 2.5 Pro, GPT-5), open-weight (e.g., Skywork-R1V3-38B), and Korean-specialized (e.g., VARCO-VISION 2.0) families using LLM judges (GPT-5-mini as primary), with high inter-judge reliability (Krippendorff’s \(\alpha=0.867\)). Key findings include: (1) top-performing models (Gemini 2.5 Pro, GPT-5) achieve only ~48% accuracy, (2) errors concentrate in explicitness (92.1% unmet) and procedural reasoning, (3) Korean-specialized models show no clear advantage over multilingual counterparts, and (4) search-augmented inference provides inconsistent performance gains. Beyond the benchmark itself, the work contributes a reproducible methodology for building culturally grounded multimodal benchmarks across languages.

### Strengths
- Originality: The paper advances beyond prior Korean VLM benchmarks (e.g., K-VISCUIT, KRETA) by focusing on authentic, culturally embedded communication (colloquialisms, pragmatic cues, messy user queries) rather than shallow factoid tasks. The six-stage pipeline is a novel combination of automated filtering and human validation, addressing a critical gap in constructing benchmarks that reflect real-world user needs.
- Significance: This work addresses a critical limitation of current VLM evaluation—overreliance on standardized, English-centric benchmarks that fail to capture cultural context and real-world query complexity. The reproducible pipeline provides a blueprint for building culturally grounded benchmarks in other languages, fostering more inclusive and realistic VLM evaluation. The finding that top models struggle with explicitness and procedural reasoning also identifies actionable areas for VLM improvement.

### Weaknesses
- Human-LM Judge Alignment Details: While the paper mentions human annotators rated LLM judge scores (mean appropriateness = 4.13/5), it lacks key metrics for quantitative alignment between LLM judges and human experts. For example, there is no report of Pearson correlation between LLM scores and human scores for the same responses, which is critical to validating the LLM judge’s reliability as a substitute for human evaluation.
- Error Analysis Granularity for SOTA Models: The error analysis in Section 4.4 aggregates results across all 39 models, diluting insights into the specific failures of state-of-the-art (SOTA) systems like Gemini 2.5 Pro and GPT-5. Focusing error analysis on these top models would reveal more actionable patterns—e.g., whether their procedural reasoning gaps are domain-specific (e.g., gaming vs. science) or universal.
- Diagnostic Breakdown of Model Failures: The paper does not disentangle the root causes of model errors. For instance, it is unclear whether a model’s failure on a Korean cultural item stems from (1) poor Korean language proficiency, (2) lack of Korean cultural knowledge, or (3) insufficient general multimodal reasoning. Such a breakdown would help identify whether gaps are cultural-specific or broader, guiding targeted model improvements, and that will help us find whether HAERAE-VISION is a general challenging benchmark in Korean language, or a challenging benchmark with strong Korean characteristics (Korean culture, knowledge, etc. )

### Questions
1. Could you provide quantitative alignment metrics (e.g., Cohen’s \(\kappa\), Pearson/Spearman correlation) between GPT-5-mini’s scores and human expert scores for the 250-sample validation dataset? This would more rigorously validate the LLM judge’s ability to approximate human evaluation.
2. Can you extend the error analysis in Section 4.4 to focus exclusively on SOTA models (Gemini 2.5 Pro, GPT-5)? For example, what percentage of their failures in each domain (e.g., Entertainment vs. Science) are due to explicitness vs. procedural reasoning gaps? This would yield more targeted insights for model developers.
3. Have you conducted a root-cause analysis of model failures to distinguish between (1) Korean language proficiency gaps, (2) Korean cultural knowledge gaps, and (3) general multimodal reasoning gaps? If not, could you add a small-scale analysis (e.g., for 50 representative items) to clarify which factors drive performance on HAERAE-Vision?
4. Regarding the typo in Line 253: Will you correct the example to reflect the correct score calculation (e.g., “4.0/5 when two items are partially satisfied and three are fully met”)? Additionally, are there other minor inconsistencies in the paper that need revision?

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
5

### Summary
In this work, the authors present a new vision-language benchmark called HAERAE VISION, a Korean real-world cultural benchmark that consists of 653 validated questions across 13 domains. They construct this benchmark by filtering 86K question-image pairs using a six-stage process comprising appropriateness filtering, difficulty calibration, image dependency verification, checklist-based decomposition, and multi-phase human review. Finally, they comprehensively evaluate a mix of open and closed API models with LLM judges.

### Strengths
- The authors introduced a new benchmark that I have not encountered in my past experience with multimodal evaluation. It is refreshing to see a new evaluation that covers important topics such as cultures and life from countries that are not typically represented in existing evaluations.
- The authors evaluated 39 VLMs, including both open and closed models and some Korean-specialized models, making this a comprehensive benchmark that allows us to understand how these models perform and the landscape of Korean culture and life.
- The multi-stage filtering is comprehensive and heavily filters out questions by going from ~86K image-QA pairs down to a mere 653 examples.

### Weaknesses
- The paper shows that frontier VLMs still struggle with Korean culture question-answering, yet Stage 2 uses GPT-4o to filter subjective/unverifiable items and Stage 4 uses Gemini 2.0 Flash to decide image dependency. Both are arguably weak and older models in general. If these models are weak in this domain (especially something important as assessing and understanding another culture), they are likely weak filters too. That can remove valid questions (or keep the wrong ones), and I don't see a check against this. For a narrow cultural benchmark, I would expect native Korean speakers to be involved earlier in the data pipeline (not only at Stage 6) to spot filter errors and to rate how important or relevant each question is to Korean life. This would reduce over-filtering and help build a benchmark that covers the Korean culture holistically.
- I am concerned about the dataset's size. After rigorous filtering, we are left with only 653 examples across four categories and 13 subcategories. The number of questions per subcategory seems to be too small for certain subcategories to get a meaningful per-category performance breakdown. For example, the "Health/Medical" subcategory has only 21 questions, even though health and medical matters are an essential aspect of Korean life and culture.
- In section 3.2, it is not clear to me why a temperature of 0.6 was chosen for all VLMs, and why 1.0 was used for the evaluation judge. I would also like more details on the prompting strategy and on how the authors arrived at the optimal prompts for a fair assessment of these VLMs.

### Questions
- Why was a checklist-based auto-judge selected over other types of judges?
- According to section 5.3, "...the full 13-category test set is hosted on a rate-limited, anonymous evaluation server to prevent overfitting and support fair model comparison". It is not clear to me what the server is and how it is being used. Is it open to the public when the dataset is released? I am not sure how it is being rate-limited.
- In Stage 3 of the data filtering pipeline, please clarify "ground-truth answer bundle". Do you mean a set of acceptable gold answers/aliases used to score overlap (0–1), or were models actually given these answers in the prompt? If the latter, that leaks labels and invalidates the difficulty screening. Please specify how the bundle is constructed.

### Soundness
2

### Presentation
4

### Contribution
2
