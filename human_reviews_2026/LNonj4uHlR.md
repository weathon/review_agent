# Gen-Review: A Dataset and Large-scale Study of AI-Generated and Human-Authored Peer Reviews

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 4

## Abstract
How does the increased adoption of Large Language Models (LLMs) impact the scientific peer review? This multifaceted question is fundamental to the integrity and outcomes of the scientific process. Timely evidence suggests LLMs may have already been used for peer-review, e.g., at the 2024 International Conference of Learning Representations (ICLR), and the LLMs' integration in peer-review was confirmed by various editorial boards (including that of ICLR'25). To seek answers, a comprehensive dataset is needed, but lacking until now. We therefore present Gen-Review the largest dataset of LLM-written reviews so far. Our dataset includes 81K reviews generated for all submissions to the 2018--2025 editions of the ICLR and by providing the LLM with three independent prompts: a negative, a positive, and a neutral one. Gen-Review also links to the papers and the conference reviews thereby enabling a broad range of investigations. We make a start and use Gen-Review to scrutinize: if LLMs exhibit bias in reviewing (they do); if LLM-written reviews can be automatically detected (so far, they can); if LLMs can rigorously follow reviewing instructions (not always) and whether LLM-provided ratings align with a papers' final outcome (happens only for accepted papers).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Collected 81K AI-generated reviews.

### Strengths
A good effort of collecting huge dataset for interesting analyses.

### Weaknesses
Why did authors choose three options of aspects (positive, negative, neutral)? What if reviewers have more fuzzy, indecisive opinions?
I strongly suggest to change the sub-title of the dataset "Large-Scale Dataset of Peer Reviews" with a clear note of saying "this is AI-generated" like "Large-Scale Dataset of AI-generated Reviews,"
As noted in RQ1, the generated reviews seem far different from human reviews, and I wonder the prompts and setup authors used are the best possible reviews where AI can achieve. Have you tried other alternative framework, like agentic reviewing or self-refinement, or simply more advacned prompts?

Overall, despite some value of the collected dataset and findings, I am not sure the specific use cases of the data to investigate further findings, or quality of AI data given the limited setup of framework.

### Questions
See my comments above

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
4

### Summary
The manuscript introduces Gen-Review, a large corpus of LLM generated peer review text. It uses ICLR submissions from 2018 to 2025 (32,652 papers) and 124,615 human written review comments from OpenReview. For each paper, the authors aimed to produce up to 3 LLM generated review using a ChatPDF and prompts crafted to yield neutral, positive and negative assessments. Due to PDF size limits and API failures, the final release comprises 81,850 LLM generated reviews, hence some papers received fewer than 3 reviews. The LLM generated review dataset is further used for 4 studies: rating / bias analysis across the 3 prompt conditions (concludes a skew toward positive scores even for neutral prompt; alignment between LLM assigned scores and official ICLR decisions (limited alignment - particularly for rejections); instruction following analysis; and detecting LLM generated reviews by using Binoculars detector.

### Strengths
1. Dataset is larger than prior work on LLM generated reviews - 81,850 LLM reviews for ICLR submissions from 2018 to 2025. Dataset scale and linkage can be a playground for further studying LLM behavior in academic peer review.
2. Simple prompting scheme - can help with regeneration & extension to other models. Three prompt scheme (positive / neutral / negative) allows for comparisons of model bias and instruction following in different intent signals.

### Weaknesses
1. Scope is narrow - dataset as well as the analyses is from a single venue (ICLR), and a single black-box LLM service (ChatPDF). Hence, the generality of the empirical findings is limited. It is unclear if the same bias patterns, instruction following behavior or detector performance would hold for other venues, for different reviewing guidelines, for other LLMs, and for other academic research areas.
2. ChatPDF is a blackbox - so full reproducibility is difficult. Exact model identifier, parameters, internal system prompts, etc. is unknown.
3. Data/exposure confounds: Analysis is on publicly available ICLR papers from 2018 onwards. The manuscript does not check whether these papers or the reviews were already seen during training of the LLM. As a result, part of the positive skew noted in the manuscript may come from training time exposure to the same or closely related content, not from the prompt design alone. Also, using the decision by ICLR as the only ground truth ignores the fact that some of the rejected papers are later accepted at other comparable venues, hence, the disagreement between LLM and ICLR official decision may not necessarily indicate LLM error.

### Questions
1. Some ICLR rejected papers are later accepted at other top venues. Did you consider controlling for later acceptance when interpreting misalignment between LLM scores and ICLR decisions? If not, could you comment on how often this might explain disagreements?
2. Did you perform any check for possible LLM training time exposure to the papers and reviews?
3. You currently evaluate only one detector (Binoculars). Can you please add at least one additional detector and report per year false positive / false negative rates?

### Soundness
2

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
This paper presents Gen-Review, a large-scale dataset of AI-generated peer reviews, related to recent interese in investigate how LLMs are affecting scientific peer reviewing. The authors created the dataset by using ChatPDF (which relies on GPT-4o models) with three different prompting strategies for each paper. They try to simulate an "honest-but-lazy" reviewer who might turn to LLMs to fulfill reviewing duties under time pressure. The paper explores four key research questions through analysis of the dataset, including substantial positive bias in LLM-generated reviews, the detectability of LLM-generated reviews, etc.

### Strengths
1. The paper is clearly written and well-organized. The authors explicitly state their setting and how they arrived at this design choice.

2. Gen-Review is a large-scale dataset of LLM-generated peer reviews, which can be useful to the community.

3. The paper addresses four important research questions about bias, alignment with outcomes, instruction-following, and detectability, which could significantly inform policy decisions and best practices for scientific peer review in the age of LLMs.

### Weaknesses
1. Although the authors clearly state their modeling assumption of an "honest-but-lazy" reviewer in **DESIGN CHOICES** section, I think this important limitation deserves more thorough discussion in the paper's limitation section.

2. While authors have mentioned in the limitation section that the reviews in Gen-Review relied on OpenAI GPT-4o models solely, it would still be helpful to test the robustness of findings across models, at least on the paper subset.

3. In **Alignment of neutral-prompted reviews with human-driven paper’s outcome** Paragraph, while results show that the LLM’s recommendation does not seem to align with the paper’s final decision, it is important to note that disagreement also arises among human reviewers themselves and between reviewers and ACs.

### Questions
See Weakness.

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
5

### Summary
This work curates and generates a dataset of human and LLM reviews. The contribution is a dataset of 81k LLM reviews (and their related human reviews) that is used to study: (i) bias in LLM reviews; (ii) alignment between LLM reviews and human review outcomes; (iii) LLM reviewers instruction following; and (iv) AI generated review detection.

### Strengths
The work raises four research questions relevant to LLM reviews:

1. Bias in LLM-written reviews: Comparing score distributions of AI reviewers that are neutral, positive, and negative.

2. Alignment between neutral reviews and human review outcomes.

3. Evaluation of LLM review instruction following.

4. Detection of LLM reviews.

### Weaknesses
1. LLMs do a better job at reviewing when they review not only the paper but the entire submission, ie, the data, code, and paper. In practice, this can be done by zipping the data, code, and paper (in a submission.zip) and asking an LLM such as GPT-5 Pro to review the entire submission.

2. LLM reviews can be rephrased using tools (such as Grammarly) which avoid their detection.

3. This work has a narrow perspective on LLM usage: "they can facilitate routine scientific tasks, allowing researchers to focus on the scientific discovery," and is not well-informed. LLMs are also valuable for scientific discovery.
Automated scientific discovery (ASD) is an emerging field; see, for example, the 1st Open Conference on Agents for Science (Agents4Science 2025).

4. This work and benchmark are not novel.
Previous work on AI reviewing curated, large-scale datasets of human and LLM reviews.

5. The work finds that LLMs agree with human reviewers on acceptance, however under those settings they favor acceptance over human reviews.

### Questions
"We used LLMs to, of course, generate our proposed Gen-Review dataset. The way we used them
is described in Section 3
Aside from this task, we did not use LLMs at all (nor for source-code development, or as a writing
assistant)."

Were LLMs used to write code for generating the figures? or for generating the figures directly from data?

### Soundness
3

### Presentation
2

### Contribution
1
