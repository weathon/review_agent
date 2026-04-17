# AI Respondents for Policy Monitoring: From Data Extraction to AI-Driven Survey Responses in the OECD STIP Compass

- Decision: Reject
- Scores: 2, 0, 4, 2

## Abstract
Science, Technology, and Innovation (STI) policies are central to national and international competitiveness, yet their complexity makes systematic mapping and continuous monitoring a persistent challenge. This study draws on one of the largest initiatives in the field, the OECD STIP Compass survey, which collects and organizes data on STI policy from OECD countries and has historically relied on extensive manual survey efforts to ensure global consistency. Large Language Models (LLMs) are redefining representation learning in NLP, enabling them to process and internalize knowledge from long unstructured documents. This paper presents a novel application of LLMs for structured information extraction and generation from STI policy documents, focusing on OECD data across six sample countries. We develop a data extraction pipeline based on long-context in-context learning to encode task-specific schemas that allow learning of survey taxonomy labels from public URLs referencing policy initiatives. The pipeline integrates validation steps using a secondary LLM for relevance and evidence scoring, and comparison with survey responses completed by human respondents. For evaluation, we apply multiple overlap measures, including overlap ratios, agreement scores between human-generated and LLM-generated policy indicators, and K-fold cross-validation for AI-generated labels. Our findings indicate that LLMs can achieve high overlap with human respondents for policy indicators (84-95%). Qualitative analysis reveals that the model tends to provide more detailed descriptions, complementing human-written content. Our approach points to the potential of an AI-assisted framework for STI policy monitoring, enhancing both efficiency and quality in international policy intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors prompt LLMs to summarize and classify science, technology, and innovation policy documents, then analyze the differences between the LLM responses and human responses.

### Strengths
* There is substantial need to assess the utility of LLMs for analysis of policy documents

### Weaknesses
* The introduction does not explain the task clearly. What do these policy documents look like? What does "mapping" and "monitoring" these policies look like? That is, what are the inputs and outputs of the day-to-day tasks people in this field are trying to automate with LLMs?
* No agreement between the evaluator-LLMs and humans was calculated. Using an LLM as an evaluator is only okay if we have a measure of how well the evaluator LLM agrees with human evaluation judgments.
* No estimation of how hard humans find the task is included. The authors claim "achieving consistently high agreement is challenging and depends on interpretation, comprehension, and background knowledge" but do not demonstrate this by measuring expert agreement on policy instrument, target group, and theme labels.
* The classification agreement is calculated over "all overlapping cases where there is an overlap for at least one policy label" but should also be calculated over all cases.
* The cross-validation experiments are not defined in sufficient detail. How was the training data formatted? It couldn't have been the same as for the LLMs, given that RoBERTa has a tiny context window.
* The overlap analysis is too coarse grained. A finer grained analysis of what kinds of things LLMs and humans coincide on and what kinds of things they diverge on is needed. This can be seen in problematic claims like "LLMs tend to provide more detailed procedural and descriptive accounts, human experts emphasize the contextual and societal implications of policy initiatives" when the analyses show only that humans and LLMs diverge more on objectives than they do on description. A finer grained analysis that studies "procedural" and "societal" language in the summaries would be needed to support such claims.

### Questions
* Are all of the models in the cross-validation section fine-tuned? E.g., you fine-tuned GPT-4o 128k?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper focuses on assigning labels to Science, Technology, and Innovation (STI) policies for documents of OECD countries. It develops an LLM-based classification system that integrates intermediate validation steps -- like relevance and evidence scoring. They evaluate the system with an existing set of answers by human respondents.

### Strengths
The paper clearly has strengths. The overall topic is likely very relevant for a lot of people in the policy space. The use case is well-presented and worked out with a good amount of carefulness. It is fairly well-written.

### Weaknesses
However, this paper presents a clear mismatch with what I would consider relevant for an AI conference. It is an LLM engineering use case from which possibly policy-related researchers could be inspired in their respective publication outlets. It does not deliver a significant contribution (method, datasets, etc.) in the AI/ML space. It just applies an LLM to data. This shallowness in contribution is also reflected in the fact that only superficial literature is cited with no reference to an actual stream of literature in ML, where this would be relevant. I think any other evaluation is pointless, given that the paper belongs to a different community.

### Questions
What literature do you contribute to?
Which method or data innovation do you present?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel application of Large Language Models (LLMs) to automate data extraction and survey completion for the OECD STIP Compass, which monitors Science, Technology, and Innovation (STI) policies across countries. The authors develop a pipeline using GPT-4o with long-context in-context learning to extract structured information from policy documents and generate survey responses. Testing on six countries, they achieve 84-95% overlap with human respondents on structured policy indicators, though qualitative differences emerge in free-text fields where LLMs provide more procedural detail while humans emphasize contextual impacts.

### Strengths
The paper addresses a genuinely important problem in policy research: the labor-intensive nature of international policy monitoring. The methodology is reasonably transparent, including validation mechanisms and multiple evaluation approaches (overlap analysis, agreement scoring, cross-validation). The recognition that LLMs and human experts complement rather than substitute for each other is sophisticated and realistic. The finding that LLMs excel at structured extraction but differ from humans in interpretive framing aligns well with social science understanding of how context and tacit knowledge shape policy analysis. The study's acknowledgment of potential biases and "model collapse" risks demonstrates appropriate caution about AI-generated data quality.

### Weaknesses
The evaluation relies heavily on overlap metrics rather than assessing the quality or validity of policy interpretation. High agreement doesn't necessarily mean accurate understanding of policy intent or impact. the country selection is reasonable but also limits generalizability. The paper does not address LLMs possible systematically misunderstand of culturally-specific policy contextsThere is no discussion of how power dynamics and epistemological questions (who defines what counts as valid policy knowledge?) intersect with AI automation. 

Also, policies in countries with less digital transparency, or older initiatives, will be systematically underrepresented. The maper measures web presence as much as policy content. This is a fundamental validity threat the paper barely acknowledges.

The paper lacks an assessment of whether LLM-generated policy classifications maintains the correlation structures, distributional properties, or causal relationships that policy researchers need. Overlap does not equate to statistical exchangeability.

The paper lacks a formal framework for weighting human vs. AI inputs. It has no analysis of when to trust which source, no error propagation model.

### Questions
How do you quantify and communicate the uncertainty in your LLM-generated policy labels? The binary validation (0/1 for "evidenced" and "relevant") seems crud. Could one use confidence scores or calibrated probabilities? How should policy analysts interpret disagreements between human and AI responses?Which types of policies or countries does the model systematically fail on? Can you provide instance-level confidence measures? You claim this enables 'scalable' policy monitoring, but you've only tested six countries and relied on expensive proprietary models (€447 for 845 samples). What's the actual cost-benefit at scale, and what happens when GPT-4o becomes unavailable or changes?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces an AI-assisted data extraction pipeline for the OECD STIP Compass survey, which collects cross-country data on Science, Technology, and Innovation (STI) policies. The authors propose using Large Language Models (LLMs) as “AI respondents” that can automatically pre-fill structured survey fields (policy instruments, target groups, themes) from policy webpages using long-context in-context learning (ICL). A second LLM layer validates these outputs by assessing relevance and evidence.

The pipeline is tested across six OECD countries (Canada, Finland, Germany, South Korea, Spain and Turkey). Results show high agreement between AI- and human-filled structured codes (84–95%) and reasonable alignment in free-text descriptions (~74% high overlap). Textual differences reveal complementary strengths: LLMs generate procedural detail, while humans emphasize contextual and societal framing. Cross-validation (GPT-4o vs others) yields ~98 F1 micro for AI labels. The authors conclude that hybrid human: & AI survey workflows could reduce reporting burden and increase consistency, provided oversight avoids bias or model collapse from synthetic feedback loops.

### Strengths
**MAJOR STRENGHTS**

- **Useful + impactful for policy work**: a practical pipeline that can materially reduce survey burden and increase consistency in STI policy monitoring.
- **Well-written + easy to follow**: clear structure and framing; good figures/tables; reads smoothly even for non-specialists.
- **Interesting application**: treats LLMs as AI respondents to pre-fill structured policy surveys (not just generic classification).
- **Sound two-stage design**: long-context ICL for extraction + a validator LLM that scores relevance/evidence (mitigates hallucination/misreadings).
- **Solid empirical grounding**: multi-country evaluation with quantitative overlap/F1 results against human entries.
- **Human+AI complementarity surfaced**: LLMs add procedural/detail; humans add context/societal framing, which is a potentially useful insight that hints at a deeper human vs LLMs follow-up paper?
- **Generalizable blueprint**: pipeline looks readily adaptable to other policy domains (beyond OECD/STI) and to human–AI co-workflows.
- **Risk-aware**: discusses pitfalls (hallucination, synthetic-feedback/model-collapse), not just benefits.
- **Operationally realistic**: choices (long-context prompting, schema outputs, validator checks) feel deployable by policy teams.

### Weaknesses
**MAJOR WEAKNESS/MISMATCH**
- **Overall fit (why I list this as a poor contribution for ICLR):** While the paper is genuinely useful and seems to be well-executed for policy analysis, it reads primarily as an LLM application/pipeline for policy data collection (human+AI survey workflows) rather than as a core methodological contribution in ML. That makes it feel better suited to policy-oriented venues (AI for governance, digital government, measurement) than to ICLR’s typical focus on new learning algorithms, theory, or broadly generalizable methodology? I’m open to being convinced otherwise by the editor/chair, but my expertise and reviewing lens skew toward method innovation, not domain-specific deployment studies; I can’t fully judge the policy-side contribution. This mismatch, plus how different this paper is from the rest of the papers I'm currently reviewing for ICLR, drives my evaluation of contribution/venue fit.

**MINOR POINTS**
- **Make “causal models” in the K-fold section explicit**: the paper mentions using causal models for cross-validation, but please specify which models, what “causal” means here (e.g., structure priors? debiasing? just a model name?), and how they affect label quality or agreement.
- **Code and assets availability**: if a repo or artifacts (prompts, validators, evaluation scripts, country lists/URLs) are not publicly available, please release them; if they are, surface the link prominently (abstract or intro).
- **Broaden evaluation beyond overlap**: is it possible to add expert adjudication on a stratified sample (policy analysts rating correctness/justification), and report hallucination/error categories (fabricated initiatives, wrong mappings, stale links)? That would strengthen conclusions.
- **Human+AI workflow clarity**: it would be nice if there were a pipeline figure showing hand-offs (AI pre-fill → validator → human editor), with failure modes and fallback steps (what triggers manual review); just a suggestion.

### Questions
See points above.

### Soundness
3

### Presentation
3

### Contribution
1
