Let me revisit the paper content provided in the user message. The key claims I need to verify:

1. Does the paper discuss access control for the retriever? 
2. Does the paper acknowledge that the GPT attack exploits the `myfiles_browser` tool?
3. Does the paper discuss the base model comparison fairly?
4. Does the paper frame RAG as providing privacy/copyright guarantees?

From the paper content:
- The introduction says: "RAG offers a mechanism for training LMs with low-risk data while moving high-risk data to external datastores... thereby supports attribution and opting out to hopefully avoid potential legal concerns"
- The GPT section does mention discovering `myfiles_browser` and `search` function from system prompts
- The ablation compares base vs instruction-tuned models

Now let me finalize my assessment:

**Key strengths verified:**
1. Comprehensive empirical evaluation across 10+ models
2. Well-designed ablation studies (instruction tuning effect, position effects, chunk size effects)
3. Real production system evaluation (100% attack success on 25 GPTs)
4. Concrete mitigation proposal (PINE + safety-aware prompts)

**Key weaknesses verified:**
1. The attack is simple - asking models to repeat context. This is a straightforward instruction-following exploit.
2. The threat model conflates retrieval layer access with model-level vulnerability
3. Mitigation evaluated on only one model (Llama3-8b)
4. Harry Potter results conflate memorization with extraction
5. The "RAG privacy" framing is overstated given that the retriever already provides access

Let me calibrate the score against similar papers:
- DEAL (similar RAG privacy attack, simple method): Rejected, scores 5,5,3,5
- Scalable Extraction (training data extraction from aligned models): Accepted poster, scores 6,6,8,6,8,6
- Catastrophic Jailbreak (simple attack on open-source LLMs): Accepted spotlight, scores 8,8,6,6
- BadRAG (RAG vulnerability): Withdrawn/Rejected, scores 5,3,3,3
- Prompt Infection (prompt injection in agents): Rejected, scores 5,6,5,5,5

This paper is in a similar space to DEAL and BadRAG but with significantly more empirical breadth and real-world evaluation. It's more comprehensive than DEAL but shares the same fundamental issue of a somewhat simple attack. Compared to Scalable Extraction (accepted poster), this paper provides real-world attack on production GPTs but has a less novel attack methodology. The Catastrophic Jailbreak paper (spotlight) had a simpler attack method but was very clearly presented and had cleaner evaluation.

Score calibration: The paper has solid empirical work, real-world significance, and practical implications for RAG deployment. But it has meaningful overclaiming issues and the core attack is straightforward. I'd place this around 5-6, leaning toward 5.5. It's better than DEAL (which was rejected) because of the comprehensive ablations and production system evaluation, but the overclaiming and threat model issues are significant enough to prevent a strong accept.</think>## Summary

This paper studies the vulnerability of Retrieval-In-Context (RIC) RAG systems to prompt-injected data extraction attacks, where an adversary instructs instruction-tuned LMs to regurgitate retrieved context verbatim. The authors demonstrate that this simple attack works effectively across multiple open-weight models (7B–72B), with vulnerability increasing with model size and instruction tuning, and extend the attack to production GPTs (achieving 100% attack success on 25 randomly selected GPTs and extracting 41% of a book with 100 queries). They propose PINE combined with safety-aware prompts as a mitigation strategy.

## Strengths

1. **Comprehensive empirical evaluation across model families and sizes**: The paper systematically tests 10+ instruction-tuned LMs across 7B–72B, consistently showing high extractability (e.g., Qwen1.5-72B reaching ~99 on all metrics). Table 1 provides a clear vulnerability landscape. The finding that larger, more capable models are *more* vulnerable is practically important and counterintuitive.

2. **Well-designed ablation studies providing mechanistic insight**: The paper goes beyond demonstrating the attack to investigating *why* it works. The instruction-tuning comparison (Figure 2, ~65 ROUGE point jump), seen vs. unseen knowledge source (Table 2), context size and chunking effects (Figures 3–4), and position effects (Figure 5, U-shaped curve) collectively offer genuine insight into the underlying mechanisms.

3. **Real production system demonstration with concrete metrics**: The 100% attack success rate on 25 GPTs (17 with a single query) and the reconstruction curves (Figure 6) showing 41.73% extraction from a book with 100 queries provide compelling practical evidence. The reconstruction rate metric is a useful corpus-level measure.

4. **Constructive mitigation proposal**: Rather than only identifying the vulnerability, the paper proposes PINE combined with safety-aware prompts (Table 3), reducing the reconstruction rate from 88.88% to 52.34%. This connects the position bias finding (Figure 5) to a concrete defense strategy.

## Weaknesses

### Fatal

None.

### Major

1. **The threat model conflates retrieval-layer access control failures with model-level vulnerabilities, and the framing overstates the implications for RAG privacy.** The paper frames RAG as offering a "balance between generation performance and the demands of data stewardship including copyright and privacy" (Introduction), then presents its results as undermining this promise. However, in the paper's own threat model, the adversary sends queries to the RAG system, and the retriever happily returns relevant chunks from the datastore to include in the model context. The "leakage" thus occurs at two levels: (a) the retriever is already serving up relevant documents, which any user could read by posing natural questions; and (b) the model can be instructed to regurgitate these documents verbatim. The paper does not separate these two channels or discuss what access control policies might prevent (a). For the GPTs attack specifically, the adversary invokes `myfiles_browser.search(...)`, which is the tool explicitly designed to retrieve and display file content—essentially using the system as intended. The paper acknowledges this ("runs a query over the file(s) uploaded in the current conversation and displays the results") but still treats this as a model safety failure. This conflation weakens the normative force of the broad privacy/copyright claims.

2. **Mitigation evaluation is too narrow to support the general claim that "position bias elimination strategies" are an effective defense.** Table 3 evaluates PINE + safety-aware prompts on only one model (Llama3-8b-Instruct), whereas all prior attack results use different models (Llama2, Mistral, Qwen, etc.). Given that the paper's own results show vulnerability varies dramatically across models (e.g., Qwen1.5-72b reaches ~99 on all metrics), validating the defense on only a single, different model is insufficient. Additionally, the reconstruction rate after mitigation remains at 52.34%, which the paper characterizes as "greatly mitigated" (Abstract) — but over half the corpus remains reconstructible. No comparison is made with simpler defenses (output filtering, retrieval access restrictions, context masking), leaving it unclear whether PINE offers genuine advantages over straightforward engineering solutions.

3. **The Harry Potter experiment conflates datastore extraction with training data memorization, and the paper's controls are insufficient.** Table 2 shows ~10 ROUGE-L point gains for Harry Potter (likely in pre-training data) vs. Wikipedia (unlikely in pre-training data), which the paper correctly notes "leads to a hypothesis that they have been trained on Harry Potter." However, the main GPT experiment (Figure 6) also uses Harry Potter to demonstrate 41.73% reconstruction without adequately separating memorization from extraction. The Wikipedia corpus experiment achieves only 3.22% reconstruction, which substantially undermines the severity of the claim regarding large unseen datastores. The gap between 41.73% and 3.22% is not fully explained by corpus size alone, and the paper lacks analysis of what fraction of the 3.22% constitutes novel content vs. boilerplate or generic text.

### Minor

1. **The observed instruction-tuning effect (Figure 2) does not clearly distinguish a "new vulnerability" from intended model behavior.** Base models scoring ~10-18 ROUGE versus instruction-tuned models scoring ~78-82 could simply reflect that instruction-tuned models correctly follow the explicit instruction to "copy and output all the text," while base models fail to parse the prompt format. The paper frames this as "instruction tuning substantially enhances exploitability" (Section 3.1), but an alternative interpretation is that instruction-tuned models are working as designed—they follow instructions, including adversarial ones.

2. **The adversarial prompts are straightforward and not tested for robustness.** The paper uses essentially one prompt template ("copy and output all the text before/after...") for the open-weight experiments. No exploration of robustness to prompt variation, model refusal, or safety guardrails beyond the minimal "Do not repeat any content from the context" is provided. The DEAL paper (Zeng et al., 2024), cited as the closest prior work, uses LLM-optimized prompts, but no direct comparison is made on shared settings.

3. **No analysis of utility-security trade-offs.** The paper does not measure any impact of PINE on normal RAG task performance (e.g., question answering accuracy), leaving open whether the defense fundamentally degrades RAG utility.

### Trivial

- The title "Follow My Instruction and Spill the Beans" is catchy but could be seen as informal for the venue.

## Nice-to-Haves

- Evaluate PINE mitigation on the same model families/ sizes used in the attack analysis (Llama2-70b, Qwen1.5-72b) where vulnerability is highest.
- Compare against simpler baseline defenses like output filtering (checking if generated text overlaps with retrieved context) or retrieval-level access controls.
- Report per-query extraction efficiency curves (diminishing returns analysis) and compare against naive retrieval without prompt injection.
- Include qualitative examples of extracted content from the Wikipedia corpus to show what the 3.22% reconstruction actually looks like.

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **Claim that the GPT attack is "not primarily a model safety failure" but an "access-control / API-design issue"** (Harsh Critic, Point 4): While the paper could better distinguish access-control from model-level issues, this is not purely an access-control problem. The adversary is able to *prompt the model to call a search function* that it should not invoke on behalf of users, and the model obeys adversarial instructions rather than its system prompt—this is genuinely a prompt injection vulnerability at the model level, even if access control could separately mitigate it. Partially removed: the point about access control is valid as a confound for the privacy framing, but the claim that the GPT attack is *entirely* an access-control issue is itself overstated.

2. **Demand for comparison with Zeng et al. (2024) on shared settings** (Human Finder, Weakness 1; Spark, Suggestion 5): The paper explicitly discusses and differentiates from Zeng et al. (2024) in Section 5, noting they did not test production systems or analyze underlying factors. A direct experimental comparison would be ideal but the differentiation in scope is sufficient. Moved to Nice-to-Haves.

3. **Demand for PII-specific evaluation** (Human Finder, Weakness 6): The paper's stated contribution is about data extraction from datastores (copyright, private data), not specifically PII leakage. Requiring a PII evaluation metric is scope creep.

4. **Formatting and presentation nitpicks** (various reviewers): Removed per the rule against formatting/style nitpicks.

5. **Criticism that the paper doesn't test non-RIC RAG architectures** (Spark): The paper explicitly scopes itself to RIC-based RAG systems. Testing other architectures would broaden scope but is not a flaw within the declared scope.

6. **Claim that the paper doesn't discuss the threat model at all** (Harsh Critic, Point 1): The paper does define a black-box threat model (Section 2) with explicit assumptions. The criticism that it should further specify access control policies is valid (moved to Major weakness 1), but claiming the threat model is entirely absent is factually wrong.

## Novel Insights

The finding that stronger instruction-following capability directly translates to *greater* vulnerability to data extraction is a genuine and important insight: as models become better at following instructions, they become equally better at following adversarial instructions embedded in user queries. This creates a fundamental tension between the goals of helpfulness and data privacy in RAG systems that cannot be resolved through alignment training alone. The U-shaped position bias curve (Figure 5) provides concrete mechanistic grounding for why this vulnerability exists and persists, linking it to well-documented recency and primacy effects in transformers.

## Suggestions

- Reframe the core claims: instead of presenting this as "RAG breaks privacy guarantees," position it as "RIC-RAG systems are vulnerable to context-regurgitation attacks via instruction following, and this vulnerability has important practical implications for datastore protection." This narrows the claim to what the evidence actually supports.

- Evaluate PINE on at least one 70B model where vulnerability is highest (e.g., Llama2-70b or Qwen1.5-72b) to establish whether the defense scales.

- Add a simple output-filtering baseline (e.g., check whether generated content overlaps with the retrieved context above a threshold and redact) to contextualize the practical value of PINE.

- For the GPTs experiment, add a condition using non-Harry-Potter text that can be confirmed absent from training data (e.g., synthetic or very recent documents) to cleanly isolate datastore extraction from memorization.

## Score and Decision

**Calibration against similar papers:**

- **DEAL** (RAG privacy extraction, similar scope): Rejected (scores 5,5,3,5). Slightly weaker than this paper—less comprehensive evaluation, similar simplicity of attack.
- **Scalable Extraction from Aligned LLMs** (training data extraction): Accepted poster (scores 6,6,8,6,8,6). Stronger novelty (divergence attack), more rigorous evaluation, but different problem setting (training data memorization vs. RAG datastore extraction).
- **Catastrophic Jailbreak** (simple decoding-based jailbreak on open-source LLMs): Accepted spotlight (scores 8,8,6,6). Similarly simple attack but clearer framing and thorough evaluation.
- **BadRAG** (RAG vulnerability, poor evaluation): Withdrawn/Rejected (scores 5,3,3,3). Much weaker than this paper.
- **Prompt Infection** (prompt injection in agents): Rejected (scores 5,6,5,5,5). Similar attack surface but less thorough evaluation.

This paper sits between DEAL (rejected) and Catastrophic Jailbreak (spotlight). It has substantially more comprehensive evaluation than DEAL and a real production system demonstration, which is a genuine contribution. However, the overclaiming on RAG privacy implications, the narrow mitigation evaluation, and the simple attack method prevent it from reaching the level of Catastrophic Jailbreak or Scalable Extraction.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>