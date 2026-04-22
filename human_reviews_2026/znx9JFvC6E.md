# Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
The security issue of large language models (LLMs) has gained wide attention recently, with various defense mechanisms developed to prevent harmful output, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the output distribution of text embedding models is severely biased with a large mean. Inspired by this observation, we propose novel, efficient methods to search for **universal magic words** that attack text embedding models. Universal magic words as suffixes can shift the embedding of any text towards the bias direction, thus manipulating the similarity of any text pair and misleading safeguards. Attackers can jailbreak the safeguards by appending magic words to user prompts and requiring LLMs to end answers with magic words. Experiments show that magic word attacks significantly degrade safeguard performance on JailbreakBench, cause real-world chatbots to produce harmful outputs in full-pipeline attacks, and generalize across input/output texts, models, and languages. To eradicate this security risk, we also propose defense methods against such attacks, which can correct the bias of text embeddings and improve downstream performance in a train-free manner.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates a new vulnerability in LLM that rely on text embedding models. The authors identify a strong bias in the embedding distribution and leverage it to design efficient algorithms for finding universal magic words, which are an adversarial suffixes that can manipulate embedding similarity. In this way, these magic words can bypass text-embedding-based safeguards by altering perceived similarity between harmful and benign text pairs. The paper proposes three methods (brute-force, context-free, and gradient-based), evaluates their efficiency and transferability across models and languages. Moreover, the authors suggest renormalization-based defenses that improve robustness without retraining.

### Strengths
1. The paper studies vulnerability from an interesting angle by finding the universal magic words.
2. The paper shows strong attack and defense results for both the attack and the defense strategy.
3. The paper is well motivated and well written.

### Weaknesses
1. There lack of analysis on the possible number of magic words existing in a model.
2. The influence of repetition count, token length, or embedding normalization choices is not systematically analyzed.
3. There lack of analysis on the randomness in learning magic words across different random seeds, etc.
4. There lack of discussion on the origin/root/insights of the identified magic words.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper shows that many text-embedding models have a strong mean-bias in their vector space and that short, universal “magic-word” suffixes can push any input toward this bias direction, manipulating cosine similarities that underpin embedding-based safeguards. Building on this observation, the authors present efficient search procedures—including a context-free black-box method aligned to the bias and a one-step white-box gradient approach—to find transferable suffixes. They demonstrate end-to-end jailbreaks by appending these words to user prompts and by requiring the model to end its responses with the same words, thereby bypassing both input and output guards. Experiments across multiple embedding backends and safety detectors report large drops in detection performance and cross-model transfer. To mitigate, the paper proposes a simple, train-free fix—mean-centering plus renormalization of embeddings—which substantially restores guard performance.

### Strengths
* this paper proposes a bias-direction analysis for text-embedding models, which is new
* this paper offers a simple, train-free mitigation for the proposed attack

### Weaknesses
* the usage of "magic" suffix has been proposed in other works (like GCG), making the contribution of this paper a bit incremental
* using renormalization for defense is promising but its impact on diverse downstream retrieval/semantic tasks (beyond the reported classifiers) remains underexplored
* lacks head-to-head experimental comparison with other whitebox attacks
* some inherited limitations of whitebox attacks

### Questions
* Could you please explain the novelty of the propose methodology comparing to other similar attacks? e.g.: https://arxiv.org/abs/2307.15043, https://people.eecs.berkeley.edu/~daw/papers/iris-naacl25.pdf
* Some papers question the transferability/generalizability of these universal adversarial triggers (e.g.: https://arxiv.org/abs/2404.16020v1). Under what conditions do your universal suffixes fail, and how does that compare to the existing analyses?

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
2

### Summary
This paper introduces a novel attack vector against LLM safeguards that are based on text embedding models. Appending magical words can fool the safeguard, i.e., classifiers trained on text embedding to distinguish harmful and harmless prompts. The core of the work is a key observation: the output distribution of several popular text embedding models is highly anisotropic, concentrating in a specific "band" on the unit hypersphere. The authors formalize this bias by identifying a "bias direction" (e*), which is the normalized mean of a large corpus of text embeddings. The paper then proposes defense methods against such attacks by fixing the defect of uneven embedding distribution.

### Strengths
The discovery and empirical validation of the non-uniform, biased distribution of text embeddings (Fig. 1) is a significant and insightful contribution. It provides a principled and elegant explanation for the existence of universal adversarial attacks, moving beyond simple heuristics. This observation itself is of high value to the representation learning. The paper is well-written and the finding on embedding is interesting. The authors also propose different methods driven by their finding.

### Weaknesses
* Does correcting the bias harm the embedding model's performance on its primary tasks (e.g., semantic search, classification)? An empirical evaluation is necessary. The setting on bypassing safeguard may also not be so useful for real-world applicability.
* The final step of Alg. 3 involves a Cartesian product of candidate tokens, which can lead to a combinatorial explosion. The practical limits on the magic word length and candidate size should be discussed.
* The defense method has not been tested for adaptive attacks, such as whether the method can defend against changes in jailbreakers.
* In Alg. 3, the comment "empirically better than zeros(h,m)" for random initialization is an interesting detail. A brief sentence of intuition would be helpful for the reader.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
2
