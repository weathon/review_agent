# Natural Identifiers for Privacy and Data Audits in Large Language Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 2, 6

## Abstract
Assessing the privacy of large language models (LLMs) presents significant challenges. In particular, most existing methods for auditing *differential privacy* require the insertion of specially crafted canary data *during training*, making them impractical for auditing already-trained models without costly retraining. Additionally, *dataset inference*, which audits whether a suspect dataset was used to train a model, is *infeasible* without access to a private non-member held-out dataset. Yet, such held-out datasets are often unavailable or difficult to construct for real-world cases since they have to be from the same distribution (IID) as the suspect data. These limitations severely hinder the ability to conduct scalable, *post-hoc* audits. To enable such audits, this work introduces **natural identifiers (NIDs)** as a novel solution to the above-mentioned challenges. NIDs are structured random strings, such as cryptographic hashes and shortened URLs, naturally occurring in common LLM training datasets. Their format enables the generation of unlimited additional random strings from the same distribution, which can act as alternative canaries for audits and as same-distribution held-out data for dataset inference. Our evaluation highlights that indeed, using NIDs, we can facilitate post-hoc differential privacy auditing *without any retraining* and enable dataset inference for any suspect dataset containing NIDs without the need for a private non-member held-out dataset.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces Natural Identifiers (NIDs) as a novel mechanism to enable post-hoc privacy and data auditing in large language models (LLMs) without retraining or requiring held-out datasets. The authors define NIDs as structured random strings (e.g., cryptographic hashes, shortened URLs, Ethereum addresses) naturally present in public datasets such as GitHub and StackExchange.
The paper demonstrates that these identifiers can act as natural canaries allowing differential privacy (DP) audits and dataset inference (DI) to be performed post training. It adapts the one run DP auditing framework (Steinke et al., 2023) to work with NIDs and introduces a ranking-based inference procedure that computes tighter DP lower bounds. Extensive experiments on Pythia and OLMo models show that NIDs yield reliable post-hoc DP auditing and DI with no retraining and zero false positives, advancing practical LLM privacy auditing.

### Strengths
1. Uses real-world random structures (NIDs) as reusable canaries a clever, previously unexplored idea.
2. Adapts Steinke et al. (2023) into a ranking based auditing theorem (Theorem 1) with rigorous bounds.
3.  Evaluations across models (Pythia, OLMo) and datasets (Pile, Dolma) confirm the scalability and reliability of the method.
4. Eliminates retraining and manual canary insertion, reducing the computational barrier for auditing large LLMs.
5. Directly addresses real world auditing needs in LLM regulation and compliance contexts.

### Weaknesses
1. The method assumes datasets contain sufficient NIDs; domains lacking structured identifiers (e.g., medical text) may not benefit.
2.  While appendices are comprehensive, code availability is not stated; releasing tooling for NID extraction would improve accessibility.
4.  Results for smaller ε or limited NIDs could use confidence intervals across more random seeds.
5. A short discussion of potential misuse (e.g., adversarial data reconstruction) would strengthen the ethics section.

### Questions
1. How sensitive is DI performance to the number and diversity of NIDs in the suspect dataset?
2. How does the ranking-based ε-bound compare empirically with the two-alternative case when the same number of samples is used?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces natural identifiers (NIDs), ie random-looking strings common on the web (such as hashes, short URLs, eth addresses) as a scalable basis for post-hoc privacy auditing and dataset inference in LLMs. As NIDs have known generation rules, the authors mint matched generated identifiers (GIDs) to serve as iid controls, replacing bespoke canaries. This enables a single-run DP audit (picked up from the literature) that ranks NIDs vs. GIDs to yield tighter lower bounds with fewer samples, and a dataset-inference test that flags whether a suspect corpus was in training using only that corpus plus GIDs. Experiments on Pythia/OLMo and standard datasets show accurate hits on true training subsets and no false positives on held-out data. This makes NIDs a practical, general auditing tool.

### Strengths
The paper repurposes naturally occurring structured random strings as audit beacons and then samples from their known generators to create IID controls; this is a simple but fresh reframing that removes the retraining crutch many prior audits depend on, and I found it genuinely interesting and surprising; the ranking-based adaptation of one-run DP auditing further broadens that idea beyond top-1 decisions and tightens the link to theory.

The work further formalizes NIDs/GIDs and their sampling assumptions, connects the audit to an epsilon-DP theorem, and then evaluates on models with known training data (Pythia, OLMo) where ground truth is verifiable, reporting consistent DI behavior with no false positives on held-out subsets and showing how cardinality affects statistical power.

In terms of clarity, the exposition is well structured: the paper motivates why IID held-out data is the bottleneck, illustrates NIDs with concrete examples (e.g hashes, shortened URLs, crypto addresses), and uses clean figures and notation to map the end-to-end pipeline from NID extraction to GID generation to ranked auditing. 

Overall I believe the paper's significance is credible, because NIDs are abundant across standard pretraining corpora, making the method deployable in practice and potentially valuable for post-hoc audits or discovery in contentious settings.

### Weaknesses
In my view, the paper's main issue is that it depends on artifacts that are trivially removable at ingest: hashes, wallet addresses, shortened URLs, and similar strings are easy for a data curator to strip, normalize, or mask with a one-line regex during scraping or deduplication. Given that the method's leverage comes from these rigid formats and abundance, a mild shift in preprocessing policy could sharply reduce coverage, making the technique short-lived in practice. The authors show that many current corpora contain thousands of such identifiers (especially code-heavy subsets like GitHub/StackExchange and broad crawls like RefinedWeb) but those same tables also reveal how uneven the distribution is across domains, foreshadowing fragility under even modest curation pressure. A responsible next step would be to measure how counts and audit power change under realistic filtering pipelines (newline/HTML stripping, de-tokenization, heuristic hash removal, URL canonicalization), and to report DI and epsilon lower bounds before/after such defenses on the same models and suspect sets.

### Questions
1. How robust is the approach to trivial preprocessing that filters or normalizes NIDs? Today's training pipelines often apply regex-based scrubbing and URL canonicalization; because your extraction relies on highly rigid patterns (e.g., [a-fA-F0-9]{32,128}, 0x[a-fA-F0-9]{40}), a curator could remove or mask these in one pass. Please quantify the drop in NID counts and DI/DP-audit power after simulating common cleaning steps. A table like the distribution-of-NIDs summary, but "before vs. after" realistic filters, would show whether the method degrades to impracticality with minimal countermeasures.

2. Are GIDs sampled from the exact same effective distribution as the observed NIDs, not just the generator's support? Many NIDs in the wild reflect non-uniformities (e.g., address casing biases, popular shorteners, repository-specific hash exposure).

3. Is it possible to analyze failure modes where small generator inaccuracies leak membership signal? For example, Ethereum checksums, preferred hash casing patterns, or URL-shortener path semantics might let the model distinguish NIDs from your GIDs.

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
3

### Summary
This work introduces using natural identifiers as a novel way to audit data privacy in large language models. Natural identifiers are structured random strings such as the output of hash algorithms, shortened URLs, and crypto addresses, which are (i) naturally included in datasets such as Pile and Dolma and (ii) easy to generate held-out data from the same distribution. The authors demonstrates the effectiveness of this idea thru post-hoc DP audinting without training and practical dataset inference.

### Strengths
1. The idea is clear and well-motivated, smartly tackling the key problem of how to find held-out data from the same distribution.
2. The authors provide extensive empirical experiments, including ε estimation and dataset inference, to demonstrate the effectiveness of using natural identifiers in real-world scenarios.

### Weaknesses
1. The limitation of natural identifiers is obvious: Will this kind of data continually exist in pretraining data? In other words, if one model developer does not want their models to be audited, it is fairly easy for them to remove the natural identifier without sacrificing the model quality.
2. For dataset inference, it requires the audited dataset to contain natural identifiers, which could be hard constraint. For example, if someone want to test whether GSM8K is contained in training, it seems hard to use natural identifiers.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies post-hoc privacy audits for evaluating and quantifying LLMs privacy risks. 
It targets (i) auditing differential privacy (DP) claims: empirically estimating a lower bound on the DP budgets and (ii) dataset inference (DI): inferring whether an entire data subset was used to train the model. 

The key idea is to use Natural Identifiers (NIDs)—structured, random-looking strings (e.g., hashes, shortened URLs) that naturally appear in the training set—and to generate additional strings from the same distribution. NIDs enable the generation of unlimited additional random strings from the same distribution, serve both as canaries for DP auditing and as same-distribution held-out data for DI.

### Strengths
- Studies an important problem of LLM privacy audit

- Good articulation of the limits of prior privacy audits

### Weaknesses
1. Dedicated injected canaries vs NIDs. It’s unclear how results depend on which NIDs you pick. Prior work uses dedicated canaries injected from out of the training set; yours are random strings within the training set. Please quantify whether in-distribution vs OOD canaries change audit power and bias.

2. Why tighter DP lower bounds / lower sample complexity? You reported "Our privacy auditing with NIDs improves the lower bounds on the privacy parameters of an algorithm compared to the auditing framework by Steinke et al. (2023). It also significantly reduces the sample complexity, i.e., it requires fewer NID canaries."

3. DI performance/gains look modest. Is this because of the inherent limitation of your work in which you infer whether an entire data subset was used to train the model by reasoning only on NIds within the dataset, not the whole dataset?

4. Comparisons missing with Zhang et al., 2024 and Zhao et al., 2025 for DI (both performance and efficiency) to support your claims against these prior works. Also experiments lack the use of more recent MIAs.

5. Scope / external validity. Results rely on Pythia/Pile and OLMo/Dolma. Where do NIDs not exist? Provide a coverage analysis and a failure mode.

6. Over-reach beyond LLMs. You mention DI for Diffusion and Image Autoregressive Models "Beyond LLMs, DI has also been successfully applied to other types of generative models, including Diffusion Models (Dubinski et al., 2025) and ´ Image Autoregressive Models (Kowalczuk et al., 2025), widening the potential of the method." but don’t evaluate them. Either add experiments or tone down claims.

7. Lack significant technical contribution wrt to the closest prior work. As you report "Zhang et al. (2024a) propose to inject random and meaningless canaries into the data and then test how the LLM ranks the selected canary among all alternatives."
Relative to Zhang et al. (injected random strings), your technical contribution is switching to NIDs that are naturally included in LLMs’ training sets. I would note this earlier in the paper, e.g., in the introduction too.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work aims to address two of the most significant and challenging research questions in the field of privacy auditing for LLMs: the post-hoc implementation problem of differential privacy (DP) auditing, and the dependence of dataset inference (DI) on independent and identically distributed (IID) held-out sets. The authors identify two core shortcomings of current LLM privacy auditing tools in real-world deployment: the infeasibility of DP auditing and the IID constraint in dataset inference DI. To address both issues, this work introduces a novel and elegant concept: Natural Identifiers.

### Strengths
1. The NID approach provides a clever, elegant, and theoretically sound solution to a long-standing and practically intractable bottleneck in the field of DI: the “IID held-out set” limitation.
2. The proposed approach seems interesting. It operates post-hoc and is capable of generating IID samples with zero distributional shift.
3. The empirical results seem promising.

### Weaknesses
1. The proposed method in this work relies entirely on the existence of Natural Identifiers, which are found almost exclusively in technical and code-related corpora (e.g., GitHub, StackExchange) and are virtually absent in general text corpora such as books or news articles. Therefore, this work does not appear to be a general-purpose LLM auditing tool. The authors deliberately avoid discussing this major limitation in the paper.
2. The core differential privacy claim in this paper is that it audits pre-trained LLMs; however, the differential privacy experiments seems are conducted only on fine-tuning, not on pre-training.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
