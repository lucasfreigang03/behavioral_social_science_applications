# Behavioral Social Sciences Applications Project

## Quick Start

**Requirements:** conda ([Miniforge](https://github.com/conda-forge/miniforge) or [Anaconda](https://www.anaconda.com/download)), a [Deepgram](https://deepgram.com) API key (transcription), a [Groq](https://groq.com) API key (spontaneity check), and the [PodcastIndex](https://podcastindex.org) database dump (~5 GB).

```bash
conda env create -f environment.yml    # Python 3.13, ffmpeg, all packages
conda activate bssa_project

cp .env.example .env                   # add your DEEPGRAM_API_KEY and GROQ_API_KEY
```

With Anaconda Navigator, you can instead import `environment.yml` under *Environments → Import*.

Place the extracted dump at `data/podcastindex_feeds.db`.

**Run:** open `processing/run_pipeline.ipynb`, select the `bssa_project` kernel, and run the cells from top to bottom. This filters candidate podcasts, builds the ranked episode table, runs a smoke test with one podcast per group and then the full run: download, transcription, ad removal and spontaneity check (audio and its duration → `downloads/`, transcripts → `transcriptions/raw/`, ad marks → `transcriptions/ad_cleaned/`, result tables → `data/`). All settings that define the study (years, groups and filters, number of podcasts and episodes, excerpt, ad rule, models, paths) are in `config.py` and follow the preregistration (AsPredicted #326482, `preregistration-ABUELNAJ-DANZO-FREIGANG.pdf`). Do not change them once data collection has started.

The notebook must run with its own folder as working directory (VS Code default), since it finds the project folder from there. Transcription costs API credit, so check the smoke test before the full run. After a change in `config.py`, restart the kernel.

A run can be interrupted and continued: finished downloads, transcripts and ad files are reused from disk, and the result table (`data/processed.pkl`) is picked up again by the full-run cell.

**Pilot data:** the files of the pilot run (games/science test filters, minute 5–15, transcribed with `smart_format`) are kept in `pilot/` (ignored by git). They are not study data, but can be used to prepare the analysis code. The paths in `pilot/data/processed.pkl` still point to the old folders: replace `/downloads/` with `/pilot/downloads/` and `/transcriptions/` with `/pilot/transcriptions/`.

### Code overview

| File | Content |
|---|---|
| `config.py` | All settings that define the study (as preregistered) and all paths |
| `preprocessing/filter_index.py` | Stage 1: filter the PodcastIndex dump |
| `preprocessing/extract_episodes.py` | Stage 2: read the RSS feeds, one row per episode with publication date |
| `preprocessing/fetch_pipeline.py` | Stages 1–3: episode table with random ranks |
| `processing/download.py` | Stage 4: duration via `ffprobe`, middle 10 minutes via `ffmpeg` |
| `processing/transcribe.py` | Stage 5: Deepgram transcription, functions to load a transcript |
| `processing/remove_ads.py` | Stage 6: keyword rule; `load_excerpt` returns the 1,200-word excerpt the measures are computed on |
| `processing/keywords/ad_keywords.txt` | Phrase list for Stage 6 |
| `processing/classify_spontaneity.py` | Stage 7: spontaneity score via LLM |
| `processing/processing_pipeline.py` | Loop over Stages 4–7 with screening, main phase and replacement |
| `processing/run_pipeline.ipynb` | Runs everything and shows the results |

### Output for the analysis (Stages 8–9)

`data/processed.pkl` (load with `pd.read_pickle`) has one row per episode. Use the rows with `podcast_status == "passed"` and `episode_status == "processed"`. `processed` only means that the episode went through all stages; whether it counts as scripted is in the column `scripted`.

| Column | Content |
|---|---|
| `group`, `feed_url`, `podcast_title`, `episode_title`, `episode_date`, `episode_year` | Identification (`group` is `tech` or `comedy`, `episode_date` in UTC) |
| `podcast_hash`, `episode_hash` | Random ranks; `episode_hash` is also the file name in all folders |
| `episode_status`, `fail_stage`, `episode_fail_reason` | `pending` $\to$ `downloaded` $\to$ `transcribed` $\to$ `cleaned` $\to$ `processed`, or `failed` with stage and reason |
| `podcast_status`, `podcast_fail_reason` | `pending` (not needed), `passed` (screening passed) or `failed` with reason |
| `audio_path`, `transcript_path`, `cleaned_path` | Files of the episode |
| `audio_duration_sec`, `duration_source`, `excerpt_start_sec` | Episode duration (`ffprobe`, or `feed` if that failed) and start of the 10-minute excerpt (midpoint − 5 min) |
| `n_words`, `n_ad_words` | Ad-free words without filler words (at least 1,200), words removed as ads |
| `n_speakers`, `main_speaker_share`, `mean_confidence` | Computed on the 1,200-word excerpt (covariates) |
| `score`, `scripted`, `scripted_reason`, `llm_model` | Spontaneity check |

The text of an episode: `load_excerpt(transcript_path, cleaned_path)` from `processing/remove_ads.py` returns the excerpt as preregistered: the first 1,200 ad-free words without filler words, in order. Each word is a dict with `token` (the form to compute the measures on), `word`, `punctuated_word`, `start`, `end`, `confidence` and `speaker`. `token` is Deepgram's lowercased `word`. Numbers are spelled out, since transcription runs without `smart_format`. Every character except letters, digits and apostrophes is removed (`word-of-mouth` → `wordofmouth`), while `don't` stays, so it does not merge with `dont` (and `we're` not with `were`). For the robustness checks: `load_excerpt(transcript_path, remove_ads=False)` gives the excerpt without ad cleaning, and `speaker` allows host-only text.

----------------------------

# Study Plan

## General Idea

**Background.** LLM use homogenizes human language and reasoning ([Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003)). A meta-analysis finds a significant, small homogenization effect in human–AI co-creation ([de Rooij & Biskjaer](https://doi.org/10.1080/0144929X.2026.2726451)). Most of this evidence comes from written language, typically lab tasks done with or without AI.

**Why it matters.** Less diversity between people can reduce collective creativity ([Doshi & Hauser](https://doi.org/10.1126/sciadv.adn5290), [meta-analysis](https://arxiv.org/abs/2505.17241)); [Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003) give an overview.

**Gap and research question.** **Does AI also lead to homogenization in spoken language?** [Sourati et al.](https://doi.org/10.1016/j.tics.2026.01.003) suggest it but do not test it. [Yakura et al.](https://doi.org/10.48550/arXiv.2409.01754) show that AI-typical words (like "delve") increase in podcasts, but do not look at homogenization.

**Approach.** Take Tech podcasts (higher presumed AI exposure) and Comedy podcasts (control) that ran through the whole timeframe. For 2 episodes per podcast and year, transcribe the middle 10 minutes and take the first 1,200 ad-free words. Compute four lexical diversity measures per excerpt, and for each podcast-year how far the podcast deviates from the other podcasts of its group ($d$). Compare the change in $d$ from before to after the release of ChatGPT between the two groups (Difference-in-Differences).

**Open points for the introduction.**
* Literature search to confirm that homogenization of spoken language has not been studied yet (the gap is a strong claim).
* Whether to also motivate the topic via moral responsibility, in addition to collective creativity.

## Hypothesis

* **H1 (confirmatory):** Following ChatGPT's release (30 Nov 2022), spoken language in podcasts becomes more homogeneous. Tech podcasts converge more strongly in language style than Comedy podcasts from the pre-period to the post-period. Supported if the DiD estimate (Tech × Post) on $d$ is negative and its 95% bootstrap CI excludes 0 (two-sided, $\alpha = .05$).
* **Descriptive only, not tested:** AI-word rates (Yakura et al., 2024), scripted episodes per group and year, and the four measures separately (Stage 11).

----------------------------

## PLAN – Homogenization of Spoken Language in Podcasts
Course project, 2.5 ECTS (~60–75 h per person), 3 members

All design decisions are fixed in the preregistration (AsPredicted #326482, registered 2026-11-22). The values used by the code are in `config.py`. Operational details that the preregistration does not specify are listed in Stage 10.

---

### Labels

* **[CORE]** Necessary for a valid course project
* **[CHEAP]** Optional, 1–2 h, uses data generated anyway — do if time permits
* **[OPTIONAL]** Publication level — omit and mention in limitations
* Hours = rough estimate for coding share (Lucas), unless otherwise stated.

---

### Basic Principle

Each stage writes a file and skips already processed entries. No random sampling is done on the fly: every podcast and episode receives a fixed random rank once (hash of seed and ID). The first elements in rank order that pass all checks are selected. If one fails, the next in line takes its place. Every exclusion is logged with a reason (flowchart in the report).

---

### Stage 1: Candidates [CORE] — Completed

* Filter PodcastIndex dump: English (`language` starting with `en`), main category (`category1`) Technology or Comedy, oldest episode before 2018, newest episode in 2026, not dead. Only `category1` is used, so podcasts whose main topic is something else (e.g. business with technology as a secondary category) are not included. The dump has 452 Tech and 769 Comedy candidates.
* Values are matched case-insensitively with SQL `LIKE`, so `en%` covers `en`, `en-us`, `en-GB` and so on. All matching podcasts are returned (no limit), since the random order is applied to all of them in Stage 3.
* Note the snapshot date of the dump.

### Stage 2: Episode Table [CORE] — Completed

* Parse feeds. Columns: `feed_url`, `group`, `podcast_title`, `episode_title`, `episode_date` (UTC), `episode_year`, `episode_audio_url`, `episode_length` (seconds).
* Years: pre = 2018, 2019, 2021, 2022; post = 2023–2026. 2020 is excluded, since pandemic-related changes in recording conditions and formats could shift style independently of AI. 2022 episodes only count if published before the ChatGPT release (Nov 30, 2022, 00:00 UTC).
* Filter: duration at least 15 minutes, remove missing values.
* Keep a podcast only if it has at least $2 \times k$ suitable episodes in every year (buffer for replacements, intro and outro).

### Stage 3: Random Ranks and Status [CORE] — Completed

* `podcast_hash = SHA256(seed, feed_url)`; `episode_hash = SHA256(seed, episode_audio_url)`, seed 23. Uses `hashlib`, not Python’s built-in `hash()`.
* Status columns: `episode_status` (`pending` $\to$ `downloaded` $\to$ `transcribed` $\to$ `cleaned` $\to$ `processed`, or `failed` with `fail_stage` and `episode_fail_reason`) and `podcast_status` (`pending` / `passed` / `failed` with `podcast_fail_reason`).
* Selection: table sorted by `podcast_hash`, `episode_year`, `episode_hash`; take the first $N = 65$ non-failed, screened podcasts per group and the first $k = 2$ non-failed episodes per podcast and year, without data-dependent stopping.

#### Processing Order per Podcast (Important for Computational Effort)

* First, process only 2 screening episodes (Stage 4 $\to$ 5 $\to$ 6 $\to$ 7): the first in rank order from the first year (2018) and the first from the last **pre-period** year (2022). No post-period episode is used for screening (see Stage 7).
* Only process remaining episodes after the podcast passes the spontaneity check (Stage 7). The two screening episodes count toward this.
* If the podcast fails, only 2 audio excerpts ($20\text{ min}$ of audio) have been transcribed in vain.

### Stage 4: Download [CORE] — Completed

* For each podcast and year, take the first $k$ episodes in rank order.
* Read the duration of the audio file with `ffprobe` (fallback: `itunes_duration` from the feed, recorded in `duration_source`). Extract the middle 10 minutes (start = duration / 2 − 5 min) using `ffmpeg` directly from the URL (`-ss` before `-i`), $16\text{ kHz}$ mono. Duration and start are saved next to the audio file (`downloads/<episode_hash>.json`), so they are known when the file is reused.
* Temporary errors (e.g. timeouts) are retried up to 3 times. Download failed $\to$ episode `failed` with the error text, take next in rank order.
* Not implemented: download date, final URL and file hash are not logged.

### Stage 5: Transcription [CORE] + Speaker Separation — Completed

* Deepgram `nova-3` via API (`filler_words=True`, `diarize_model="v2"`, `paragraphs=True`, `punctuate=True`, `smart_format=False`). Same model and parameters for all years. The hosted model can change over time, so report the exact version stored in each response (`metadata.model_info`).
* Without `smart_format`, numbers and URLs stay spelled out as spoken ("twenty percent", "dot com slash"), so the excerpt needs no rule for formatted numbers. Punctuation (and with it the sentences used for ad removal) comes from `punctuate`.
* Save the full JSON response (word-level timestamps, confidence and speaker labels). No language detection: podcasts are already filtered to English by the feed language in Stage 1.
* Filler words are transcribed (Deepgram drops "uh" and "um" by default), since they are a marker of spontaneous speech and matter for the spontaneity check. They are removed from the 1,200-word excerpt for the measures, as preregistered: exactly Deepgram's filler words (`uh`, `um`, `mhmm`, `mm-mm`, `uh-uh`, `uh-huh`, `nuh-uh`, `FILLER_WORDS`).
* **[CHEAP]** Save mean word `confidence` per episode (column `mean_confidence`) and report per year (transcription quality).
* **[CORE]** Save number of speakers and the share of words spoken by the main speaker per episode (columns `n_speakers`, `main_speaker_share`, computed from the speaker labels in the 1,200-word excerpt). Preregistered covariates, since monologue and dialogue differ in vocabulary diversity.

### Stage 6: Ad Removal — Implemented, rule not validated yet

* **[CORE]:** Keyword rule on sentence level ("predefined keywords" in the preregistration). The phrase list is in `processing/keywords/ad_keywords.txt` (sponsor announcements, promo codes, offers, calls to action, legal disclaimers, break transitions, self-promotion; no brand names, since sponsors change over the years). Sentences with hits that are at most $30\text{ s}$ apart form a block. A block is cut only if it contains at least 2 different phrases, together with 2 sentences before and after. The rule is not changed after data collection has started. Fewer than 1,200 ad-free words without filler words $\to$ episode `failed`, replacement.
* The raw transcript is never changed. A second JSON file per episode (`transcriptions/ad_cleaned/`) holds all sentences with `hits` and `is_ad`, so the rule can be validated later without transcribing again.
* Runs before the spontaneity check, so the rule must be final before screening starts.
* Pilot (84 episodes from the games/science test filters): no sentence was marked as ad; 5 sentences had a single keyword hit, none of them an ad. Recall on real ads is untested.
* **[CHEAP]** Report proportion of removed words per year (indicator of dynamically inserted ads): column `n_ad_words` relative to `n_ad_words + n_words`.
* **[OPTIONAL]** Validation: Manually check 20–30 excerpts, half from the pre- and half from the post-period, and report precision and recall per period. Even trained classifiers reach only moderate accuracy on this task (Reddy et al., 2021).

### Stage 7: Spontaneity Check (Runs after ad removal on the 2 screening episodes) — Completed

* **[CORE]:** Podcast-level via LLM. Send the first 300 ad-free words of each of the two screening episodes, with speaker labels, to an LLM (`openai/gpt-oss-120b` via Groq). It returns a score from 1 (clearly scripted) to 4 (clearly spontaneous) and a one-sentence reason. A score below 3 counts as scripted. If either screening episode is scripted $\to$ podcast `failed`, next podcast in rank order. Model, score and reason are saved per episode; the prompt is in `processing/classify_spontaneity.py`.
* Uses ad-cleaned text, since sponsor reads are scripted and would bias the rating towards "scripted".
* **Screening uses pre-period episodes only.** If AI makes podcasts more scripted (hosts reading LLM-prepared text), screening on a post-period episode would remove exactly the affected podcasts and bias the result towards zero (selection on a post-treatment variable).
* **[CORE]** All excerpts are rated, not only the screening ones. Episodes of the main phase are kept even if they are scripted, only the score is recorded (used in Stage 11).
* **[OPTIONAL]** Validation: Read 30 excerpts and compare with LLM decision (~1 h, teammate).

#### Loop Stages 4–7 [CORE] — Completed

* Screening phase: repeat until the first $N$ non-failed podcasts per group have passed the spontaneity check.
* Main phase: repeat until every cell (Podcast $\times$ Year) of these podcasts has $k$ processed episodes. If a podcast runs out of episodes in a year, it fails and its replacement is screened first.
* An episode fails on a download or transcription error, on fewer than 1,200 words after ad removal, or on an LLM error. The next episode in rank order takes its place.
* No incomplete cells.

### Stage 8: Measures [CORE]

* Text per episode: `load_excerpt` (see "Output for the analysis"), measures computed on `token`. Lowercased word forms, not lemmatized, without punctuation.
* Per excerpt, the four measures as defined in Sourati et al. (2026):
  * Type-token ratio
  * Shannon entropy over single words (word order ignored), plug-in estimate. It underestimates entropy in short texts, but is comparable across excerpts since all have 1,200 words.
  * Simpson index, sign reversed (higher Simpson means less diverse)
  * Hapax legomena
* Per podcast-year: average each measure over the 2 episodes.
* z-standardize each measure with the pre-period mean and SD.
* Per measure: each podcast's absolute deviation from the mean of the **other** podcasts in its group and year.
* $d$ = mean of the four deviations. Lower $d$ means more homogeneity. Units: pre-period SDs of the individual measures.

### Stage 9: Statistics [CORE]

* **Model:** linear DiD regression of $d$ (one value per podcast-year) on Tech × Post, with podcast and year fixed effects (they absorb the main effects of group and period), and number of speakers and main speaker's share of words as covariates.
* **Interval:** 95% percentile podcast-cluster bootstrap CI, 10,000 resamples. In each resample, draw 65 complete podcast histories per group with replacement, recalculate each podcast's $d$ as defined in Stage 8 (excluding copies of the same podcast when computing the mean of the other podcasts), and refit the model.
* **Support for H1:** the DiD estimate is negative and its 95% CI excludes 0.
* **Pre-trends:** plot both groups' trajectories of $d$ over the years.
* No outliers are removed. 2020 is excluded from all analyses.

### Stage 10: Documentation [CORE]

* **[CORE]:** Flowchart of exclusions, limitations, and discussion of potential biases. Include the exact prompt and model used for the spontaneity check.
* **[CORE]** Report operational details that the preregistration does not specify: only `category1` for the group, language `en%`, Deepgram settings (no `smart_format`), the filler-word list, the token rule (`load_excerpt`), the episode duration from `ffprobe`, and how the covariates are aggregated to podcast-years (to be decided before the analysis, e.g. mean of the 2 episodes).

### Stage 11: Robustness and Descriptives [CORE]

Robustness (rerun of Stage 9):
* Without low-spontaneity episodes (score below 3, Stage 7).
* Without ad cleaning (`load_excerpt(transcript_path, remove_ads=False)`).
* Host-only text (main speaker's words, via `speaker`).
* Entropy with a correction for unseen words: Chao-Shen estimator (Chao & Shen, 2003). The plug-in estimate stays the main measure.
* Placebo DiD: 2018–2019 as pre and 2021–2022 as post.

Descriptive only (no test):
* AI-word rates (Yakura et al., 2024) per group and year, to assess relative AI exposure.
* Scripted episodes per group and year, to identify format changes.
* The four measures separately, to see what drives the composite.
* Cronbach's alpha of the four measures. The composite stays the outcome; if alpha is below .70, interpret it with caution alongside its components.

---

### Procedure

1. Code Stages 3–7 [CORE] — done, tested with simulated stages.
2. Pilot run — done (games/science test filters, 3 podcasts, 54 episodes, data in `pilot/`).
3. Preregistration — done (AsPredicted #326482, 2026-11-22).
4. Build the episode table for Tech and Comedy (section 2 of `processing/run_pipeline.ipynb`, no API cost) and check that each group has clearly more than 65 eligible podcasts.
5. Smoke test with the final settings: 1 podcast per group (section 4 of the notebook), for debugging only.
6. Full Run [CORE]: Deepgram cost scales with audio minutes: groups × $N$ × years × $k$ × 10 min = 2 × 65 × 8 × 2 × 10 = 20,800 min, plus screening of failed podcasts and replaced episodes. Check the current price per minute and the remaining credit before starting.
7. Analysis Stages 8–9 and 11 [CORE]: can be prepared with the pilot data.
8. Report [CORE, Team]: Including flowchart of exclusions and limitations.

---

### Work Distribution

Lucas: Stages 3–7
Carma: Stages 8–11
Gaia: Writing, report

---

### Limitations for the Report (One sentence each on why it matters)

1. **Parallel trends** (main limitation): Tech and Comedy may have experienced different industry or format changes unrelated to AI.
2. **Survivorship bias via panel design:** Only podcasts that ran through the entire timeframe.
3. **Speaker composition:** host and guests are analysed together and only controlled via number of speakers and main-speaker share; shared guests and networks link podcasts.
4. **Scripting:** spontaneity is screened in the pre-period only; podcasts may become more scripted later (reported via the per-episode score).
5. **Transcription and ad-removal errors:** spontaneity check and ad rule are not manually validated; dynamically inserted ads are removed only via keywords.
6. **Excerpt position is approximate:** `ffmpeg` seeking in variable-bitrate MP3s is estimated, `ffprobe` may estimate the duration from the bitrate, and dynamically inserted ads can differ between the `ffprobe` and the `ffmpeg` request and shift the content.
7. **Baseline variability and pre-trends:** only four pre-years (2020 excluded), so pre-trends are only roughly testable.
8. **Category labels** come from one snapshot of the PodcastIndex dump and may have changed over time.
9. **Comedy also has AI exposure, just less** $\to$ effect likely underestimated.
10. **English only, two categories.**
11. **Lexical diversity only:** the four measures describe vocabulary, not syntax or other parts of style.
12. **Limited text per podcast and year** (2 × 1,200 words) $\to$ noisy measures, low power for a small effect; a null result is not evidence of no effect.
13. **Plug-in entropy underestimates** entropy in short texts (comparable across excerpts, Chao-Shen as robustness check).
14. **AI-word rates are descriptive only.**
15. **Language is taken from the feed metadata,** not detected from the audio; single non-English episodes can slip through.
16. **ASR:** filler words may not be captured completely, and spelling of words and numbers follows Deepgram's conventions.

## Sources
- Anderson, B. R., Shah, J. H., & Kreminski, M. (2024). Homogenization Effects of Large Language Models on Human Creative Ideation. Proceedings of the 16th Conference on Creativity & Cognition, 413–425. https://doi.org/10.1145/3635636.3656204
- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜. Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, 610–623. https://doi.org/10.1145/3442188.3445922
- Bommasani, R., Creel, K. A., Kumar, A., Jurafsky, D., & Liang, P. (2022). Picking on the Same Person: Does Algorithmic Monoculture Lead to Outcome Homogenization? Advances in Neural Information Processing Systems, 35. https://doi.org/10.52202/068431-0265
- de Rooij, A., & Biskjaer, M. M. (2026). Does generative AI make us think alike? A systematic review and meta-analysis of homogenisation effects in human–AI co-creation. Behaviour & Information Technology, 0(0), 1–14. https://doi.org/10.1080/0144929X.2026.2726451
- Doshi, A. R., & Hauser, O. P. (2024). Generative AI enhances individual creativity but reduces the collective diversity of novel content. Science Advances, 10(28), eadn5290. https://doi.org/10.1126/sciadv.adn5290
- Duede, E., Teplitskiy, M., Lakhani, K., & Evans, J. (2024). Being together in place as a catalyst for scientific advance. Research Policy, 53(2), 104911. https://doi.org/10.1016/j.respol.2023.104911
- Hong, L., & Page, S. E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers. Proceedings of the National Academy of Sciences, 101(46), 16385–16389. https://doi.org/10.1073/pnas.0403723101
- Muthukrishna, M., & Henrich, J. (2016). Innovation in the collective brain. Philosophical Transactions of the Royal Society B: Biological Sciences, 371(1690), 20150192. https://doi.org/10.1098/rstb.2015.0192
- Nijstad, B. A., Stroebe, W., & Lodewijkx, H. F. M. (2002). Cognitive stimulation and interference in groups: Exposure effects in an idea generation task. Journal of Experimental Social Psychology, 38(6), 535–544. https://doi.org/10.1016/S0022-1031(02)00500-0
- Page, S. E. (2008). The Difference: How the Power of Diversity Creates Better Groups, Firms, Schools, and Societies. Princeton University Press.
- Sourati, Z., S. Ziabari, A., & Dehghani, M. (2026). The homogenizing effect of large language models on human expression and thought. Trends in Cognitive Sciences, 30(9), 805–816. https://doi.org/10.1016/j.tics.2026.01.003
- Yakura, H., Lopez-Lopez, E., Brinkmann, L., de la Serna, I., Kirfel, L., Gupta, P., Soraperra, I., Eisenmann, T. F., Wulff, D. U., & Rahwan, I. (2024). Empirical evidence of Large Language Model's influence on human spoken communication. arXiv. https://doi.org/10.48550/arXiv.2409.01754

To do: one entry could not be identified (Scopus record https://www.scopus.com/pages/publications/85166037135). Sources used in the plan but still missing here: Chao & Shen (2003, Chao-Shen entropy estimator), Reddy et al. (2021, Detecting Extraneous Content in Podcasts, EACL, https://aclanthology.org/2021.eacl-main.99).
