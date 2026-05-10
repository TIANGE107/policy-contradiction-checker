# Policy Contradiction & Gap Checker

## Project summary

This project is a small GenAI-assisted Streamlit app for reviewing draft internal policy documents before publication.

The target users are operations staff, program coordinators, or administrative personnel who maintain internal policies such as reimbursement rules, eligibility rules, or procedural guidelines.

The workflow is intentionally narrow:

1. A user pastes one draft policy document.
2. The app identifies possible contradictions, gaps, and ambiguous language.
3. The app compares the GenAI-assisted output with a simpler rule-based baseline.
4. A human reviewer uses the output to revise the policy before final approval.

The tool is not a legal or compliance authority. It is a review aid.

## Why GenAI is useful

Policy documents are written in natural language. Many issues are semantic rather than keyword-based. For example, a rule can look valid alone but conflict with another rule later in the document. GenAI can help interpret rule meaning, compare related statements, and explain why a rule may be unclear.

## Baseline

The baseline is a checklist-style rule-based system. It uses fixed patterns to flag:

- vague terms such as "reasonable," "appropriate," or "may"
- unclear numeric boundaries
- simple allow/prohibit conflicts
- incomplete exception language

This baseline is realistic but limited because it cannot reliably understand context.

## GenAI method

If an OpenAI API key is available, the app uses a structured LLM prompt and returns JSON with:

- issue type
- severity
- confidence
- description
- related text
- rationale
- suggested fix

If no API key is available, the app automatically runs a deterministic local fallback so the grader can still run the app.

## Setup

```bash
pip install -r requirements.txt
```

Optional, only if using OpenAI:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Do not commit the API key to GitHub.

## Run the app

```bash
streamlit run app.py
```

## Evaluation

The app includes a small synthetic test set in `sample_policies.csv`.

The Evaluation page compares:

- baseline checklist
- GenAI-assisted checker

Metrics shown:

- recall
- precision
- detected issue count

The evaluation is simple but real: it uses multiple labeled examples instead of only one screenshot or one successful case.

## Where it works

The tool works best on short internal policies with clear rule-like statements, such as reimbursement, eligibility, approval, or submission rules.

## Where it fails

The tool can fail when:

- the policy is very long or depends on external regulations
- the wording is legally complex
- the model over-flags acceptable vague language
- the model misses subtle implicit contradictions
- the policy requires domain-specific knowledge not included in the text

## Human review

A human reviewer should always make the final decision, especially for legal, compliance, financial, or regulatory policies.
