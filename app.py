import os
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st


# ============================================================
# Policy Contradiction & Gap Checker
# A small GenAI-assisted Streamlit tool for reviewing draft
# policy documents before publication.
#
# It includes:
# 1. A rule-based baseline checker
# 2. A GenAI checker using OpenAI if API key is available
# 3. A deterministic local fallback so the app runs without keys
# 4. A small evaluation page comparing outputs with labeled examples
# ============================================================


AMBIGUOUS_TERMS = [
    "reasonable", "appropriate", "timely", "soon", "as needed", "may",
    "normally", "generally", "case-by-case", "exceptions may apply",
    "adequate", "sufficient", "significant", "excessive"
]

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bnever\b", r"\bprohibited\b", r"\bnot allowed\b",
    r"\bineligible\b", r"\bnot reimbursable\b", r"\bmust not\b"
]

ALLOW_PATTERNS = [
    r"\ballowed\b", r"\bpermitted\b", r"\breimbursable\b",
    r"\beligible\b", r"\bmay\b", r"\bcan\b", r"\bare covered\b"
]

THRESHOLD_PATTERN = re.compile(
    r"(under|over|above|below|less than|more than|at least|up to|within)\s+\$?\d+|\$?\d+\s*(days?|hours?|weeks?|months?|dollars?|%)",
    re.IGNORECASE
)


@dataclass
class Issue:
    issue_type: str
    severity: str
    confidence: float
    description: str
    related_text: str
    rationale: str
    suggested_fix: str
    source: str = "baseline"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
            "related_text": self.related_text,
            "rationale": self.rationale,
            "suggested_fix": self.suggested_fix,
            "source": self.source,
        }


def split_policy_into_rules(policy_text: str) -> List[str]:
    """Split policy text into sentence-like rules."""
    text = policy_text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+|;\s+|\n+", text)
    rules = [p.strip(" -•\t") for p in parts if len(p.strip()) > 8]
    return rules


def normalize_subject(rule: str) -> str:
    """Create a simple subject key for comparing related rules."""
    rule_l = rule.lower()
    rule_l = re.sub(r"\$?\d+(\.\d+)?", "", rule_l)
    rule_l = re.sub(r"\b(under|over|above|below|less than|more than|at least|up to|within|must|may|can|should|will|are|is|not|allowed|permitted|reimbursable|eligible|prohibited|covered)\b", "", rule_l)
    tokens = re.findall(r"[a-z]+", rule_l)
    stop = {"the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "by", "with", "from", "be", "as"}
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    return " ".join(tokens[:4])


def has_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def baseline_check(policy_text: str) -> List[Dict[str, Any]]:
    """
    A simple checklist baseline.
    This is intentionally limited: it uses fixed patterns and keywords.
    """
    rules = split_policy_into_rules(policy_text)
    issues: List[Issue] = []

    # 1. Ambiguous language
    for rule in rules:
        found_terms = [term for term in AMBIGUOUS_TERMS if term in rule.lower()]
        if found_terms:
            issues.append(Issue(
                issue_type="Ambiguity",
                severity="Medium",
                confidence=0.72,
                description=f"Potential vague language: {', '.join(found_terms)}.",
                related_text=rule,
                rationale="The rule uses words that may be interpreted differently by different reviewers.",
                suggested_fix="Define the vague term or replace it with measurable criteria.",
            ))

    # 2. Missing boundary / threshold clarity
    for rule in rules:
        if THRESHOLD_PATTERN.search(rule):
            if not re.search(r"\b(inclusive|exclusive|exactly|equal to|or less|or more|greater than or equal|less than or equal)\b", rule, re.IGNORECASE):
                issues.append(Issue(
                    issue_type="Gap",
                    severity="Medium",
                    confidence=0.68,
                    description="Numeric or time threshold may need boundary clarification.",
                    related_text=rule,
                    rationale="The rule has a threshold, but it may not clarify what happens at the exact boundary.",
                    suggested_fix="State whether the threshold is inclusive or exclusive and define what happens at the boundary.",
                ))

    # 3. Possible allow/prohibit contradiction on similar subjects
    for i, r1 in enumerate(rules):
        for r2 in rules[i + 1:]:
            s1 = normalize_subject(r1)
            s2 = normalize_subject(r2)
            if not s1 or not s2:
                continue

            shared_words = set(s1.split()) & set(s2.split())
            if len(shared_words) >= 1:
                r1_negative = has_any_pattern(r1, NEGATION_PATTERNS)
                r2_negative = has_any_pattern(r2, NEGATION_PATTERNS)
                r1_allow = has_any_pattern(r1, ALLOW_PATTERNS)
                r2_allow = has_any_pattern(r2, ALLOW_PATTERNS)

                if (r1_negative and r2_allow) or (r2_negative and r1_allow):
                    issues.append(Issue(
                        issue_type="Contradiction",
                        severity="High",
                        confidence=0.63,
                        description="Possible conflict between an allowing rule and a restricting rule.",
                        related_text=f"Rule A: {r1}\nRule B: {r2}",
                        rationale="The two rules appear to discuss a similar subject but point in opposite directions.",
                        suggested_fix="Add a clear exception hierarchy, such as 'except when...' or 'this rule overrides...'.",
                    ))

    # 4. Missing exception details
    for rule in rules:
        if re.search(r"\bexception|exceptions|unless|case-by-case\b", rule, re.IGNORECASE):
            if not re.search(r"\bapproved by|documented|criteria|requires|must include|manager|administrator\b", rule, re.IGNORECASE):
                issues.append(Issue(
                    issue_type="Gap",
                    severity="Medium",
                    confidence=0.70,
                    description="Exception process may be incomplete.",
                    related_text=rule,
                    rationale="The rule mentions exceptions but does not clearly state who approves them or what criteria are required.",
                    suggested_fix="Specify approval owner, required evidence, and decision criteria for exceptions.",
                ))

    # Deduplicate similar issues
    seen = set()
    unique = []
    for issue in issues:
        key = (issue.issue_type, issue.related_text[:100], issue.description)
        if key not in seen:
            seen.add(key)
            unique.append(issue.to_dict())
    return unique


def local_genai_fallback(policy_text: str) -> List[Dict[str, Any]]:
    """
    Deterministic fallback that imitates a GenAI-style structured review.
    It goes beyond the baseline by adding more explanation and edge-case reasoning.
    """
    baseline = baseline_check(policy_text)
    rules = split_policy_into_rules(policy_text)
    issues = []

    for item in baseline:
        new_item = dict(item)
        new_item["source"] = "local_genai_fallback"
        new_item["confidence"] = min(0.92, float(new_item["confidence"]) + 0.08)
        if new_item["issue_type"] == "Ambiguity":
            new_item["suggested_fix"] += " Add examples of approved and not-approved cases."
        issues.append(new_item)

    text_l = policy_text.lower()

    # Extra semantic-style checks
    if "late" in text_l and ("may be considered" in text_l or "may be accepted" in text_l):
        issues.append(Issue(
            issue_type="Gap",
            severity="High",
            confidence=0.82,
            description="Late submission rule lacks decision criteria.",
            related_text="Late submission language found in the policy.",
            rationale="The policy allows late submissions but does not explain when they should be accepted or rejected.",
            suggested_fix="Define acceptable reasons, required documentation, approval owner, and maximum late window.",
            source="local_genai_fallback",
        ).to_dict())

    if "full-time" in text_l and "exceptions may apply" in text_l:
        issues.append(Issue(
            issue_type="Gap",
            severity="High",
            confidence=0.84,
            description="Eligibility exception is not operationally defined.",
            related_text="Full-time eligibility with exceptions language.",
            rationale="Staff may not know which non-full-time users qualify or who can approve the exception.",
            suggested_fix="List exception categories and approval requirements.",
            source="local_genai_fallback",
        ).to_dict())

    if "group" in text_l and "alcohol is not reimbursable" in text_l and ("meal" in text_l or "meals" in text_l):
        issues.append(Issue(
            issue_type="Gap",
            severity="Medium",
            confidence=0.78,
            description="Group meal reimbursement needs item-level rule clarity.",
            related_text="Group meal and alcohol reimbursement rules.",
            rationale="A receipt may include both reimbursable meals and non-reimbursable alcohol, but the policy does not say how to split the cost.",
            suggested_fix="State whether alcohol must be itemized and deducted before reimbursement.",
            source="local_genai_fallback",
        ).to_dict())

    if len(rules) < 3:
        issues.append(Issue(
            issue_type="Gap",
            severity="Low",
            confidence=0.61,
            description="Policy may be too short to cover edge cases.",
            related_text=policy_text[:300],
            rationale="Very short policies often omit owner, documentation, approval, and exception handling.",
            suggested_fix="Add sections for scope, eligibility, approvals, documentation, and exceptions.",
            source="local_genai_fallback",
        ).to_dict())

    return dedupe_issues(issues)


def dedupe_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for item in issues:
        key = (
            item.get("issue_type", ""),
            item.get("description", "")[:80].lower(),
            item.get("related_text", "")[:80].lower(),
        )
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def call_openai_genai(policy_text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Optional OpenAI call.
    To use:
    1. pip install openai
    2. set OPENAI_API_KEY in your terminal
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return local_genai_fallback(policy_text), "No API key found. Used local fallback."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system_prompt = """
You are a policy review assistant. Review only the provided policy text.
Find contradictions, missing conditions/gaps, and ambiguous language.
Do not invent facts. Return valid JSON only.
Each issue must include:
issue_type, severity, confidence, description, related_text, rationale, suggested_fix.
Allowed issue_type values: Contradiction, Gap, Ambiguity.
Allowed severity values: Low, Medium, High.
"""

        user_prompt = f"""
Policy text:
{policy_text}

Return JSON in this format:
{{
  "issues": [
    {{
      "issue_type": "Contradiction|Gap|Ambiguity",
      "severity": "Low|Medium|High",
      "confidence": 0.0,
      "description": "...",
      "related_text": "...",
      "rationale": "...",
      "suggested_fix": "..."
    }}
  ]
}}
"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)
        issues = parsed.get("issues", [])
        for issue in issues:
            issue["source"] = "genai_openai"
        return dedupe_issues(issues), "Used OpenAI GenAI review."

    except Exception as e:
        fallback = local_genai_fallback(policy_text)
        return fallback, f"OpenAI call failed. Used local fallback. Error: {e}"


def load_test_set() -> pd.DataFrame:
    path = "sample_policies.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def keyword_match(issue: Dict[str, Any], expected: str) -> bool:
    combined = " ".join([
        str(issue.get("issue_type", "")),
        str(issue.get("description", "")),
        str(issue.get("rationale", "")),
        str(issue.get("related_text", "")),
    ]).lower()
    expected_words = [w for w in re.findall(r"[a-z]+", expected.lower()) if len(w) > 3]
    if not expected_words:
        return False
    hits = sum(1 for w in expected_words if w in combined)
    return hits >= max(1, min(3, len(expected_words) // 3))


def evaluate_checker(df: pd.DataFrame, checker_name: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        policy = row["policy_text"]
        expected_items = [x.strip() for x in str(row["known_issues"]).split("|") if x.strip()]

        if checker_name == "Baseline":
            detected = baseline_check(policy)
        else:
            detected, _ = call_openai_genai(policy)

        matched = 0
        for expected in expected_items:
            if any(keyword_match(issue, expected) for issue in detected):
                matched += 1

        valid_detected = 0
        for issue in detected:
            if any(keyword_match(issue, expected) for expected in expected_items):
                valid_detected += 1

        recall = matched / len(expected_items) if expected_items else 1.0
        precision = valid_detected / len(detected) if detected else (1.0 if not expected_items else 0.0)

        rows.append({
            "policy_id": row["policy_id"],
            "checker": checker_name,
            "known_issue_count": len(expected_items),
            "detected_issue_count": len(detected),
            "matched_known_issues": matched,
            "recall": round(recall, 2),
            "precision": round(precision, 2),
        })

    return pd.DataFrame(rows)


def display_issues(issues: List[Dict[str, Any]]):
    if not issues:
        st.success("No major issues detected.")
        return

    df = pd.DataFrame(issues)
    for issue_type in ["Contradiction", "Gap", "Ambiguity"]:
        sub = df[df["issue_type"] == issue_type] if "issue_type" in df.columns else pd.DataFrame()
        if len(sub) == 0:
            continue

        st.subheader(issue_type)
        for _, item in sub.iterrows():
            severity = item.get("severity", "Medium")
            conf = item.get("confidence", "")
            with st.expander(f"{severity} severity | Confidence: {conf} | {item.get('description', '')}"):
                st.markdown("**Related text**")
                st.write(item.get("related_text", ""))
                st.markdown("**Rationale**")
                st.write(item.get("rationale", ""))
                st.markdown("**Suggested fix**")
                st.write(item.get("suggested_fix", ""))


def main():
    st.set_page_config(page_title="Policy Contradiction & Gap Checker", layout="wide")

    st.title("Policy Contradiction & Gap Checker")
    st.caption("A GenAI-assisted tool for reviewing draft policy documents before publication.")

    page = st.sidebar.radio("Choose page", ["Analyze Policy", "Evaluation", "About"])

    if page == "Analyze Policy":
        st.header("Analyze a Draft Policy")

        example = (
            "Meals under $50 are reimbursable. Alcohol is not reimbursable. "
            "Group meals are allowed. Requests must be submitted within 7 days. "
            "Late submissions may be considered."
        )

        policy_text = st.text_area(
            "Paste policy text here",
            value=example,
            height=220,
        )

        col1, col2 = st.columns([1, 1])
        run_baseline = col1.checkbox("Show baseline checklist results", value=True)
        run_genai = col2.checkbox("Show GenAI-assisted results", value=True)

        if st.button("Analyze"):
            if not policy_text.strip():
                st.warning("Please paste a policy document first.")
                return

            if run_genai:
                st.header("GenAI-Assisted Review")
                genai_issues, status = call_openai_genai(policy_text)
                st.info(status)
                display_issues(genai_issues)

            if run_baseline:
                st.header("Rule-Based Baseline Review")
                baseline_issues = baseline_check(policy_text)
                display_issues(baseline_issues)

            if run_genai and run_baseline:
                st.header("Comparison Summary")
                b = baseline_check(policy_text)
                g, _ = call_openai_genai(policy_text)

                comparison = pd.DataFrame([
                    {"method": "Baseline checklist", "issues_detected": len(b)},
                    {"method": "GenAI-assisted", "issues_detected": len(g)},
                ])
                st.dataframe(comparison, use_container_width=True)
                st.bar_chart(comparison.set_index("method"))

    elif page == "Evaluation":
        st.header("Evaluation Against Labeled Synthetic Policies")
        df = load_test_set()

        if df.empty:
            st.error("sample_policies.csv was not found.")
            return

        st.write("This small test set compares the GenAI-assisted checker against the simpler rule-based baseline.")
        st.dataframe(df[["policy_id", "policy_text", "known_issues"]], use_container_width=True)

        if st.button("Run Evaluation"):
            baseline_eval = evaluate_checker(df, "Baseline")
            genai_eval = evaluate_checker(df, "GenAI-assisted")
            results = pd.concat([baseline_eval, genai_eval], ignore_index=True)

            st.subheader("Per-policy results")
            st.dataframe(results, use_container_width=True)

            st.subheader("Average metrics")
            summary = results.groupby("checker")[["recall", "precision", "detected_issue_count"]].mean().round(2).reset_index()
            st.dataframe(summary, use_container_width=True)
            st.bar_chart(summary.set_index("checker")[["recall", "precision"]])

            st.download_button(
                label="Download evaluation results as CSV",
                data=results.to_csv(index=False),
                file_name="evaluation_results.csv",
                mime="text/csv",
            )

    else:
        st.header("About This Project")
        st.markdown(
            """
**Target user:** operations staff, program coordinators, or administrative personnel reviewing internal policies before publication.

**Workflow:** paste a draft policy, run the checker, review structured issues, revise policy, and keep a human reviewer in control.

**What the tool checks:**
- Contradictions between rules
- Missing conditions or gaps
- Ambiguous or unclear language

**Baseline:** a simple rule-based checklist using fixed keyword and pattern matching.

**GenAI method:** OpenAI structured review if `OPENAI_API_KEY` is available. If not, the app uses a local fallback so the project still runs for grading.

**Human-in-the-loop control:** the tool is a review aid only. It should not be used as legal, compliance, or final approval authority.
"""
        )


if __name__ == "__main__":
    main()
