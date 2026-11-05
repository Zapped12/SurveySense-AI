"""SurveySense AI: upload survey data and review possible answer quality issues."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import (
    CATEGORY_LABELS,
    DEFAULT_MODEL,
    MAX_RESPONSES,
    AnalysisError,
    analyze_responses,
    prepare_payload,
    validate_result,
)
from settings import get_setting, live_api_enabled

APP_FOLDER = Path(__file__).resolve().parent
SAMPLE_CONTEXT = (APP_FOLDER / "sample_context.txt").read_text(encoding="utf-8").strip()


def show_dashboard(result, mode, model):
    st.divider()
    st.subheader("Analysis dashboard", divider="blue")
    st.caption("Your review at a glance. Explore the evidence and decide what needs a follow-up.")
    if mode == "Demo (no API)":
        st.info("DEMO RESULTS: prepared examples for the fictional sample. No AI request was made.")
    else:
        st.caption(f"AI review using {model}")
    st.warning(
        "Flags are suggestions for human review. They do not establish that an answer "
        "is false. An answer without flags is not guaranteed to be correct."
    )
    reviewed = len(result.reviewed_response_ids)
    flagged = len({issue.response_id for issue in result.issues})
    metrics = st.columns(4, gap="medium")
    metrics[0].metric("Responses reviewed", reviewed, border=True,
                      help="Total responses included in this review.")
    metrics[1].metric("Flagged responses", flagged, border=True,
                      help="Unique responses with at least one issue to clarify.")
    metrics[2].metric("Issues to review", len(result.issues), border=True,
                      help="A response can have more than one type of issue.")
    metrics[3].metric("Without flags", reviewed - flagged, border=True,
                      help="Responses with no issues flagged; this does not guarantee correctness.")

    if not result.issues:
        st.success("No potential issues were flagged in the reviewed responses.")
    else:
        counts = pd.Series(
            {
                label: sum(issue.issue_category == category for issue in result.issues)
                for category, label in CATEGORY_LABELS.items()
            },
            name="Issues",
        )
        st.subheader("Issue breakdown")
        st.caption("Issues by category across the reviewed responses.")
        st.bar_chart(counts, horizontal=True, color="#2563EB", height=220)
        categories = st.multiselect(
            "Issue categories",
            options=list(CATEGORY_LABELS),
            default=list(CATEGORY_LABELS),
            format_func=CATEGORY_LABELS.get,
        )
        rows = []
        for issue in result.issues:
            if issue.issue_category in categories:
                rows.append(
                    {
                        "Response ID": issue.response_id,
                        "Issue category": CATEGORY_LABELS[issue.issue_category],
                        "Explanation": issue.explanation,
                        "Supporting evidence": "\n".join(
                            f"{item.column}: {item.quote if item.quote else '[blank]'}"
                            for item in issue.supporting_evidence
                        ),
                        "Suggested follow-up question": issue.suggested_follow_up_question,
                    }
                )
        st.subheader("Flagged responses")
        if rows:
            st.caption(
                f"Showing {len(rows)} issues across "
                f"{len({row['Response ID'] for row in rows})} responses. "
                "Each issue includes its evidence and a suggested follow-up."
            )
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True, row_height=80,
                column_config={
                    "Response ID": st.column_config.TextColumn(width="small"),
                    "Issue category": st.column_config.TextColumn(width="medium"),
                    "Explanation": st.column_config.TextColumn(width="large"),
                    "Supporting evidence": st.column_config.TextColumn(width="large"),
                    "Suggested follow-up question": st.column_config.TextColumn(width="large"),
                },
            )
        else:
            st.info("No issues match the selected categories.")
    with st.expander("View structured results (JSON)"):
        st.json(result.model_dump())


def main():
    st.set_page_config(page_title="SurveySense AI", page_icon=":material/analytics:", layout="wide")
    st.caption("SURVEY INTELLIGENCE  /  BUILT FOR HUMAN REVIEW")
    st.title("SurveySense AI")
    st.markdown("Make sense of feedback. **Know what to ask next.**")
    st.write("Upload your survey CSV, then review answers that may need clarification.")
    st.divider()
    st.subheader("Start a review", divider="blue")
    st.caption("Explore the fictional sample or bring your own survey data.")

    use_sample = st.checkbox("Use fictional sample survey")
    if use_sample:
        source = APP_FOLDER / "sample_survey.csv"
        st.caption("12 fictional responses about campus library use; no real respondent data.")
    else:
        source = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Use a comma-separated UTF-8 file with column names in the first row. Maximum 5 MB.",
        )

    if source is None:
        st.session_state.pop("analysis_result", None)
        st.info("Upload a CSV file to get started.")
        for column, title, description in zip(
            st.columns(3, gap="medium"),
            ("Potential contradictions", "Vague responses", "Incomplete answers"),
            (
                "Surface answers that appear to conflict within the same response.",
                "Check for specific details when the survey explicitly requires them.",
                "Find missing required information while respecting optional fields and valid skips.",
            ),
        ):
            with column.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(description)
        return
    if not use_sample and source.size > 5 * 1024 * 1024:
        st.session_state.pop("analysis_result", None)
        st.error("Use a CSV file of 5 MB or less.")
        return

    # Read the uploaded file directly into a pandas DataFrame (a table).
    try:
        # Keep IDs such as 001 and literal answers such as N/A unchanged.
        data = pd.read_csv(source, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        st.session_state.pop("analysis_result", None)
        st.error("This file is empty. Upload a CSV with column names and survey data.")
        return
    except UnicodeDecodeError:
        st.session_state.pop("analysis_result", None)
        st.error("Could not read the file's encoding. Save it as a UTF-8 CSV and try again.")
        return
    except pd.errors.ParserError:
        st.session_state.pop("analysis_result", None)
        st.error("Could not read this CSV. Check its commas, quotation marks, and row lengths.")
        return

    st.subheader("Survey data", divider="blue")
    st.caption(f"{len(data):,} rows · {len(data.columns):,} columns")
    if data.empty:
        st.warning("This CSV has column names but no data rows.")
    st.dataframe(data, width="stretch", hide_index=True)
    if data.empty:
        st.session_state.pop("analysis_result", None)
        return

    st.subheader("Review settings", divider="blue")
    st.caption("Define the questions and context so the review reflects what your survey actually asks.")
    columns = list(data.columns)
    default_id = next((column for column in columns if column.lower() in {"response_id", "id"}), None)
    id_column = st.selectbox(
        "Response ID column",
        options=[None] + columns,
        index=columns.index(default_id) + 1 if default_id is not None else 0,
        format_func=lambda column: "Generate row IDs (row-1, row-2, ...)" if column is None else column,
        help="The ID column identifies respondents and is excluded from answer analysis.",
    )
    context = st.text_area(
        "Survey questions and context",
        value=SAMPLE_CONTEXT if use_sample else "",
        key="sample_context" if use_sample else "upload_context",
        max_chars=4000,
        height=230,
        help="Map column names to full questions. Include scales, timeframes, required fields, and skip logic.",
    )
    if not context.strip():
        st.info("Add question context for a more useful review. Without it, the model is told to be conservative.")
    modes = ["Demo (no API)"]
    if live_api_enabled():
        modes.append("OpenAI API")
    else:
        st.caption("Live AI is disabled for this demonstration.")
    mode = st.radio("Analysis mode", modes, horizontal=True)
    model = DEFAULT_MODEL
    try:
        if mode not in modes:
            raise AnalysisError("The selected analysis mode is disabled on this server.")
        if mode == "OpenAI API":
            model = get_setting("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        payload = prepare_payload(data, id_column, context)
    except (AnalysisError, ValueError) as error:
        st.session_state.pop("analysis_result", None)
        st.error(str(error))
        return

    # Invalidate old results when data, context, IDs, mode, or model changes.
    signature = hashlib.sha256(
        json.dumps([payload, mode, model], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    saved = st.session_state.get("analysis_result")
    if saved and saved["signature"] != signature:
        st.session_state.pop("analysis_result", None)

    st.caption(
        f"Review scope: first {len(payload['responses'])} of {len(data):,} responses "
        f"(maximum {MAX_RESPONSES}). Rows after the first {MAX_RESPONSES} are not sent or analyzed."
    )
    demo_available = use_sample and id_column == "response_id" and context == SAMPLE_CONTEXT
    if mode == "Demo (no API)":
        st.info("Demo mode displays prepared sample flags. It does not analyze arbitrary uploads.")
        if not demo_available:
            st.info("For the demo, select the fictional sample with its original question context and response_id column.")
    else:
        st.caption(
            "Clicking Analyze Responses sends the question context, answer column names, selected IDs, "
            "and up to 20 responses to OpenAI. Remove identifying or sensitive data first."
        )

    if st.button(
        "Analyze Responses",
        type="primary",
        width="stretch",
        disabled=mode == "Demo (no API)" and not demo_available,
    ):
        st.session_state.pop("analysis_result", None)
        try:
            with st.spinner("Reviewing responses..."):
                if mode == "Demo (no API)":
                    raw_result = json.loads((APP_FOLDER / "demo_results.json").read_text(encoding="utf-8"))
                    result = validate_result(raw_result, payload)
                else:
                    api_key = get_setting("OPENAI_API_KEY")
                    result = analyze_responses(payload, api_key, model)
            st.session_state["analysis_result"] = {
                "signature": signature,
                "result": result.model_dump(),
            }
        except (AnalysisError, ValueError) as error:
            st.error(str(error))

    saved = st.session_state.get("analysis_result")
    if saved:
        show_dashboard(validate_result(saved["result"], payload), mode, model)


if __name__ == "__main__":
    main()
