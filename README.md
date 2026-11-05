# SurveySense AI

A beginner-friendly Streamlit app that uploads survey CSVs and flags answers that may need clarification. It reviews potential contradictions, vague responses, and incomplete answers. A flag is a prompt for human follow-up, not a claim that a respondent is lying or wrong.

## Run locally

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `.venv` already exists, skip the first command. Open the local URL printed by Streamlit, usually http://localhost:8501.

## Try it without an API key

1. Select **Use fictional sample survey**.
2. Keep the default `response_id` column and question context.
3. Keep **Demo (no API)** selected and click **Analyze Responses**.

The dashboard shows **12 reviewed responses, 6 flagged responses, and 6 issues**: two potential contradictions, two vague responses, and two incomplete answers. You can filter categories and expand the structured JSON.

Demo results are hand-prepared examples from `demo_results.json`, not model output or an offline AI system. Demo mode works only with the built-in sample and its original settings. It makes no API calls, even if credentials exist. Real model results may differ.

The fictional campus-library survey includes ordinary responses, mismatched visit counts, vague descriptions, missing required answers, optional blank answers, and valid skipped questions. For example, S002 says they did not visit but reports six visits; S005 did not visit and correctly skips experience questions. `sample_context.txt` documents the full questions and skip rules.

## Public GitHub repository and Streamlit demonstration

Live AI is **disabled by default**. With `ENABLE_LIVE_API` absent or false, the interface only offers **Demo (no API)**. The server also rejects direct live-analysis calls before creating an OpenAI client, even if a key exists or someone manipulates the selected mode. The fictional demo, CSV preview, dashboard, and filters remain available; prepared demo flags apply only to the original fictional sample.

For a public Streamlit demonstration, deploy `app.py` with `requirements.txt`. Leave `ENABLE_LIVE_API` unset or set it to `false` in the host's server environment or Streamlit secrets, and omit `OPENAI_API_KEY`. No credentials are needed for the demonstration. Visitors cannot enable live AI through the interface or uploaded data. The app has no authentication or per-user API budget, so reserve live API access for a private local instance.

Only the credential-free `.streamlit/secrets.toml.example` belongs in the public repository. `.gitignore` excludes local `secrets*.toml` files (including `.streamlit/secrets.toml`), `.env` files, private `.key`/`.pem` files, and `.aws/`. Keep actual credentials out of source code, survey files, screenshots, and commits. Do not force-add ignored secret files.

Once the folder is a Git repository, check the exclusions and staged paths before publishing:

```powershell
git check-ignore .streamlit/secrets.toml .env
git ls-files -- ":(glob)**/secrets*.toml" ":(glob)**/.env" ":(glob)**/.env.*"
git diff --cached --name-only
```

The first command should list both ignored paths. The second should list no secret files; a credential-free `.env.example` is allowed. Ignore rules do not untrack files already committed. If a secret was previously tracked, remove it from Git's index while keeping the local file, and revoke/replace any exposed key before publishing; earlier commits may still contain it.

## Enable live AI locally

An OpenAI API account with suitable model access and quota is needed. Live requests may incur usage charges. Keep the public demonstration disabled and configure a separate local instance.

1. If you do not already have `.streamlit/secrets.toml`, copy `.streamlit/secrets.toml.example` to that path. Preserve an existing private file.
2. In the private file, set `ENABLE_LIVE_API = true` and fill the empty `OPENAI_API_KEY` value with your own key. Never edit the public example to contain a credential.
3. Restart Streamlit. **OpenAI API** now appears in **Analysis mode**. Select it, review the data/context, and click **Analyze Responses**. A missing key produces a setup message instead of a request.
4. To disable live access again, set `ENABLE_LIVE_API = false` and restart the app.

Settings can also come from the server process environment. Only boolean `true` in Streamlit secrets or the text `true` (case-insensitive) enables live access; missing, false, or unrecognized values disable it. Nonempty environment values take precedence over Streamlit secrets, so an environment value of `false` also overrides a secret value of `true`. Restart after changing configuration. `.env` files are ignored by Git but are **not automatically loaded**.

For an environment-only local session, this PowerShell example prompts for the key without echoing it or putting its literal value in shell history:

```powershell
$env:ENABLE_LIVE_API = "true"
$surveyApiKey = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $surveyApiKey).Password
.\.venv\Scripts\python.exe -m streamlit run app.py
# After stopping Streamlit:
Remove-Item Env:OPENAI_API_KEY
Remove-Item Env:ENABLE_LIVE_API
```

The optional `OPENAI_MODEL` setting defaults to `gpt-4.1-mini`. Set it in private secrets or the environment to another model that supports the Responses API and Structured Outputs. Model availability depends on your account. Credentials stay on the server.

## Use your own CSV

Upload a UTF-8, comma-separated file with column names in the first row, up to 5 MB. The existing interactive data table remains available.

- Choose the response ID column. IDs must be unique and nonblank in the analyzed rows. Generated IDs (`row-1`, `row-2`, etc.) refer to the first data row as row 1, excluding the CSV header. The selected ID column is excluded from answer analysis.
- Map column names to full survey questions in **Survey questions and context**. Include scales, timeframes, required/optional fields, and skip logic. For example: `Q1: visits last month; Q2: satisfaction, 1–5, required only if Q1 > 0; Q3: optional suggestion`.
- In a local instance with live AI enabled, select **OpenAI API** and click **Analyze Responses**. Without context, the model is instructed to interpret cautiously rather than invent question meanings. The public demonstration can preview uploads but only analyzes the built-in fictional sample using prepared results.

Only the **first 20 responses** are sent for analysis, even if the file contains more. This is a bounded preview, not a representative sample or an analysis of the entire CSV. The app also limits input to 30 answer columns, 2,000 characters per answer, 4,000 context characters, and 60,000 characters in the serialized JSON input. Oversized input is rejected rather than silently shortened.

CSV values are read as text so IDs like `001`, zeroes, and literal `N/A` answers are preserved. Empty cells stay empty. An ID column should contain anonymous identifiers; this app does not automatically anonymize other columns.

## How the code works

| File | Purpose |
| --- | --- |
| `app.py` | Existing uploader/table, review settings, Analyze button, session results, and dashboard. |
| `analysis.py` | Data limits, result schema, model instructions, API request, and validation. |
| `settings.py` | Shared server-side settings and the default-off live API policy. |
| `.streamlit/secrets.toml.example` | Public configuration example with live access disabled and no credential. |
| `sample_survey.csv` | 12 fictional library-survey responses. |
| `sample_context.txt` | Questions, scales, and skip rules for that sample. |
| `demo_results.json` | Prepared review examples for testing the interface without an API call. |
| `tests/` | Automated validation, mocked API, and Streamlit interface checks. |

The data flow is:

```text
CSV -> pandas table -> first 20 rows + question context
    -> Responses API (or prepared demo results)
    -> schema and evidence validation -> dashboard
```

### The AI integration

`prepare_payload()` creates a JSON object containing the question context, original answer column names, and rows with response IDs and answers. Grouping a row's answers together lets the model compare answers from the same respondent while understanding what each column asks.

`analyze_responses()` checks the server's `ENABLE_LIVE_API` policy on every call and raises `AnalysisError` when it is disabled. When enabled, it creates an OpenAI client and calls `client.responses.parse()`. Its `instructions` parameter supplies the review rules, and its user input contains the survey JSON. `text_format=AnalysisResult` supplies a Pydantic schema: Python classes defining the required result fields, allowed issue categories, and field types. The SDK uses that schema to request structured output and parse the response into Python objects. See the official [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).

The schema includes `reviewed_response_ids` for coverage and an `issues` list. Each issue contains:

- `response_id`
- `issue_category`: `potential_contradiction`, `vague_response`, or `incomplete_answer`
- `explanation`
- `supporting_evidence`: a list of original column names and verbatim answer excerpts
- `suggested_follow_up_question`

`validate_result()` then checks that every submitted ID is covered exactly once, issues reference known IDs and columns, evidence quotes occur in the original cells, and a potential contradiction includes two distinct excerpts. An empty quote is accepted only for an actually blank cell. Duplicate categories for one response are rejected. Extra fields and unknown categories fail schema validation.

The app checks for incomplete or declined output and handles key, quota/rate-limit, connection, timeout, model-access, and validation errors with readable messages. It does not display raw provider errors containing request details. There is one request per button click, a 60-second timeout, a 6,000-output-token limit, and no automatic retries. A large or difficult review can still exceed the output limit; it is rejected instead of displayed as complete.

### The dashboard and Streamlit state

The dashboard counts reviewed rows, unique flagged rows, total issues, and rows without flags. One row can have multiple issue categories, so the issue count can exceed the flagged-row count. The category chart and filter help browse the evidence and follow-up questions.

Streamlit reruns the script when widgets change. Results live in `st.session_state` so filtering the dashboard does not make another paid request. A hash of the submitted data, context, mode, and model identifies which inputs produced the result; changing these inputs clears stale results. Survey data and results are not cached globally or written to disk by the app.

## Decisions to explain in an interview

| Decision | What you can justify |
| --- | --- |
| Use a model for contextual review | Whether two answers conflict depends on question wording, timeframe, and skip logic. Simple keyword matching misses that context. A production version could combine deterministic missing-field checks with model review. |
| Keep flags tentative | Differences may come from interpretation or missing context. The tool supports human clarification and does not score honesty or automatically remove responses. |
| Supply question context | A missing rating can be a valid skip for a non-user. A short answer can fully answer a question. Column names alone may be ambiguous. |
| Use structured output and validation | A predictable schema supports the dashboard. Local checks reject invented IDs, categories, and evidence; valid JSON alone is insufficient. |
| Limit to 20 rows and bounded text | Keeps a student prototype's cost, latency, and request size manageable. The first 20 rows cannot establish quality for the full dataset. |
| Separate UI and analysis | `app.py` handles interaction; `analysis.py` handles preparation and model review. Each has a clear role and can be tested independently. |
| Provide a labeled demo | You can demonstrate the complete interface without credentials, network access, or spending money. Prepared examples do not evaluate real model quality. |
| Treat survey text as untrusted | Respondents can include instructions such as “ignore the rules.” Headers, context, and answers stay in user data, never in trusted instructions. The model receives no tools or code-execution access. |

Prompt instructions, structured output, and exact evidence checks reduce risk; they cannot guarantee resistance to prompt injection or correct interpretation. A model can quote real text and still misinterpret it, miss an issue, or produce a false positive. Before real use, evaluate model results against human-reviewed examples, including valid skips and ordinary answers. Useful measures are precision (how many flags were useful) and recall (how many known issues were found).

## Data handling

Demo mode keeps data local. Live mode sends the question context, answer headers, selected IDs, and up to 20 rows to OpenAI. The code does not log payloads, credentials, or raw API errors. It uses `store=False` to disable response storage for later API retrieval; that setting does not promise zero retention across all provider systems. Review the provider's [data controls](https://developers.openai.com/api/docs/guides/your-data) before using sensitive data. Remove identifying data before live analysis.

This is a learning prototype with a public demonstration mode. It has no authentication, database, or automatic anonymization. Streamlit configuration follows its [secrets documentation](https://docs.streamlit.io/develop/api-reference/connections/st.secrets).

## Run the checks without API calls

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The default test run uses mocked OpenAI calls and Streamlit's testing framework, needs no credentials, and makes no API requests. Tests cover the default-off policy, disabled direct calls with a key present, manipulated modes, hidden API options, enabled local access, Git secret exclusions, and the existing analysis/demo behavior. API contract tests enable the policy only inside their isolated test environment.

Model-quality regression tests are skipped unless `SURVEYSENSE_RUN_MODEL_TESTS=1` is explicitly set. Those tests temporarily enable the policy in the test process and use `OPENAI_API_KEY` from the environment, making paid requests. Leave that test flag unset for public demonstrations and ordinary offline checks. Mocked tests do not establish the accuracy of real model judgments.
