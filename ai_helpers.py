from shared_core import *

# =========================================================
# GEMINI / AI HELPER FUNCTIONS
# =========================================================
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None, "Gemini API key was not found in the backend environment. Set GEMINI_API_KEY or GOOGLE_API_KEY in your terminal before running Streamlit."

    if genai is None:
        return None, "Gemini SDK is not installed. Run: pip install google-genai"

    try:
        return genai.Client(api_key=api_key), None
    except Exception as e:
        return None, f"Could not initialize Gemini client: {e}"


def extract_json_from_text(text):
    if not text:
        return None

    text = text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def build_ai_dataset_context(df, max_cols=40):
    cols_info = []
    for col in df.columns[:max_cols]:
        series = df[col]
        sample_values = series.dropna().astype(str).head(5).tolist()
        cols_info.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 2),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": sample_values
        })

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_info": cols_info
    }


def normalize_name(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def resolve_column_name(name, columns):
    if name is None:
        return None

    name_str = str(name).strip()
    if name_str in columns:
        return name_str

    norm_target = normalize_name(name_str)

    exact_norm = {normalize_name(col): col for col in columns}
    if norm_target in exact_norm:
        return exact_norm[norm_target]

    for col in columns:
        col_norm = normalize_name(col)
        if norm_target and (norm_target in col_norm or col_norm in norm_target):
            return col

    return None


def resolve_many_column_names(names, columns):
    resolved = []
    for name in names or []:
        col = resolve_column_name(name, columns)
        if col and col not in resolved:
            resolved.append(col)
    return resolved


def infer_local_cleaning_plan(command, df):
    text = (command or "").strip().lower()
    columns = df.columns.tolist()
    operations = []

    if not text:
        return {"assistant_note": "No command was provided.", "operations": []}

    def mentioned_columns():
        matched = []
        for col in columns:
            if str(col).lower() in text or normalize_name(col) in normalize_name(text):
                matched.append(col)
        return matched

    matched_cols = mentioned_columns()

    if "duplicate" in text:
        operations.append({
            "operation": "remove_duplicates",
            "duplicate_type": "full_row",
            "subset_columns": [],
            "keep": "first"
        })

    if any(word in text for word in ["null", "missing", "na", "blank"]):
        method = None
        if "median" in text:
            method = "median"
        elif "mean" in text or "average" in text:
            method = "mean"
        elif "mode" in text or "most frequent" in text:
            method = "mode"
        elif "forward fill" in text:
            method = "forward_fill"
        elif "backward fill" in text or "back fill" in text:
            method = "backward_fill"
        elif "constant" in text:
            method = "constant"

        target_cols = matched_cols[:]
        if not target_cols:
            target_cols = columns[:]

        for col in target_cols:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            chosen_method = method
            if chosen_method is None:
                chosen_method = "median" if is_num else "most_frequent"
            operations.append({
                "operation": "fill_missing",
                "column": col,
                "method": chosen_method
            })

    if "standardize" in text and "casing" in text or "lowercase" in text or "title case" in text or "trim spaces" in text:
        mode = "trim"
        if "lower" in text:
            mode = "lower"
        elif "title" in text or "proper" in text:
            mode = "title"

        for col in matched_cols:
            if str(df[col].dtype) in ["object", "string", "category", "bool"]:
                operations.append({
                    "operation": "standardize_text",
                    "column": col,
                    "mode": mode
                })

    rename_matches = re.findall(r"rename\s+([a-zA-Z0-9_ ]+)\s+to\s+([a-zA-Z0-9_ ]+)", text)
    if rename_matches:
        mapping = {}
        for old_name, new_name in rename_matches:
            old_resolved = resolve_column_name(old_name.strip(), columns)
            if old_resolved:
                mapping[old_resolved] = new_name.strip().replace(" ", "_")
        if mapping:
            operations.append({
                "operation": "rename_columns",
                "mapping": mapping
            })

    if "drop column" in text or "remove column" in text:
        cols_to_drop = matched_cols
        if cols_to_drop:
            operations.append({
                "operation": "drop_columns",
                "columns": cols_to_drop
            })

    if "scale" in text or "normalize" in text or "standardize" in text:
        method = None
        if "min-max" in text or "min max" in text or "normalize" in text:
            method = "minmax"
        elif "z-score" in text or "z score" in text or "standardize" in text:
            method = "zscore"

        candidate_cols = [col for col in matched_cols if pd.api.types.is_numeric_dtype(df[col])]
        if candidate_cols and method:
            operations.append({
                "operation": "scale_columns",
                "columns": candidate_cols,
                "method": method
            })

    return {
        "assistant_note": "This is a locally inferred plan. If Gemini is available, it can provide a richer plan.",
        "operations": operations
    }


def generate_ai_cleaning_plan(command, df):
    local_plan = infer_local_cleaning_plan(command, df)
    client, client_error = get_gemini_client()

    if client is None:
        return local_plan, client_error

    schema_example = {
        "assistant_note": "brief explanation",
        "operations": [
            {"operation": "fill_missing", "column": "price", "method": "median"},
            {"operation": "remove_duplicates", "duplicate_type": "full_row", "subset_columns": [], "keep": "first"},
            {"operation": "standardize_text", "column": "category", "mode": "title"},
            {"operation": "drop_columns", "columns": ["unneeded_col"]},
            {"operation": "rename_columns", "mapping": {"Old Name": "new_name"}},
            {"operation": "scale_columns", "columns": ["sales"], "method": "zscore"},
            {"operation": "drop_rows_missing", "columns": ["id"]},
            {"operation": "drop_columns_missing_threshold", "threshold_percent": 60},
            {"operation": "convert_type", "column": "date", "target_type": "datetime", "datetime_format": ""},
            {"operation": "map_replace", "column": "status", "mapping": {"m": "Male", "f": "Female"}, "set_unmatched_to_other": False},
            {"operation": "create_formula_column", "new_column": "margin_ratio", "formula_type": "colA / colB", "col_a": "profit", "col_b": "revenue"},
            {"operation": "create_binned_column", "source_column": "age", "new_column": "age_band", "method": "Equal-width", "bins": 4},
            {"operation": "validation_rule", "rule_type": "Numeric range check", "column": "age", "min_value": 0, "max_value": 100}
        ]
    }

    prompt = f"""
You are helping a Streamlit data cleaning app.
Return ONLY valid JSON.

The user's command is:
{command}

Dataset context:
{json.dumps(build_ai_dataset_context(df), ensure_ascii=False, indent=2)}

Available operation names:
- fill_missing
- remove_duplicates
- standardize_text
- drop_columns
- rename_columns
- scale_columns
- drop_rows_missing
- drop_columns_missing_threshold
- convert_type
- map_replace
- create_formula_column
- create_binned_column
- validation_rule

Rules:
1. Use exact dataset column names whenever possible.
2. Only include operations that are clearly useful for the user's command.
3. For fill_missing methods use one of: constant, mean, median, mode, most_frequent, forward_fill, backward_fill.
4. For scale_columns methods use one of: minmax, zscore.
5. For standardize_text modes use one of: trim, lower, title.
6. For validation_rule rule_type use one of: Numeric range check, Allowed categories list, Non-null constraint.
7. If the user asks to clean broadly, include sensible steps such as duplicate removal when mentioned.
8. If something is unclear, make a best reasonable assumption and mention it in assistant_note.

Return JSON in this shape:
{json.dumps(schema_example, ensure_ascii=False, indent=2)}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        response_text = getattr(response, "text", "") or ""
        parsed = extract_json_from_text(response_text)
        if isinstance(parsed, dict) and "operations" in parsed:
            return parsed, "Gemini plan generated successfully."
        return local_plan, "Gemini returned an invalid format, so a local fallback plan was used."
    except Exception as e:
        return local_plan, f"Gemini request failed, so a local fallback plan was used: {e}"


def generate_ai_chart_suggestions(user_prompt, df):
    local_suggestions = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_candidates = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_candidates.append(col)
        else:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.95:
                datetime_candidates.append(col)

    if numeric_cols:
        local_suggestions.append(f"Histogram: good for checking the distribution of {numeric_cols[0]}.")
    if len(numeric_cols) >= 2:
        local_suggestions.append(f"Scatter Plot: useful for seeing the relationship between {numeric_cols[0]} and {numeric_cols[1]}.")
        local_suggestions.append("Heatmap / Correlation Matrix: useful for comparing numeric columns together.")
    if categorical_cols and numeric_cols:
        local_suggestions.append(f"Bar Chart: compare {numeric_cols[0]} across categories in {categorical_cols[0]}.")
        local_suggestions.append(f"Box Plot: compare the spread of {numeric_cols[0]} across groups in {categorical_cols[0]}.")
    if datetime_candidates and numeric_cols:
        local_suggestions.append(f"Line Chart: show how {numeric_cols[0]} changes over time using {datetime_candidates[0]}.")

    local_text = "\n".join([f"- {item}" for item in local_suggestions]) if local_suggestions else "- Upload or prepare more columns to get chart suggestions."

    client, client_error = get_gemini_client()
    if client is None:
        return f"AI is unavailable right now.\n\nLocal suggestions:\n{local_text}\n\n{client_error}"

    prompt = f"""
You are helping a data visualization builder.
Give concise, practical chart recommendations.

User request:
{user_prompt}

Dataset context:
{json.dumps(build_ai_dataset_context(df), ensure_ascii=False, indent=2)}

Available chart types in this app:
- Histogram
- Box Plot
- Scatter Plot
- Line Chart
- Bar Chart
- Heatmap / Correlation Matrix

Write:
1. 4 to 6 recommended charts.
2. For each one, say exactly which columns would work best.
3. Keep it simple and practical.
4. Mention when a chart is not a good fit.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        response_text = getattr(response, "text", "") or ""
        if response_text.strip():
            return response_text.strip()
        return f"Local suggestions:\n{local_text}"
    except Exception as e:
        return f"Gemini request failed.\n\nLocal suggestions:\n{local_text}\n\nError: {e}"


