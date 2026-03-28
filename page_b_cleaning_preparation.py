from shared_core import *
from ai_helpers import *


def render_page_b():
    st.title("Cleaning & Preparation Studio")

    if st.session_state.working_df is None:
        st.warning("Please upload a dataset first on Page A.")
    else:
        df = st.session_state.working_df.copy()
        render_workflow_controls(show_metrics=False)

        st.write("### AI Assistant")
        enable_ai_assistant_b = st.toggle(
            "Enable AI assistant",
            value=False,
            key="enable_ai_assistant_page_b"
        )
        st.caption("AI outputs may be imperfect. Please review the suggested cleaning plan before applying it.")

        if enable_ai_assistant_b:
            ai_cleaning_command = st.text_area(
                "Describe what you want to clean",
                placeholder="Example: replace nulls in price with median, standardize category casing, remove duplicates, rename order date to order_date, scale sales with z-score, and validate age between 0 and 100",
                key="ai_cleaning_command"
            )

            ai_col1, ai_col2 = st.columns(2)
            with ai_col1:
                if st.button("Suggest AI cleaning plan", use_container_width=True, key="suggest_ai_cleaning_plan_btn"):
                    plan, feedback = generate_ai_cleaning_plan(ai_cleaning_command, df)
                    st.session_state.ai_cleaning_plan = plan
                    st.session_state.ai_cleaning_feedback = feedback

            with ai_col2:
                if st.button("Clear AI plan", use_container_width=True, key="clear_ai_cleaning_plan_btn"):
                    st.session_state.ai_cleaning_plan = None
                    st.session_state.ai_cleaning_feedback = ""

            def apply_ai_operation(current_df, op):
                op_type = str(op.get("operation", "")).strip()
                after_df = current_df.copy()
                affected_columns = []
                params = dict(op)
                history_required = True
                info_message = ""

                if op_type == "fill_missing":
                    col = resolve_column_name(op.get("column"), after_df.columns.tolist())
                    if not col:
                        return current_df, None, None, None, False, "Skipped fill_missing because the column was not found."
                    method = str(op.get("method", "")).strip().lower()
                    affected_columns = [col]
                    if method == "constant":
                        after_df[col] = after_df[col].fillna(op.get("constant_value"))
                    elif method == "mean":
                        x = pd.to_numeric(after_df[col], errors="coerce")
                        after_df[col] = x.fillna(x.mean())
                    elif method == "median":
                        x = pd.to_numeric(after_df[col], errors="coerce")
                        after_df[col] = x.fillna(x.median())
                    elif method in ["mode", "most_frequent"]:
                        mode_val = after_df[col].mode(dropna=True)
                        if not mode_val.empty:
                            after_df[col] = after_df[col].fillna(mode_val.iloc[0])
                    elif method == "forward_fill":
                        after_df[col] = after_df[col].ffill()
                    elif method == "backward_fill":
                        after_df[col] = after_df[col].bfill()
                    else:
                        return current_df, None, None, None, False, f"Skipped fill_missing for {col} because the method was not supported."
                    params = {"column": col, "method": method}
                    return after_df, "ai_fill_missing", params, affected_columns, history_required, info_message

                if op_type == "remove_duplicates":
                    duplicate_type = str(op.get("duplicate_type", "full_row"))
                    keep = str(op.get("keep", "first"))
                    subset_columns = resolve_many_column_names(op.get("subset_columns", []), after_df.columns.tolist())
                    before_rows = len(after_df)
                    if duplicate_type == "subset" and subset_columns:
                        after_df = after_df.drop_duplicates(subset=subset_columns, keep=keep)
                        affected_columns = subset_columns
                    else:
                        after_df = after_df.drop_duplicates(keep=keep)
                        affected_columns = after_df.columns.tolist()
                    params = {
                        "duplicate_type": duplicate_type,
                        "subset_columns": subset_columns,
                        "keep": keep,
                        "rows_removed": int(before_rows - len(after_df))
                    }
                    return after_df, "ai_remove_duplicates", params, affected_columns, history_required, info_message

                if op_type == "standardize_text":
                    col = resolve_column_name(op.get("column"), after_df.columns.tolist())
                    if not col:
                        return current_df, None, None, None, False, "Skipped standardize_text because the column was not found."
                    mode = str(op.get("mode", "trim")).lower()
                    s = after_df[col].astype("string")
                    if mode == "trim":
                        after_df[col] = s.str.strip()
                    elif mode == "lower":
                        after_df[col] = s.str.strip().str.lower()
                    elif mode == "title":
                        after_df[col] = s.str.strip().str.title()
                    else:
                        return current_df, None, None, None, False, f"Skipped standardize_text for {col} because the mode was not supported."
                    return after_df, "ai_standardize_text", {"column": col, "mode": mode}, [col], history_required, info_message

                if op_type == "drop_columns":
                    cols = resolve_many_column_names(op.get("columns", []), after_df.columns.tolist())
                    if not cols:
                        return current_df, None, None, None, False, "Skipped drop_columns because no valid columns were found."
                    after_df = after_df.drop(columns=cols)
                    return after_df, "ai_drop_columns", {"columns_dropped": cols}, cols, history_required, info_message

                if op_type == "rename_columns":
                    mapping = op.get("mapping", {}) or {}
                    resolved_mapping = {}
                    for old_name, new_name in mapping.items():
                        old_col = resolve_column_name(old_name, after_df.columns.tolist())
                        if old_col and str(new_name).strip():
                            resolved_mapping[old_col] = str(new_name).strip()
                    if not resolved_mapping:
                        return current_df, None, None, None, False, "Skipped rename_columns because no valid mapping was found."
                    after_df = after_df.rename(columns=resolved_mapping)
                    return after_df, "ai_rename_columns", {"mapping": resolved_mapping}, list(resolved_mapping.keys()), history_required, info_message

                if op_type == "scale_columns":
                    cols = resolve_many_column_names(op.get("columns", []), after_df.columns.tolist())
                    method = str(op.get("method", "")).lower()
                    if not cols:
                        return current_df, None, None, None, False, "Skipped scale_columns because no valid columns were found."
                    for col in cols:
                        if method == "minmax":
                            s = pd.to_numeric(after_df[col], errors="coerce")
                            valid = s.dropna()
                            if not valid.empty and valid.max() != valid.min():
                                after_df[col] = (s - valid.min()) / (valid.max() - valid.min())
                        elif method == "zscore":
                            s = pd.to_numeric(after_df[col], errors="coerce")
                            valid = s.dropna()
                            if not valid.empty and valid.std() not in [0, np.nan] and pd.notna(valid.std()):
                                after_df[col] = (s - valid.mean()) / valid.std()
                    return after_df, "ai_scale_columns", {"columns": cols, "method": method}, cols, history_required, info_message

                if op_type == "drop_rows_missing":
                    cols = resolve_many_column_names(op.get("columns", []), after_df.columns.tolist())
                    if not cols:
                        return current_df, None, None, None, False, "Skipped drop_rows_missing because no valid columns were found."
                    before_rows = len(after_df)
                    after_df = after_df.dropna(subset=cols)
                    return after_df, "ai_drop_rows_missing", {"columns": cols, "rows_removed": int(before_rows - len(after_df))}, cols, history_required, info_message

                if op_type == "drop_columns_missing_threshold":
                    threshold = float(op.get("threshold_percent", 50))
                    missing_pct = after_df.isna().mean() * 100
                    cols = missing_pct[missing_pct > threshold].index.tolist()
                    after_df = after_df.drop(columns=cols)
                    return after_df, "ai_drop_columns_missing_threshold", {"threshold_percent": threshold, "columns_dropped": cols}, cols, history_required, info_message

                if op_type == "convert_type":
                    col = resolve_column_name(op.get("column"), after_df.columns.tolist())
                    if not col:
                        return current_df, None, None, None, False, "Skipped convert_type because the column was not found."
                    target_type = str(op.get("target_type", "")).lower()
                    if target_type == "numeric":
                        after_df[col] = pd.to_numeric(after_df[col], errors="coerce")
                    elif target_type == "categorical":
                        after_df[col] = after_df[col].astype("category")
                    elif target_type == "datetime":
                        dt_format = op.get("datetime_format") or None
                        if dt_format:
                            after_df[col] = pd.to_datetime(after_df[col], format=dt_format, errors="coerce")
                        else:
                            after_df[col] = pd.to_datetime(after_df[col], errors="coerce")
                    else:
                        return current_df, None, None, None, False, f"Skipped convert_type for {col} because the target type was not supported."
                    return after_df, "ai_convert_type", {"column": col, "target_type": target_type}, [col], history_required, info_message

                if op_type == "map_replace":
                    col = resolve_column_name(op.get("column"), after_df.columns.tolist())
                    if not col:
                        return current_df, None, None, None, False, "Skipped map_replace because the column was not found."
                    mapping = op.get("mapping", {}) or {}
                    set_unmatched_to_other = bool(op.get("set_unmatched_to_other", False))
                    current_series = after_df[col].astype(str)
                    mapped_series = current_series.map(mapping)
                    if set_unmatched_to_other:
                        after_df[col] = mapped_series.fillna("Other")
                    else:
                        after_df[col] = mapped_series.where(mapped_series.notna(), current_series)
                    return after_df, "ai_map_replace", {"column": col, "mapping_size": len(mapping), "set_unmatched_to_other": set_unmatched_to_other}, [col], history_required, info_message

                if op_type == "create_formula_column":
                    new_col = str(op.get("new_column", "")).strip()
                    if not new_col:
                        return current_df, None, None, None, False, "Skipped create_formula_column because the new column name was missing."
                    formula_type = str(op.get("formula_type", "")).strip()
                    col_a = resolve_column_name(op.get("col_a"), after_df.columns.tolist())
                    col_b = resolve_column_name(op.get("col_b"), after_df.columns.tolist())
                    base_col = resolve_column_name(op.get("base_col"), after_df.columns.tolist())
                    log_col = resolve_column_name(op.get("log_col"), after_df.columns.tolist())
                    if formula_type == "colA / colB" and col_a and col_b:
                        a = pd.to_numeric(after_df[col_a], errors="coerce")
                        b = pd.to_numeric(after_df[col_b], errors="coerce").replace(0, np.nan)
                        after_df[new_col] = a / b
                    elif formula_type == "log(col)" and log_col:
                        x = pd.to_numeric(after_df[log_col], errors="coerce")
                        after_df[new_col] = np.where(x > 0, np.log(x), np.nan)
                    elif formula_type == "colA - mean(colA)" and base_col:
                        x = pd.to_numeric(after_df[base_col], errors="coerce")
                        after_df[new_col] = x - x.mean()
                    else:
                        return current_df, None, None, None, False, "Skipped create_formula_column because the requested formula could not be built."
                    return after_df, "ai_create_formula_column", {"new_column": new_col, "formula_type": formula_type, "col_a": col_a, "col_b": col_b, "base_col": base_col, "log_col": log_col}, [new_col], history_required, info_message

                if op_type == "create_binned_column":
                    source_col = resolve_column_name(op.get("source_column"), after_df.columns.tolist())
                    new_col = str(op.get("new_column", "")).strip()
                    method = str(op.get("method", "Equal-width"))
                    bins = int(op.get("bins", 4))
                    if not source_col or not new_col:
                        return current_df, None, None, None, False, "Skipped create_binned_column because required column information was missing."
                    numeric_series = pd.to_numeric(after_df[source_col], errors="coerce")
                    if method == "Quantile":
                        after_df[new_col] = pd.qcut(numeric_series, q=bins, duplicates="drop")
                    else:
                        after_df[new_col] = pd.cut(numeric_series, bins=bins)
                    return after_df, "ai_create_binned_column", {"source_column": source_col, "new_column": new_col, "method": method, "bins": bins}, [source_col, new_col], history_required, info_message

                if op_type == "validation_rule":
                    rule_type = str(op.get("rule_type", "")).strip()
                    violations_df = pd.DataFrame()
                    if rule_type == "Numeric range check":
                        col = resolve_column_name(op.get("column"), after_df.columns.tolist())
                        if not col:
                            return current_df, None, None, None, False, "Skipped validation_rule because the numeric column was not found."
                        min_value = float(op.get("min_value", 0))
                        max_value = float(op.get("max_value", 100))
                        series = pd.to_numeric(after_df[col], errors="coerce")
                        mask = series.notna() & ((series < min_value) | (series > max_value))
                        violations_df = after_df.loc[mask].copy()
                        violations_df["Violation Rule"] = f"{col} not in [{min_value}, {max_value}]"
                    elif rule_type == "Allowed categories list":
                        col = resolve_column_name(op.get("column"), after_df.columns.tolist())
                        allowed = [str(x).strip() for x in op.get("allowed_values", []) if str(x).strip()]
                        if not col:
                            return current_df, None, None, None, False, "Skipped validation_rule because the category column was not found."
                        series = after_df[col].astype(str)
                        mask = ~series.isin(allowed)
                        violations_df = after_df.loc[mask].copy()
                        violations_df["Violation Rule"] = f"{col} not in allowed list"
                    elif rule_type == "Non-null constraint":
                        cols = resolve_many_column_names(op.get("columns", []), after_df.columns.tolist())
                        if not cols:
                            return current_df, None, None, None, False, "Skipped validation_rule because no valid columns were found for the non-null rule."
                        mask = after_df[cols].isna().any(axis=1)
                        violations_df = after_df.loc[mask].copy()
                        violations_df["Violation Rule"] = "Non-null constraint violated"
                    else:
                        return current_df, None, None, None, False, "Skipped validation_rule because the rule type was not supported."

                    st.session_state.validation_violations_df = violations_df.copy()
                    history_required = False
                    params = {"rule_type": rule_type, "violations_found": int(len(violations_df))}
                    return after_df, "ai_validation_rule", params, [], history_required, f"Validation rule completed. Violations found: {len(violations_df)}"

                return current_df, None, None, None, False, f"Skipped unsupported AI operation: {op_type}"

            if st.session_state.ai_cleaning_feedback:
                st.info(st.session_state.ai_cleaning_feedback)

            plan = st.session_state.ai_cleaning_plan
            if isinstance(plan, dict):
                operations = plan.get("operations", []) or []
                assistant_note = plan.get("assistant_note", "")

                if assistant_note:
                    st.write("**AI note:**", assistant_note)

                if operations:
                    display_rows = []
                    for i, op in enumerate(operations, start=1):
                        display_rows.append({
                            "Step": i,
                            "Operation": op.get("operation", ""),
                            "Details": json.dumps(op, ensure_ascii=False)
                        })
                    st.dataframe(pd.DataFrame(display_rows), use_container_width=True)

                    if st.button("Confirm and apply AI cleaning plan", type="primary", use_container_width=True, key="apply_ai_cleaning_plan_btn"):
                        overall_before_df = st.session_state.working_df.copy()
                        current_df = st.session_state.working_df.copy()
                        applied_count = 0
                        info_messages = []

                        for op in operations:
                            after_df, step_name, params, affected_cols, history_required, info_message = apply_ai_operation(current_df, op)

                            if info_message:
                                info_messages.append(info_message)

                            if step_name:
                                if history_required:
                                    push_history_state(current_df)
                                update_working_df(after_df)
                                log_step(step_name, params or {}, affected_columns=affected_cols or [])
                                current_df = after_df.copy()
                                applied_count += 1

                        if applied_count > 0:
                            st.success(f"AI cleaning applied successfully. Steps applied: {applied_count}")
                            before_after_summary(overall_before_df, current_df, list(current_df.columns))
                            show_small_preview(overall_before_df.head(8), current_df.head(8))
                        else:
                            st.warning("No AI cleaning steps were applied.")

                        if info_messages:
                            for msg in info_messages:
                                st.info(msg)
                else:
                    st.info("The AI assistant did not return any cleaning steps for this command.")


        st.markdown("---")

        # ---------------------------------------------
        # LOCAL HELPERS FOR PAGE B
        # ---------------------------------------------
        def safe_update_and_log(after_df, step_name, params, before_df=None, affected_columns=None, preview_cols=None):
            try:
                if affected_columns is None:
                    affected_columns = []

                if before_df is not None:
                    push_history_state(before_df)

                update_working_df(after_df)
                log_step(step_name, params, affected_columns=affected_columns)
                st.success(f"{step_name} applied successfully.")

                if before_df is not None:
                    before_after_summary(before_df, after_df, affected_columns)

                    if preview_cols:
                        valid_preview_cols = [c for c in preview_cols if c in before_df.columns and c in after_df.columns]
                        if valid_preview_cols:
                            show_small_preview(before_df[valid_preview_cols], after_df[valid_preview_cols])
                        else:
                            show_small_preview(before_df, after_df)
                    else:
                        show_small_preview(before_df, after_df)
            except Exception as e:
                st.error(f"Could not apply {step_name}: {e}")

        def get_missing_summary(input_df):
            return pd.DataFrame({
                "Column": input_df.columns,
                "Missing Count": input_df.isna().sum().values,
                "Missing %": ((input_df.isna().mean()) * 100).round(2).values
            }).sort_values(by=["Missing Count", "Column"], ascending=[False, True])

        def get_duplicate_groups(input_df, subset_cols=None):
            if subset_cols:
                group_mask = input_df.duplicated(subset=subset_cols, keep=False)
                count_mask = input_df.duplicated(subset=subset_cols, keep="first")
            else:
                group_mask = input_df.duplicated(keep=False)
                count_mask = input_df.duplicated(keep="first")
            return input_df[group_mask].copy(), count_mask

        def clean_dirty_numeric(series):
            return pd.to_numeric(
                series.astype(str).str.replace(r"[,\$\€\£\%\s]", "", regex=True),
                errors="coerce"
            )

        def parse_datetime_series(series, fmt=None):
            if fmt and fmt.strip():
                return pd.to_datetime(series, format=fmt, errors="coerce")
            return pd.to_datetime(series, errors="coerce")

        def standardize_text(series, mode="trim"):
            s = series.astype("string")
            if mode == "trim":
                return s.str.strip()
            elif mode == "lower":
                return s.str.strip().str.lower()
            elif mode == "title":
                return s.str.strip().str.title()
            return s

        def outlier_mask_iqr(series):
            s = pd.to_numeric(series, errors="coerce")
            valid = s.dropna()
            if valid.empty:
                return pd.Series(False, index=series.index)
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (s < lower) | (s > upper)

        def outlier_mask_zscore(series, threshold=3.0):
            s = pd.to_numeric(series, errors="coerce")
            valid = s.dropna()
            if valid.empty or valid.std() == 0 or pd.isna(valid.std()):
                return pd.Series(False, index=series.index)
            z = (s - valid.mean()) / valid.std()
            return z.abs() > threshold

        def winsorize_series(series, lower_q=0.01, upper_q=0.99):
            s = pd.to_numeric(series, errors="coerce")
            valid = s.dropna()
            if valid.empty:
                return s
            lower_val = valid.quantile(lower_q)
            upper_val = valid.quantile(upper_q)
            return s.clip(lower=lower_val, upper=upper_val), lower_val, upper_val

        def min_max_scale_series(series):
            s = pd.to_numeric(series, errors="coerce")
            valid = s.dropna()
            if valid.empty:
                return s
            min_v = valid.min()
            max_v = valid.max()
            if min_v == max_v:
                return s
            return (s - min_v) / (max_v - min_v)

        def zscore_scale_series(series):
            s = pd.to_numeric(series, errors="coerce")
            valid = s.dropna()
            if valid.empty:
                return s
            std = valid.std()
            if std == 0 or pd.isna(std):
                return s
            return (s - valid.mean()) / std

        def stats_summary(series):
            s = pd.to_numeric(series, errors="coerce")
            return {
                "count": int(s.notna().sum()),
                "mean": round(float(s.mean()), 4) if s.notna().any() else None,
                "std": round(float(s.std()), 4) if s.notna().any() else None,
                "min": round(float(s.min()), 4) if s.notna().any() else None,
                "max": round(float(s.max()), 4) if s.notna().any() else None,
            }

        def build_formula_column(input_df, formula_type, col_a=None, col_b=None, base_col=None, log_col=None):
            out = None
            if formula_type == "colA / colB":
                a = pd.to_numeric(input_df[col_a], errors="coerce")
                b = pd.to_numeric(input_df[col_b], errors="coerce")
                out = a / b.replace(0, np.nan)

            elif formula_type == "log(col)":
                x = pd.to_numeric(input_df[log_col], errors="coerce")
                out = np.where(x > 0, np.log(x), np.nan)

            elif formula_type == "colA - mean(colA)":
                x = pd.to_numeric(input_df[base_col], errors="coerce")
                out = x - x.mean()

            return pd.Series(out, index=input_df.index)

        if "validation_violations_df" not in st.session_state:
            st.session_state.validation_violations_df = pd.DataFrame()

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()
        categorical_like_cols = [
            c for c in df.columns
            if str(df[c].dtype) in ["object", "string", "category", "bool"]
        ]

        tabs = st.tabs([
            "Missing Values",
            "Duplicates",
            "Data Types & Parsing",
            "Categorical Tools",
            "Numeric Cleaning",
            "Scaling",
            "Column Operations",
            "Validation Rules"
        ])

        # =================================================
        # 4.1 Missing Values
        # =================================================
        with tabs[0]:
            st.write("### Missing Value Summary")
            missing_df = get_missing_summary(df)
            st.dataframe(missing_df, use_container_width=True)

            st.write("### Missing Value Actions")
            mv_action = st.selectbox(
                "Choose action",
                [
                    "Drop rows with missing values (selected columns)",
                    "Drop columns above missing threshold (%)",
                    "Replace missing values"
                ],
                key="mv_action"
            )

            if mv_action == "Drop rows with missing values (selected columns)":
                selected_cols = st.multiselect("Select columns", all_cols, key="mv_drop_rows_cols")

                if st.button("Apply row drop", key="mv_drop_rows_btn"):
                    if not selected_cols:
                        st.warning("Please select at least one column.")
                    else:
                        before_df = df.copy()
                        after_df = df.dropna(subset=selected_cols)
                        safe_update_and_log(
                            after_df,
                            "drop_rows_missing_selected_columns",
                            {
                                "columns": selected_cols,
                                "rows_removed": int(before_df.shape[0] - after_df.shape[0])
                            },
                            before_df=before_df,
                            affected_columns=selected_cols
                        )

            elif mv_action == "Drop columns above missing threshold (%)":
                threshold = st.slider("Threshold (%)", 0, 100, 50, key="mv_threshold")

                if st.button("Apply column drop", key="mv_drop_cols_btn"):
                    before_df = df.copy()
                    missing_pct = df.isna().mean() * 100
                    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
                    after_df = df.drop(columns=cols_to_drop)

                    safe_update_and_log(
                        after_df,
                        "drop_columns_above_missing_threshold",
                        {
                            "threshold_percent": threshold,
                            "columns_dropped": cols_to_drop
                        },
                        before_df=before_df,
                        affected_columns=cols_to_drop if cols_to_drop else []
                    )

            elif mv_action == "Replace missing values":
                selected_col = st.selectbox("Select column", all_cols, key="mv_replace_col")
                is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
                st.write(f"**Detected type:** {'Numeric' if is_numeric else 'Categorical'}")

                methods = ["Constant value", "Forward fill", "Backward fill"]
                if is_numeric:
                    methods += ["Mean", "Median", "Mode"]
                else:
                    methods += ["Most frequent"]

                fill_method = st.selectbox("Replacement method", methods, key="mv_replace_method")
                constant_value = None
                if fill_method == "Constant value":
                    constant_value = st.text_input("Enter constant value", key="mv_constant")

                if st.button("Apply replacement", key="mv_replace_btn"):
                    before_df = df.copy()
                    after_df = df.copy()

                    if fill_method == "Constant value":
                        after_df[selected_col] = after_df[selected_col].fillna(constant_value)
                    elif fill_method == "Mean":
                        x = pd.to_numeric(after_df[selected_col], errors="coerce")
                        after_df[selected_col] = x.fillna(x.mean())
                    elif fill_method == "Median":
                        x = pd.to_numeric(after_df[selected_col], errors="coerce")
                        after_df[selected_col] = x.fillna(x.median())
                    elif fill_method in ["Mode", "Most frequent"]:
                        mode_val = after_df[selected_col].mode(dropna=True)
                        if not mode_val.empty:
                            after_df[selected_col] = after_df[selected_col].fillna(mode_val.iloc[0])
                    elif fill_method == "Forward fill":
                        after_df[selected_col] = after_df[selected_col].ffill()
                    elif fill_method == "Backward fill":
                        after_df[selected_col] = after_df[selected_col].bfill()

                    safe_update_and_log(
                        after_df,
                        "replace_missing_values",
                        {
                            "column": selected_col,
                            "method": fill_method
                        },
                        before_df=before_df,
                        affected_columns=[selected_col],
                        preview_cols=[selected_col]
                    )

        # =================================================
        # 4.2 Duplicates
        # =================================================
        with tabs[1]:
            st.write("### Duplicate Detection")

            dup_mode = st.radio(
                "Detect duplicates by",
                ["Full row", "Subset of columns"],
                key="dup_mode"
            )

            subset_cols = None
            if dup_mode == "Subset of columns":
                subset_cols = st.multiselect("Select key columns", all_cols, key="dup_subset_cols")

            if dup_mode == "Full row":
                duplicate_groups, dup_mask = get_duplicate_groups(df)
            else:
                duplicate_groups, dup_mask = get_duplicate_groups(df, subset_cols if subset_cols else None)

            st.metric("Duplicate Rows Found", int(dup_mask.sum()))

            st.write("### Duplicate Groups")
            if not duplicate_groups.empty:
                st.dataframe(duplicate_groups, use_container_width=True)
            else:
                st.info("No duplicate groups found.")

            keep_option = st.selectbox("Keep", ["first", "last"], key="dup_keep_option")

            if st.button("Remove duplicates", key="dup_remove_btn"):
                if dup_mode == "Subset of columns" and not subset_cols:
                    st.warning("Please select key columns.")
                else:
                    before_df = df.copy()
                    if dup_mode == "Full row":
                        after_df = df.drop_duplicates(keep=keep_option)
                        used_cols = all_cols
                    else:
                        after_df = df.drop_duplicates(subset=subset_cols, keep=keep_option)
                        used_cols = subset_cols

                    safe_update_and_log(
                        after_df,
                        "remove_duplicates",
                        {
                            "duplicate_type": dup_mode,
                            "subset_columns": used_cols,
                            "keep": keep_option,
                            "rows_removed": int(before_df.shape[0] - after_df.shape[0])
                        },
                        before_df=before_df,
                        affected_columns=used_cols
                    )

        # =================================================
        # 4.3 Data Types & Parsing
        # =================================================
        with tabs[2]:
            st.write("### Data Type Conversion")
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Current Type": df.dtypes.astype(str).values
            })
            st.dataframe(dtype_df, use_container_width=True)

            dt_col = st.selectbox("Select column", all_cols, key="dtype_col")
            convert_type = st.selectbox(
                "Convert to",
                ["numeric", "categorical", "datetime"],
                key="dtype_target"
            )

            datetime_format = ""
            if convert_type == "datetime":
                datetime_format = st.text_input(
                    "Datetime format (optional, e.g. %d/%m/%Y). Leave blank for auto parse.",
                    key="datetime_format"
                )

            numeric_dirty = False
            if convert_type == "numeric":
                numeric_dirty = st.checkbox(
                    "Clean dirty numeric strings first (commas, currency signs, %, spaces)",
                    value=True,
                    key="numeric_dirty_checkbox"
                )

            if st.button("Apply conversion", key="dtype_convert_btn"):
                before_df = df.copy()
                after_df = df.copy()

                if convert_type == "numeric":
                    if numeric_dirty:
                        after_df[dt_col] = clean_dirty_numeric(after_df[dt_col])
                    else:
                        after_df[dt_col] = pd.to_numeric(after_df[dt_col], errors="coerce")

                elif convert_type == "categorical":
                    after_df[dt_col] = after_df[dt_col].astype("category")

                elif convert_type == "datetime":
                    after_df[dt_col] = parse_datetime_series(after_df[dt_col], fmt=datetime_format)


                safe_update_and_log(
                    after_df,
                    "convert_column_type",
                    {
                        "column": dt_col,
                        "target_type": convert_type,
                        "datetime_format": datetime_format if convert_type == "datetime" else None,
                        "dirty_numeric_cleaning": numeric_dirty if convert_type == "numeric" else None
                    },
                    before_df=before_df,
                    affected_columns=[dt_col],
                    preview_cols=[dt_col]
                )

        # =================================================
        # 4.4 Categorical Data Tools
        # =================================================
        with tabs[3]:
            st.write("### Categorical Data Tools")

            if not all_cols:
                st.info("No columns available.")
            else:
                cat_col = st.selectbox("Select column", all_cols, key="cat_tools_col")

                subtab1, subtab2, subtab3, subtab4 = st.tabs([
                    "Value Standardization",
                    "Mapping / Replacement",
                    "Rare Category Grouping",
                    "One-hot Encoding"
                ])

                with subtab1:
                    standard_mode = st.selectbox(
                        "Standardization option",
                        ["trim", "lower", "title"],
                        key="cat_standard_mode"
                    )

                    if st.button("Apply standardization", key="cat_standard_btn"):
                        before_df = df.copy()
                        after_df = df.copy()
                        after_df[cat_col] = standardize_text(after_df[cat_col], mode=standard_mode)

                        safe_update_and_log(
                            after_df,
                            "standardize_categorical_values",
                            {
                                "column": cat_col,
                                "mode": standard_mode
                            },
                            before_df=before_df,
                            affected_columns=[cat_col],
                            preview_cols=[cat_col]
                        )

                with subtab2:
                    st.write("Enter mapping pairs below:")
                    unique_vals = pd.Series(df[cat_col].dropna().astype(str).unique()).sort_values().tolist()
                    seed_rows = [{"old_value": v, "new_value": ""} for v in unique_vals[:20]]

                    mapping_editor = st.data_editor(
                        pd.DataFrame(seed_rows),
                        num_rows="dynamic",
                        use_container_width=True,
                        key="mapping_editor"
                    )

                    set_unmatched_other = st.checkbox(
                        "Set unmatched values to 'Other'",
                        key="set_unmatched_other"
                    )

                    if st.button("Apply mapping", key="apply_mapping_btn"):
                        before_df = df.copy()
                        after_df = df.copy()

                        mapping_df = mapping_editor.copy()
                        mapping_df["old_value"] = mapping_df["old_value"].astype(str)
                        mapping_df["new_value"] = mapping_df["new_value"].astype(str)

                        mapping_dict = {
                            row["old_value"]: row["new_value"]
                            for _, row in mapping_df.iterrows()
                            if row["old_value"] != "" and row["new_value"] != ""
                        }

                        current_series = after_df[cat_col].astype(str)
                        mapped_series = current_series.map(mapping_dict)

                        if set_unmatched_other:
                            after_df[cat_col] = mapped_series.fillna("Other")
                        else:
                            after_df[cat_col] = mapped_series.where(mapped_series.notna(), current_series)

                        safe_update_and_log(
                            after_df,
                            "map_replace_categorical_values",
                            {
                                "column": cat_col,
                                "mapping_size": len(mapping_dict),
                                "set_unmatched_to_other": set_unmatched_other
                            },
                            before_df=before_df,
                            affected_columns=[cat_col],
                            preview_cols=[cat_col]
                        )

                with subtab3:
                    threshold_pct = st.slider(
                        "Group categories below frequency threshold (%)",
                        0.0, 20.0, 5.0, 0.5,
                        key="rare_threshold_pct"
                    )

                    if st.button("Apply rare grouping", key="rare_group_btn"):
                        before_df = df.copy()
                        after_df = df.copy()

                        freq_pct = after_df[cat_col].value_counts(normalize=True, dropna=False) * 100
                        rare_cats = freq_pct[freq_pct < threshold_pct].index.tolist()

                        after_df[cat_col] = after_df[cat_col].apply(
                            lambda x: "Other" if x in rare_cats else x
                        )

                        safe_update_and_log(
                            after_df,
                            "group_rare_categories",
                            {
                                "column": cat_col,
                                "threshold_percent": threshold_pct,
                                "rare_categories_count": len(rare_cats)
                            },
                            before_df=before_df,
                            affected_columns=[cat_col],
                            preview_cols=[cat_col]
                        )

                with subtab4:
                    ohe_cols = st.multiselect(
                        "Select columns to one-hot encode",
                        all_cols,
                        default=[cat_col] if cat_col else [],
                        key="ohe_cols"
                    )
                    drop_first = st.checkbox("Drop first dummy column", key="ohe_drop_first")

                    if st.button("Apply one-hot encoding", key="ohe_btn"):
                        if not ohe_cols:
                            st.warning("Please select at least one column.")
                        else:
                            before_df = df.copy()
                            after_df = pd.get_dummies(df, columns=ohe_cols, drop_first=drop_first)

                            safe_update_and_log(
                                after_df,
                                "one_hot_encode_columns",
                                {
                                    "columns": ohe_cols,
                                    "drop_first": drop_first
                                },
                                before_df=before_df,
                                affected_columns=ohe_cols
                            )

        # =================================================
        # 4.5 Numeric Cleaning
        # =================================================
        with tabs[4]:
            st.write("### Numeric Cleaning")

            if not all_cols:
                st.info("No columns available.")
            else:
                num_clean_col = st.selectbox("Select numeric column", all_cols, key="num_clean_col")
                detect_method = st.selectbox("Outlier method", ["IQR", "Z-score"], key="outlier_method")
                z_threshold = 3.0
                if detect_method == "Z-score":
                    z_threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1, key="z_thresh")

                s = pd.to_numeric(df[num_clean_col], errors="coerce")
                mask = outlier_mask_iqr(s) if detect_method == "IQR" else outlier_mask_zscore(s, threshold=z_threshold)

                st.write("### Outlier Detection Summary")
                summary_df = pd.DataFrame({
                    "column": [num_clean_col],
                    "non_null_values": [int(s.notna().sum())],
                    "outliers_found": [int(mask.sum())],
                    "outlier_percent": [round((mask.sum() / len(df)) * 100, 2) if len(df) else 0]
                })
                st.dataframe(summary_df, use_container_width=True)

                outlier_action = st.selectbox(
                    "Choose action",
                    ["Do nothing", "Cap / winsorize at quantiles", "Remove outlier rows"],
                    key="outlier_action"
                )

                lower_q, upper_q = 0.01, 0.99
                if outlier_action == "Cap / winsorize at quantiles":
                    lower_q = st.slider("Lower quantile", 0.0, 0.2, 0.01, 0.01, key="lower_q")
                    upper_q = st.slider("Upper quantile", 0.8, 1.0, 0.99, 0.01, key="upper_q")

                if st.button("Apply outlier action", key="outlier_apply_btn"):
                    before_df = df.copy()
                    after_df = df.copy()

                    if outlier_action == "Do nothing":
                        st.info("No changes applied.")

                    elif outlier_action == "Cap / winsorize at quantiles":
                        capped_series, lower_val, upper_val = winsorize_series(after_df[num_clean_col], lower_q, upper_q)
                        changed_count = int((pd.to_numeric(after_df[num_clean_col], errors="coerce") != capped_series).sum())
                        after_df[num_clean_col] = capped_series

                        safe_update_and_log(
                            after_df,
                            "winsorize_outliers",
                            {
                                "column": num_clean_col,
                                "method": detect_method,
                                "lower_quantile": lower_q,
                                "upper_quantile": upper_q,
                                "lower_cap_value": float(lower_val) if pd.notna(lower_val) else None,
                                "upper_cap_value": float(upper_val) if pd.notna(upper_val) else None,
                                "values_capped": changed_count
                            },
                            before_df=before_df,
                            affected_columns=[num_clean_col],
                            preview_cols=[num_clean_col]
                        )

                    elif outlier_action == "Remove outlier rows":
                        rows_removed = int(mask.sum())
                        after_df = after_df.loc[~mask].copy()

                        safe_update_and_log(
                            after_df,
                            "remove_outlier_rows",
                            {
                                "column": num_clean_col,
                                "method": detect_method,
                                "rows_removed": rows_removed
                            },
                            before_df=before_df,
                            affected_columns=[num_clean_col]
                        )

        # =================================================
        # 4.6 Normalization / Scaling
        # =================================================
        with tabs[5]:
            st.write("### Normalization / Scaling")

            scaling_cols = st.multiselect(
                "Select numeric columns",
                all_cols,
                key="scaling_cols"
            )
            scaling_method = st.selectbox(
                "Scaling method",
                ["Min-max scaling", "Z-score standardization"],
                key="scaling_method"
            )

            if st.button("Apply scaling", key="scaling_btn"):
                if not scaling_cols:
                    st.warning("Please select at least one column.")
                else:
                    invalid_scaling_cols = [col for col in scaling_cols if not pd.api.types.is_numeric_dtype(df[col])]
                    if invalid_scaling_cols:
                        st.error("Scaling can only be applied to numeric columns. Please remove these columns: " + ", ".join(invalid_scaling_cols))
                    else:
                        before_df = df.copy()
                        after_df = df.copy()

                        before_stats_rows = []
                        after_stats_rows = []

                        for col in scaling_cols:
                            before_stats_rows.append({"column": col, **stats_summary(after_df[col])})

                            if scaling_method == "Min-max scaling":
                                after_df[col] = min_max_scale_series(after_df[col])
                            else:
                                after_df[col] = zscore_scale_series(after_df[col])

                            after_stats_rows.append({"column": col, **stats_summary(after_df[col])})

                    st.write("### Before scaling")
                    st.dataframe(pd.DataFrame(before_stats_rows), use_container_width=True)

                    st.write("### After scaling")
                    st.dataframe(pd.DataFrame(after_stats_rows), use_container_width=True)

                    safe_update_and_log(
                        after_df,
                        "scale_numeric_columns",
                        {
                            "columns": scaling_cols,
                            "method": scaling_method
                        },
                        before_df=before_df,
                        affected_columns=scaling_cols,
                        preview_cols=scaling_cols[:3]
                    )

        # =================================================
        # 4.7 Column Operations
        # =================================================
        with tabs[6]:
            st.write("### Column Operations")

            col_subtabs = st.tabs([
                "Rename Columns",
                "Drop Columns",
                "Create Formula Column",
                "Binning"
            ])

            with col_subtabs[0]:
                rename_col = st.selectbox("Select column to rename", all_cols, key="rename_col")
                new_name = st.text_input("New column name", key="rename_new_name")

                if st.button("Rename column", key="rename_btn"):
                    if not new_name.strip():
                        st.warning("Please enter a new column name.")
                    elif new_name in df.columns:
                        st.warning("That column name already exists.")
                    else:
                        before_df = df.copy()
                        after_df = df.rename(columns={rename_col: new_name})

                        safe_update_and_log(
                            after_df,
                            "rename_column",
                            {
                                "old_name": rename_col,
                                "new_name": new_name
                            },
                            before_df=before_df,
                            affected_columns=[rename_col]
                        )

            with col_subtabs[1]:
                drop_cols = st.multiselect("Select columns to drop", all_cols, key="drop_cols")

                if st.button("Drop selected columns", key="drop_cols_btn"):
                    if not drop_cols:
                        st.warning("Please select at least one column.")
                    else:
                        before_df = df.copy()
                        after_df = df.drop(columns=drop_cols)

                        safe_update_and_log(
                            after_df,
                            "drop_columns",
                            {
                                "columns_dropped": drop_cols
                            },
                            before_df=before_df,
                            affected_columns=drop_cols
                        )

            with col_subtabs[2]:
                formula_name = st.text_input("New formula column name", key="formula_name")
                formula_type = st.selectbox(
                    "Formula type",
                    ["colA / colB", "log(col)", "colA - mean(colA)"],
                    key="formula_type"
                )

                col_a = col_b = base_col = log_col = None

                if formula_type == "colA / colB":
                    col_a = st.selectbox("Select colA", all_cols, key="formula_col_a")
                    col_b = st.selectbox("Select colB", all_cols, key="formula_col_b")
                elif formula_type == "log(col)":
                    log_col = st.selectbox("Select column", all_cols, key="formula_log_col")
                elif formula_type == "colA - mean(colA)":
                    base_col = st.selectbox("Select column", all_cols, key="formula_base_col")

                if st.button("Create formula column", key="formula_btn"):
                    if not formula_name.strip():
                        st.warning("Please enter a new column name.")
                    elif formula_name in df.columns:
                        st.warning("That column name already exists.")
                    else:
                        before_df = df.copy()
                        after_df = df.copy()
                        after_df[formula_name] = build_formula_column(
                            after_df,
                            formula_type=formula_type,
                            col_a=col_a,
                            col_b=col_b,
                            base_col=base_col,
                            log_col=log_col
                        )

                        safe_update_and_log(
    after_df,
    "create_formula_column",
    {
        "new_column": formula_name,
        "formula_type": formula_type,
        "col_a": col_a,
        "col_b": col_b,
        "base_col": base_col,
        "log_col": log_col
    },
    before_df=before_df,
    affected_columns=[],
    preview_cols=[]
    )

            with col_subtabs[3]:
                bin_col = st.selectbox("Select numeric column to bin", all_cols, key="bin_col")
                bin_new_name = st.text_input("New binned column name", key="bin_new_name")
                bin_method = st.selectbox("Binning method", ["Equal-width", "Quantile"], key="bin_method")
                num_bins = st.slider("Number of bins", 2, 10, 4, key="num_bins")

                if st.button("Create binned column", key="bin_btn"):
                    if not bin_new_name.strip():
                        st.warning("Please enter a new column name.")
                    elif bin_new_name in df.columns:
                        st.warning("That column name already exists.")
                    elif not pd.api.types.is_numeric_dtype(df[bin_col]) and pd.to_numeric(df[bin_col], errors="coerce").notna().sum() == 0:
                        st.error("Binning can only be applied to numeric columns or columns that can be converted to numeric.")
                    else:
                        before_df = df.copy()
                        after_df = df.copy()

                        numeric_series = pd.to_numeric(after_df[bin_col], errors="coerce")

                        try:
                            if bin_method == "Equal-width":
                                after_df[bin_new_name] = pd.cut(numeric_series, bins=num_bins)
                            else:
                                after_df[bin_new_name] = pd.qcut(numeric_series, q=num_bins, duplicates="drop")

                            safe_update_and_log(
                                after_df,
                                "create_binned_column",
                                {
                                    "source_column": bin_col,
                                    "new_column": bin_new_name,
                                    "method": bin_method,
                                    "bins": num_bins
                                },
                                before_df=before_df,
                                affected_columns=[],
                                preview_cols=[]
                            )
                        except Exception as e:
                            st.error(f"Binning failed: {e}")

        # =================================================
        # 4.8 Data Validation Rules
        # =================================================
        with tabs[7]:
            st.write("### Data Validation Rules")

            rule_type = st.selectbox(
                "Rule type",
                ["Numeric range check", "Allowed categories list", "Non-null constraint"],
                key="validation_rule_type"
            )

            violations_df = pd.DataFrame()

            if rule_type == "Numeric range check":
                val_col = st.selectbox("Select column", all_cols, key="val_num_col")
                min_val = st.number_input("Minimum allowed value", value=0.0, key="val_min")
                max_val = st.number_input("Maximum allowed value", value=100.0, key="val_max")

                if st.button("Run numeric range validation", key="run_num_validation"):
                    series = pd.to_numeric(df[val_col], errors="coerce")
                    mask = series.notna() & ((series < min_val) | (series > max_val))
                    violations_df = df.loc[mask].copy()
                    violations_df["Violation Rule"] = f"{val_col} not in [{min_val}, {max_val}]"

            elif rule_type == "Allowed categories list":
                val_col = st.selectbox("Select column", all_cols, key="val_cat_col")
                allowed_raw = st.text_input(
                    "Allowed categories (comma-separated)",
                    placeholder="A, B, C",
                    key="allowed_cats_raw"
                )

                if st.button("Run allowed-category validation", key="run_cat_validation"):
                    allowed = [x.strip() for x in allowed_raw.split(",") if x.strip() != ""]
                    series = df[val_col].astype(str)
                    mask = ~series.isin(allowed)
                    violations_df = df.loc[mask].copy()
                    violations_df["Violation Rule"] = f"{val_col} not in allowed list"

            elif rule_type == "Non-null constraint":
                val_cols = st.multiselect("Select columns", all_cols, key="nonnull_cols")

                if st.button("Run non-null validation", key="run_nonnull_validation"):
                    if not val_cols:
                        st.warning("Please select at least one column.")
                    else:
                        mask = df[val_cols].isna().any(axis=1)
                        violations_df = df.loc[mask].copy()
                        violations_df["Violation Rule"] = "Non-null constraint violated"

            if not violations_df.empty:
                st.write("### Violations Table")
                st.dataframe(violations_df, use_container_width=True)

                st.session_state.validation_violations_df = violations_df.copy()

                csv_bytes = violations_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download violations table as CSV",
                    data=csv_bytes,
                    file_name="validation_violations.csv",
                    mime="text/csv",
                    key="download_violations_csv"
                )
            else:
                if st.session_state.validation_violations_df is not None and not st.session_state.validation_violations_df.empty:
                    st.info("No new violations from the latest run, or no rule has been run yet.")
                else:
                    st.info("Run a validation rule to see violations.")
    # =========================================================
