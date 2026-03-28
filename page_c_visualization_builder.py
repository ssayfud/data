from shared_core import *
from ai_helpers import *


def render_page_c():
    st.title("Visualization Builder")

    if "working_df" not in st.session_state or st.session_state.working_df is None:
        st.warning("Please upload a dataset first on Page A.")
        st.stop()

    df = st.session_state.working_df.copy()

    if df.empty:
        st.warning("The dataset is empty.")
        st.stop()

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    st.write("### AI Assistant")
    enable_ai_assistant_c = st.toggle(
        "Enable AI assistant",
        value=False,
        key="enable_ai_assistant_page_c"
    )
    st.caption("AI outputs may be imperfect. Use this to get chart ideas before building the chart below.")

    if enable_ai_assistant_c:
        ai_chart_prompt = st.text_area(
            "Ask for chart suggestions",
            placeholder="Example: Given these columns, recommend good visualizations for trends, categories, and relationships.",
            key="ai_chart_prompt"
        )

        if st.button("Get AI chart suggestions", use_container_width=False, key="get_ai_chart_suggestions_btn"):
            st.session_state.ai_chart_suggestions = generate_ai_chart_suggestions(ai_chart_prompt, df)

        if st.session_state.ai_chart_suggestions:
            st.info(st.session_state.ai_chart_suggestions)

    st.markdown("---")

    chart_save_col1, chart_save_col2 = st.columns([2, 1])
    with chart_save_col1:
        dashboard_chart_title = st.text_input(
            "Dashboard chart title",
            value="",
            placeholder="Optional: give this chart a clean title for Page D",
            key="dashboard_chart_title"
        )
    with chart_save_col2:
        if st.button("Add last chart to dashboard", use_container_width=True, key="add_last_chart_to_dashboard_btn"):
            ok, message = add_last_chart_to_dashboard(dashboard_chart_title)
            if ok:
                st.success(message)
            else:
                st.info(message)

    saved_count = len(st.session_state.dashboard_saved_charts)
    st.caption(f"Saved dashboard charts: {saved_count}")

    st.write("### Choose Your Chart")

    plot_type = st.selectbox(
        "Plot type",
        [
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "Line Chart",
            "Bar Chart",
            "Heatmap / Correlation Matrix"
        ],
        key="viz_plot_type"
    )

    enable_hover = st.checkbox(
        "Enable hover effect (interactive)",
        value=True,
        key="viz_enable_hover"
    )

    # ---------------------------------------------------------
    # Filters
    # ---------------------------------------------------------
    st.write("### Filters")
    filtered_df = df.copy()

    filter_mode = st.selectbox(
        "Filter mode",
        ["No filter", "Category filter", "Numeric range filter"],
        key="viz_filter_mode"
    )

    if filter_mode == "Category filter":
        if categorical_cols:
            filter_col = st.selectbox(
                "Select categorical column",
                categorical_cols,
                key="viz_cat_filter_col"
            )

            available_values = sorted(
                filtered_df[filter_col].dropna().astype(str).unique().tolist()
            )

            selected_values = st.multiselect(
                "Select category values",
                available_values,
                default=available_values[:5] if len(available_values) > 5 else available_values,
                key="viz_cat_filter_vals"
            )

            if selected_values:
                filtered_df = filtered_df[
                    filtered_df[filter_col].astype(str).isin(selected_values)
                ]
        else:
            st.info("No categorical columns available for filtering.")

    elif filter_mode == "Numeric range filter":
        if numeric_cols:
            filter_col = st.selectbox(
                "Select numeric column",
                numeric_cols,
                key="viz_num_filter_col"
            )

            col_data = pd.to_numeric(filtered_df[filter_col], errors="coerce").dropna()

            if not col_data.empty:
                min_val = float(col_data.min())
                max_val = float(col_data.max())

                selected_range = st.slider(
                    "Select numeric range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key="viz_num_filter_range"
                )

                filtered_df = filtered_df[
                    pd.to_numeric(filtered_df[filter_col], errors="coerce").between(
                        selected_range[0], selected_range[1]
                    )
                ]
            else:
                st.info("Selected column has no valid numeric values.")
        else:
            st.info("No numeric columns available for filtering.")

    st.write(f"Filtered rows: **{len(filtered_df):,}**")

    # ---------------------------------------------------------
    # Common selections
    # ---------------------------------------------------------
    x_col = None
    y_col = None
    color_col = None
    agg_option = None
    top_n = None

    if plot_type in ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot"]:
        x_col = st.selectbox("X column", all_cols, key="viz_x_col")

    if plot_type in ["Histogram", "Box Plot", "Scatter Plot", "Line Chart", "Bar Chart"]:
        if not numeric_cols:
            st.warning("No numeric columns available for Y axis.")
            st.stop()
        y_col = st.selectbox("Y column", numeric_cols, key="viz_y_col")

    if plot_type in ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot"]:
        color_options = ["None"] + all_cols
        color_col = st.selectbox(
            "Optional color/group column",
            color_options,
            key="viz_color_col"
        )
        if color_col == "None":
            color_col = None

    if plot_type in ["Line Chart", "Bar Chart"]:
        agg_option = st.selectbox(
            "Optional aggregation",
            ["None", "sum", "mean", "count", "median"],
            key="viz_agg_option"
        )

    if plot_type == "Bar Chart":
        top_n = st.number_input(
            "Top N categories",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="viz_top_n"
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def prepare_x_series(series):
        """
        Decide whether x should be numeric, datetime, or category.
        Returns: cleaned_series, x_kind
        x_kind in {"numeric", "datetime", "category"}
        """
        if pd.api.types.is_numeric_dtype(series):
            x_num = pd.to_numeric(series, errors="coerce")
            return x_num, "numeric"

        # Try numeric conversion first for object columns like "1", "2", "3"
        x_num_try = pd.to_numeric(series, errors="coerce")
        numeric_ratio = x_num_try.notna().mean()

        if numeric_ratio > 0.95:
            return x_num_try, "numeric"

        # Try datetime only if clearly date-like
        x_dt_try = pd.to_datetime(series, errors="coerce")
        datetime_ratio = x_dt_try.notna().mean()

        if datetime_ratio > 0.95:
            return x_dt_try, "datetime"

        # Otherwise treat as category/text
        return series.astype(str), "category"

    # ---------------------------------------------------------
    # Generate chart
    # ---------------------------------------------------------
    if st.button("Generate Chart", key="viz_generate_btn"):
        plot_df = filtered_df.copy()

        if plot_df.empty:
            st.warning("No data available after filtering.")
            st.stop()

        try:
            # =====================================================
            # 1. HISTOGRAM
            # =====================================================
            if plot_type == "Histogram":
                series = pd.to_numeric(plot_df[y_col], errors="coerce").dropna()

                if series.empty:
                    st.warning("No valid numeric data to plot.")
                    st.stop()

                if enable_hover:
                    hist_df = pd.DataFrame({y_col: series})
                    fig = px.histogram(
                        hist_df,
                        x=y_col,
                        nbins=30,
                        title=f"Histogram of {y_col}"
                    )
                    fig.update_layout(
                        xaxis_title=y_col,
                        yaxis_title="Frequency",
                        hovermode="closest",
                        bargap=0.05
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    set_last_generated_plotly_chart(fig, f"Histogram of {y_col}", "Histogram")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(series, bins=30)
                    ax.set_title(f"Histogram of {y_col}")
                    ax.set_xlabel(y_col)
                    ax.set_ylabel("Frequency")
                    plt.tight_layout()
                    st.pyplot(fig)
                    set_last_generated_matplotlib_chart(fig, f"Histogram of {y_col}", "Histogram")

            # =====================================================
            # 2. BOX PLOT
            # =====================================================
            elif plot_type == "Box Plot":
                temp_df = plot_df.copy()
                temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors="coerce")
                temp_df = temp_df.dropna(subset=[y_col])

                if temp_df.empty:
                    st.warning("No valid data to plot.")
                    st.stop()

                if enable_hover:
                    if color_col:
                        fig = px.box(
                            temp_df,
                            x=color_col,
                            y=y_col,
                            color=color_col,
                            points="all",
                            hover_data=temp_df.columns.tolist(),
                            title=f"Box Plot of {y_col}"
                        )
                        fig.update_layout(
                            xaxis_title=color_col,
                            yaxis_title=y_col,
                            hovermode="closest"
                        )
                    else:
                        temp_df["_box_single_group"] = y_col
                        fig = px.box(
                            temp_df,
                            x="_box_single_group",
                            y=y_col,
                            points="all",
                            hover_data=[col for col in temp_df.columns if col != "_box_single_group"],
                            title=f"Box Plot of {y_col}"
                        )
                        fig.update_layout(
                            xaxis_title="",
                            yaxis_title=y_col,
                            hovermode="closest",
                            showlegend=False
                        )
                        fig.update_xaxes(showticklabels=True)

                    st.plotly_chart(fig, use_container_width=True)
                    set_last_generated_plotly_chart(fig, f"Box Plot of {y_col}", "Box Plot")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if color_col:
                        grouped_data = []
                        labels = []

                        for grp_name, grp_data in temp_df.groupby(color_col):
                            vals = grp_data[y_col].dropna().values
                            if len(vals) > 0:
                                grouped_data.append(vals)
                                labels.append(str(grp_name))

                        if not grouped_data:
                            st.warning("No grouped data available for box plot.")
                            st.stop()

                        ax.boxplot(grouped_data, tick_labels=labels)
                        ax.set_xlabel(color_col)
                        ax.set_xticks(range(1, len(labels) + 1))
                        ax.set_xticklabels(labels, rotation=45, ha="right")
                    else:
                        ax.boxplot(temp_df[y_col].dropna().values)
                        ax.set_xticks([1])
                        ax.set_xticklabels([y_col])

                    ax.set_title(f"Box Plot of {y_col}")
                    ax.set_ylabel(y_col)
                    plt.tight_layout()
                    st.pyplot(fig)
                    set_last_generated_matplotlib_chart(fig, f"Box Plot of {y_col}", "Box Plot")

            # =====================================================
            # 3. SCATTER PLOT
            # =====================================================
            elif plot_type == "Scatter Plot":
                temp_df = plot_df.copy()
                temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors="coerce")
                temp_df[x_col], x_kind = prepare_x_series(temp_df[x_col])
                temp_df = temp_df.dropna(subset=[x_col, y_col])

                if temp_df.empty:
                    st.warning("No valid data to plot.")
                    st.stop()

                if enable_hover:
                    if color_col:
                        fig = px.scatter(
                            temp_df,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            hover_data=temp_df.columns.tolist(),
                            title=f"Scatter Plot: {y_col} vs {x_col}"
                        )
                    else:
                        fig = px.scatter(
                            temp_df,
                            x=x_col,
                            y=y_col,
                            hover_data=temp_df.columns.tolist(),
                            title=f"Scatter Plot: {y_col} vs {x_col}"
                        )

                    if x_kind == "category":
                        unique_x = temp_df[x_col].drop_duplicates().tolist()
                        if len(unique_x) <= 30:
                            fig.update_xaxes(
                                type="category",
                                tickmode="array",
                                tickvals=unique_x,
                                ticktext=[str(v) for v in unique_x]
                            )
                        else:
                            fig.update_xaxes(type="category")

                    fig.update_layout(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        hovermode="closest"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    set_last_generated_plotly_chart(fig, f"Scatter Plot: {y_col} vs {x_col}", "Scatter Plot")

                else:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if color_col:
                        for grp_name, grp_data in temp_df.groupby(color_col):
                            ax.scatter(grp_data[x_col], grp_data[y_col], label=str(grp_name), alpha=0.7)
                        ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc="upper left")
                    else:
                        ax.scatter(temp_df[x_col], temp_df[y_col], alpha=0.7)

                    ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)

                    if x_kind == "category":
                        unique_x = temp_df[x_col].drop_duplicates().tolist()
                        if len(unique_x) <= 20:
                            ax.set_xticks(range(len(unique_x)))
                            ax.set_xticklabels([str(v) for v in unique_x], rotation=45, ha="right")
                    else:
                        plt.xticks(rotation=45, ha="right")

                    plt.tight_layout()
                    st.pyplot(fig)
                    set_last_generated_matplotlib_chart(fig, f"Scatter Plot: {y_col} vs {x_col}", "Scatter Plot")

            # =====================================================
            # 4. LINE CHART
            # =====================================================
            elif plot_type == "Line Chart":
                temp_df = plot_df.copy()
                temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors="coerce")
                temp_df[x_col], x_kind = prepare_x_series(temp_df[x_col])
                temp_df = temp_df.dropna(subset=[x_col, y_col])

                if temp_df.empty:
                    st.warning("No valid data to plot.")
                    st.stop()

                if agg_option and agg_option != "None":
                    if color_col:
                        grouped = temp_df.groupby([x_col, color_col])[y_col]
                    else:
                        grouped = temp_df.groupby(x_col)[y_col]

                    if agg_option == "sum":
                        temp_df = grouped.sum().reset_index()
                    elif agg_option == "mean":
                        temp_df = grouped.mean().reset_index()
                    elif agg_option == "count":
                        temp_df = grouped.count().reset_index()
                    elif agg_option == "median":
                        temp_df = grouped.median().reset_index()

                temp_df = temp_df.sort_values(x_col)

                if enable_hover:
                    if color_col and color_col in temp_df.columns:
                        fig = px.line(
                            temp_df,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            markers=True,
                            hover_data=temp_df.columns.tolist(),
                            title=f"Line Chart: {y_col} over {x_col}"
                        )
                    else:
                        fig = px.line(
                            temp_df,
                            x=x_col,
                            y=y_col,
                            markers=True,
                            hover_data=temp_df.columns.tolist(),
                            title=f"Line Chart: {y_col} over {x_col}"
                        )

                    if x_kind == "category":
                        unique_x = temp_df[x_col].drop_duplicates().tolist()
                        if len(unique_x) <= 30:
                            fig.update_xaxes(
                                type="category",
                                tickmode="array",
                                tickvals=unique_x,
                                ticktext=[str(v) for v in unique_x]
                            )
                        else:
                            fig.update_xaxes(type="category")

                    fig.update_layout(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        hovermode="closest"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    set_last_generated_plotly_chart(fig, f"Line Chart: {y_col} over {x_col}", "Line Chart")

                else:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if color_col and color_col in temp_df.columns:
                        for grp_name, grp_data in temp_df.groupby(color_col):
                            grp_data = grp_data.sort_values(x_col)
                            ax.plot(grp_data[x_col], grp_data[y_col], marker="o", label=str(grp_name))
                        ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc="upper left")
                    else:
                        ax.plot(temp_df[x_col], temp_df[y_col], marker="o")

                    ax.set_title(f"Line Chart: {y_col} over {x_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)

                    if x_kind == "category":
                        unique_x = temp_df[x_col].drop_duplicates().tolist()
                        if len(unique_x) <= 20:
                            ax.set_xticks(range(len(unique_x)))
                            ax.set_xticklabels([str(v) for v in unique_x], rotation=45, ha="right")
                    else:
                        plt.xticks(rotation=45, ha="right")

                    plt.tight_layout()
                    st.pyplot(fig)
                    set_last_generated_matplotlib_chart(fig, f"Line Chart: {y_col} over {x_col}", "Line Chart")

            # =====================================================
            # 5. BAR CHART
            # =====================================================
            elif plot_type == "Bar Chart":
                temp_df = plot_df.copy()
                temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors="coerce")
                temp_df[x_col], x_kind = prepare_x_series(temp_df[x_col])
                temp_df = temp_df.dropna(subset=[x_col, y_col])

                if temp_df.empty:
                    st.warning("No valid data to plot.")
                    st.stop()

                if agg_option == "None" or agg_option is None:
                    agg_option = "sum"

                if color_col and color_col != x_col:
                    grouped = temp_df.groupby([x_col, color_col])[y_col]

                    if agg_option == "sum":
                        bar_df = grouped.sum().reset_index()
                    elif agg_option == "mean":
                        bar_df = grouped.mean().reset_index()
                    elif agg_option == "count":
                        bar_df = grouped.count().reset_index()
                    elif agg_option == "median":
                        bar_df = grouped.median().reset_index()

                    totals = (
                        bar_df.groupby(x_col)[y_col]
                        .sum()
                        .sort_values(ascending=False)
                        .head(int(top_n))
                    )

                    bar_df = bar_df[bar_df[x_col].isin(totals.index)].copy()

                    if enable_hover:
                        fig = px.bar(
                            bar_df,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            barmode="group",
                            hover_data=bar_df.columns.tolist(),
                            title=f"Bar Chart: {agg_option} of {y_col} by {x_col}"
                        )

                        shown_x = totals.index.tolist()
                        fig.update_xaxes(
                            type="category",
                            categoryorder="array",
                            categoryarray=shown_x,
                            tickmode="array",
                            tickvals=shown_x,
                            ticktext=[str(v) for v in shown_x]
                        )

                        fig.update_layout(
                            xaxis_title=x_col,
                            yaxis_title=f"{agg_option} of {y_col}",
                            hovermode="closest"
                        )

                        st.plotly_chart(fig, use_container_width=True)
                        set_last_generated_plotly_chart(fig, f"Bar Chart: {agg_option} of {y_col} by {x_col}", "Bar Chart")

                    else:
                        pivot_df = bar_df.pivot(index=x_col, columns=color_col, values=y_col).fillna(0)
                        pivot_df = pivot_df.loc[totals.index]

                        fig, ax = plt.subplots(figsize=(10, 6))
                        pivot_df.plot(kind="bar", ax=ax)

                        ax.set_title(f"Bar Chart: {agg_option} of {y_col} by {x_col}")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(f"{agg_option} of {y_col}")
                        ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc="upper left")
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig)
                        set_last_generated_matplotlib_chart(fig, f"Bar Chart: {agg_option} of {y_col} by {x_col}", "Bar Chart")

                else:
                    grouped = temp_df.groupby(x_col)[y_col]

                    if agg_option == "sum":
                        bar_df = grouped.sum().reset_index()
                    elif agg_option == "mean":
                        bar_df = grouped.mean().reset_index()
                    elif agg_option == "count":
                        bar_df = grouped.count().reset_index()
                    elif agg_option == "median":
                        bar_df = grouped.median().reset_index()

                    bar_df = bar_df.sort_values(y_col, ascending=False).head(int(top_n)).copy()

                    if enable_hover:
                        fig = px.bar(
                            bar_df,
                            x=x_col,
                            y=y_col,
                            hover_data=bar_df.columns.tolist(),
                            title=f"Bar Chart: {agg_option} of {y_col} by {x_col}"
                        )

                        shown_x = bar_df[x_col].tolist()
                        fig.update_xaxes(
                            type="category",
                            tickmode="array",
                            tickvals=shown_x,
                            ticktext=[str(v) for v in shown_x]
                        )

                        fig.update_layout(
                            xaxis_title=x_col,
                            yaxis_title=f"{agg_option} of {y_col}",
                            hovermode="closest"
                        )

                        st.plotly_chart(fig, use_container_width=True)
                        set_last_generated_plotly_chart(fig, f"Bar Chart: {agg_option} of {y_col} by {x_col}", "Bar Chart")

                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(bar_df[x_col].astype(str), bar_df[y_col])

                        ax.set_title(f"Bar Chart: {agg_option} of {y_col} by {x_col}")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(f"{agg_option} of {y_col}")
                        ax.set_xticks(range(len(bar_df[x_col])))
                        ax.set_xticklabels(bar_df[x_col].astype(str), rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig)
                        set_last_generated_matplotlib_chart(fig, f"Bar Chart: {agg_option} of {y_col} by {x_col}", "Bar Chart")

            # =====================================================
            # 6. HEATMAP / CORRELATION MATRIX
            # =====================================================
            elif plot_type == "Heatmap / Correlation Matrix":
                if len(numeric_cols) < 2:
                    st.error("Heatmap requires at least two numeric columns.")
                    st.stop()

                corr_df = plot_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
                corr = corr_df.corr()

                if corr.empty:
                    st.warning("No valid numeric correlation matrix available.")
                    st.stop()

                if enable_hover:
                    fig = px.imshow(
                        corr,
                        text_auto=".2f",
                        aspect="auto",
                        title="Correlation Matrix"
                    )
                    fig.update_layout(hovermode="closest")
                    st.plotly_chart(fig, use_container_width=True)
                    set_last_generated_plotly_chart(fig, "Correlation Matrix", "Heatmap / Correlation Matrix")
                else:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(corr, aspect="auto")

                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=90)
                    ax.set_yticks(range(len(corr.index)))
                    ax.set_yticklabels(corr.index)
                    ax.set_title("Correlation Matrix")

                    fig.colorbar(im, ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    set_last_generated_matplotlib_chart(fig, "Correlation Matrix", "Heatmap / Correlation Matrix")

        except Exception as e:
            st.error(f"An error occurred while generating the chart: {e}")
    # =========================================================
