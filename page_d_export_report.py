from shared_core import *


def render_page_d():
    st.title("Export & Report")

    if st.session_state.working_df is None:
        st.warning("Please upload a dataset first on Page A.")
    else:
        df = st.session_state.working_df.copy()
        profile_info = profile_dataset_cached(df)

        render_workflow_controls(show_metrics=True)

        total_cells = max(df.shape[0] * df.shape[1], 1)
        completeness_pct = round((1 - (profile_info["missing_cells"] / total_cells)) * 100, 2)
        numeric_count = int(len(df.select_dtypes(include=np.number).columns))
        categorical_count = int(len(df.select_dtypes(include=["object", "category", "bool"]).columns))
        validation_rows = 0 if st.session_state.validation_violations_df is None else len(st.session_state.validation_violations_df)

        st.markdown("""
        <style>
        .dashboard-hero {
            padding: 1.1rem 1.25rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
            color: white;
            margin: 0.5rem 0 1rem 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
        }
        .dashboard-card {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            color: white;
            min-height: 96px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.75rem;
        }
        .card-blue { background: linear-gradient(135deg, #2563eb, #38bdf8); }
        .card-violet { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
        .card-amber { background: linear-gradient(135deg, #d97706, #f59e0b); }
        .card-rose { background: linear-gradient(135deg, #dc2626, #fb7185); }
        .card-slate { background: linear-gradient(135deg, #334155, #64748b); }
        .dashboard-card h4, .dashboard-card p { margin: 0; }
        .dashboard-section-note {
            color: #475569;
            margin-top: -0.25rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            """
            <div class="dashboard-hero">
                <h2 style="margin:0;">Final Dashboard</h2>
                <p style="margin:0.35rem 0 0 0; opacity:0.96;">A final report view of the cleaned dataset, saved charts, workflow history, and export-ready outputs.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        hero1, hero2, hero3, hero4 = st.columns(4)
        hero1.markdown(f"<div class='dashboard-card card-blue'><h4>Rows</h4><p style='font-size:1.8rem;font-weight:700;'>{profile_info['rows']:,}</p></div>", unsafe_allow_html=True)
        hero2.markdown(f"<div class='dashboard-card card-violet'><h4>Columns</h4><p style='font-size:1.8rem;font-weight:700;'>{profile_info['columns']:,}</p></div>", unsafe_allow_html=True)
        hero3.markdown(f"<div class='dashboard-card card-amber'><h4>Missing Cells</h4><p style='font-size:1.8rem;font-weight:700;'>{profile_info['missing_cells']:,}</p></div>", unsafe_allow_html=True)
        hero4.markdown(f"<div class='dashboard-card card-rose'><h4>Duplicate Rows</h4><p style='font-size:1.8rem;font-weight:700;'>{profile_info['duplicate_rows']:,}</p></div>", unsafe_allow_html=True)

        extra1, extra2, extra3, extra4 = st.columns(4)
        extra1.markdown(f"<div class='dashboard-card card-slate'><h4>Completeness %</h4><p style='font-size:1.8rem;font-weight:700;'>{completeness_pct}</p></div>", unsafe_allow_html=True)
        extra2.markdown(f"<div class='dashboard-card card-slate'><h4>Numeric Columns</h4><p style='font-size:1.8rem;font-weight:700;'>{numeric_count}</p></div>", unsafe_allow_html=True)
        extra3.markdown(f"<div class='dashboard-card card-slate'><h4>Categorical Columns</h4><p style='font-size:1.8rem;font-weight:700;'>{categorical_count}</p></div>", unsafe_allow_html=True)
        extra4.markdown(f"<div class='dashboard-card card-slate'><h4>Validation Rows Saved</h4><p style='font-size:1.8rem;font-weight:700;'>{validation_rows}</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.write("### Saved Dashboard Charts")
        st.caption("Save charts from Page C and they will appear here as part of the final dashboard.")

        saved_dashboard_charts = st.session_state.dashboard_saved_charts
        dashboard_bundle_title = st.text_input(
            "Title for all saved charts download",
            value="Saved Dashboard Charts",
            key="dashboard_bundle_title"
        )
        dashboard_action_col1, dashboard_action_col2, dashboard_action_col3 = st.columns([1, 1, 1])
        with dashboard_action_col1:
            if st.button("Clear saved dashboard charts", use_container_width=True, key="clear_dashboard_charts_btn"):
                st.session_state.dashboard_saved_charts = []
                st.success("Saved dashboard charts were cleared.")
                st.rerun()
        with dashboard_action_col2:
            st.metric("Charts saved", len(saved_dashboard_charts))
        with dashboard_action_col3:
            if saved_dashboard_charts:
                bundle_data, bundle_file_name, bundle_mime = build_saved_charts_bundle_download(
                    saved_dashboard_charts,
                    dashboard_bundle_title
                )
                st.download_button(
                    "Download all saved charts",
                    data=bundle_data,
                    file_name=bundle_file_name,
                    mime=bundle_mime,
                    key="download_all_saved_dashboard_charts",
                    use_container_width=True
                )

        if saved_dashboard_charts:
            chart_columns = st.columns(2)
            for idx, saved_chart in enumerate(saved_dashboard_charts):
                with chart_columns[idx % 2]:
                    st.markdown(f"<div class='dashboard-card card-slate'><h4>{saved_chart.get('title', 'Saved chart')}</h4><p>{saved_chart.get('chart_type', '')} • saved {saved_chart.get('saved_at', '')}</p></div>", unsafe_allow_html=True)
                    if saved_chart.get("render_type") == "plotly":
                        try:
                            st.plotly_chart(pio.from_json(saved_chart["plotly_json"]), use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not reload this saved plotly chart: {e}")
                    elif saved_chart.get("render_type") == "matplotlib":
                        st.image(saved_chart.get("image_bytes"), use_container_width=True)
                    download_title = st.text_input(
                        "Title before downloading",
                        value=saved_chart.get("title", f"Saved chart {idx + 1}"),
                        key=f"dashboard_download_title_{idx}"
                    )
                    download_data, download_file_name, download_mime = build_saved_chart_download(saved_chart, download_title)
                    st.download_button(
                        "Download this saved chart",
                        data=download_data,
                        file_name=download_file_name,
                        mime=download_mime,
                        key=f"download_dashboard_chart_{idx}",
                        use_container_width=True
                    )
                    remove_key = f"remove_dashboard_chart_{idx}"
                    if st.button("Remove this chart", key=remove_key, use_container_width=True):
                        st.session_state.dashboard_saved_charts.pop(idx)
                        st.rerun()
        else:
            st.info("No charts have been saved from Page C yet. Generate a chart there and click 'Add last chart to dashboard'.")

        st.markdown("---")
        st.write("### Cleaned Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("### Transformation Log")
        if st.session_state.transformation_log:
            log_df = pd.DataFrame(st.session_state.transformation_log)

            display_log_df = log_df.copy()
            if "parameters" in display_log_df.columns:
                display_log_df["parameters"] = display_log_df["parameters"].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
                )
            if "affected_columns" in display_log_df.columns:
                display_log_df["affected_columns"] = display_log_df["affected_columns"].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                )

            st.dataframe(display_log_df, use_container_width=True)

        else:
            st.info("No transformation steps recorded yet.")

        st.write("### Current Data Quality Snapshot")
        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Missing Cells", f"{profile_info['missing_cells']:,}")
        dq2.metric("Duplicate Rows", f"{profile_info['duplicate_rows']:,}")
        dq3.metric("Validation Rows Saved", validation_rows)
        dq4.metric("Logged Steps", len(st.session_state.transformation_log))

        st.markdown("---")
        st.write("### Download Cleaned Data")

        csv_data = get_download_csv(df)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        excel_data = get_download_excel(df)
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name="cleaned_dataset_with_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        dashboard_html = get_dashboard_html(df, profile_info)
        st.download_button(
            label="Download dashboard as HTML",
            data=dashboard_html,
            file_name="final_dashboard_report.html",
            mime="text/html"
        )

        if st.session_state.validation_violations_df is not None and not st.session_state.validation_violations_df.empty:
            st.write("### Download Latest Validation Violations")
            violations_csv = st.session_state.validation_violations_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download validation violations as CSV",
                data=violations_csv,
                file_name="validation_violations.csv",
                mime="text/csv"
            )
