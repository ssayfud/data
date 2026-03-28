from shared_core import *


def render_page_a():
    st.title("AI-Assisted Data Wrangler & Visualizer")
    st.caption("A mini data preparation studio for uploading, profiling, cleaning, visualizing, and exporting datasets.")
    st.subheader("Upload & Overview")
    st.write("Upload a dataset in CSV, Excel, or JSON format, or load a Google Sheet.")

    tab1, tab2 = st.tabs(["Upload File", "Google Sheets"])
 
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "json"]
        )

        if uploaded_file is not None:
            df = load_file(uploaded_file)

            if df is not None:
                st.session_state.original_df = df.copy()
                st.session_state.working_df = df.copy()
                st.session_state.file_name = uploaded_file.name
                st.session_state.transformation_log = []
                st.session_state.recipe_steps = []
                st.session_state.history_stack = []

                log_step("dataset_loaded", {
                    "source": "file_upload",
                    "file_name": uploaded_file.name
                }, affected_columns=[])

                st.success("Dataset loaded successfully.")
            else:
                st.error("Could not read the file. Please check the format.")

    with tab2:
        sheet_url = st.text_input(
            "Paste Google Sheets link",
            placeholder="https://docs.google.com/spreadsheets/d/..."
        )

        load_sheet_btn = st.button("Load Google Sheet")

        if sheet_url and load_sheet_btn:
            df = load_google_sheet(sheet_url)

            if df is not None:
                st.session_state.original_df = df.copy()
                st.session_state.working_df = df.copy()
                st.session_state.file_name = "Google Sheet"
                st.session_state.transformation_log = []
                st.session_state.recipe_steps = []
                st.session_state.history_stack = []

                log_step("dataset_loaded", {
                    "source": "google_sheets",
                    "url": sheet_url
                }, affected_columns=[])

                st.success("Google Sheet loaded successfully.")
            else:
                st.error("Could not load Google Sheet. Make sure the link is valid and publicly accessible.")

    df = st.session_state.get("working_df")

    if df is not None:
        st.markdown("---")
        st.header("Dataset Overview")

        profile_info = profile_dataset_cached(df)

        rows = profile_info["rows"]
        cols = profile_info["columns"]
        total_missing = profile_info["missing_cells"]
        duplicate_count = profile_info["duplicate_rows"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{rows:,}")
        col2.metric("Columns", f"{cols:,}")
        col3.metric("Missing Cells", f"{total_missing:,}")
        col4.metric("Duplicate Rows", f"{duplicate_count:,}")

        st.write(f"**Source:** {st.session_state.file_name}")

        st.write("### Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("### Column Names & Inferred Data Types")
        info_df = profile_info["dtype_info"]
        st.dataframe(info_df, use_container_width=True)

        st.write("### Missing Values by Column")
        missing_df = profile_info["missing_by_column"]
        st.dataframe(missing_df, use_container_width=True)

        st.write("### Duplicate Count")
        st.write(f"Total duplicate rows: **{duplicate_count:,}**")

        st.write("### Numeric Summary Statistics")
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().transpose(), use_container_width=True)
        else:
            st.info("No numeric columns found.")

        st.write("### Categorical Summary Statistics")
        categorical_df = df.select_dtypes(include=["object", "category", "bool"])
        if not categorical_df.empty:
            st.dataframe(categorical_df.describe().transpose(), use_container_width=True)
        else:
            st.info("No categorical columns found.")
    else:
        st.info("Upload a file or load a Google Sheet to begin.")

    # =========================================================
