from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from app.ingestion.github_loader import GitHubRepositoryLoader
from app.pipeline.rag_pipeline import RAGPipeline
from UI.styles import apply_styles


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CodeRAG",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()


# ============================================================
# SESSION STATE
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "repository_info" not in st.session_state:
    st.session_state.repository_info = None

if "indexing_status" not in st.session_state:
    st.session_state.indexing_status = "Not Indexed"

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_repository_name(repo_url: str) -> str:
    """
    Extract repository name from a GitHub URL.
    """

    parsed = urlparse(repo_url)

    path = parsed.path.strip("/")

    if not path:
        return ""

    repo_name = path.split("/")[-1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    return repo_name


def is_valid_github_url(repo_url: str) -> bool:
    """
    Basic GitHub URL validation.
    """

    try:
        parsed = urlparse(repo_url)

        return (
            parsed.scheme in {"http", "https"}
            and parsed.netloc.lower() == "github.com"
            and len(parsed.path.strip("/").split("/")) >= 2
        )

    except Exception:
        return False


def render_sidebar():
    """
    Render CodeRAG sidebar.
    """

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="sidebar-logo-icon">⌘</div>
                <div class="sidebar-logo-text">CodeRAG</div>
            </div>

            <div class="sidebar-description">
                Ask questions about<br>
                GitHub repositories<br>
                using RAG
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "🏠  Home",
                "💬  Ask Questions",
                "📊  Evaluation",
                "⚙️  Settings",
            ],
            label_visibility="collapsed",
        )

        if page.startswith("🏠"):
            st.session_state.current_page = "Home"

        elif page.startswith("💬"):
            st.session_state.current_page = "Ask Questions"

        elif page.startswith("📊"):
            st.session_state.current_page = "Evaluation"

        elif page.startswith("⚙️"):
            st.session_state.current_page = "Settings"


# ============================================================
# SIDEBAR
# ============================================================

render_sidebar()


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.current_page == "Home":

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.markdown(
        '<div class="page-title">GitHub Repository</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Enter a public GitHub repository URL to index Python code."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Repository URL + Button
    # --------------------------------------------------------

    input_col, button_col = st.columns([5, 1.25])

    with input_col:

        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repository",
            label_visibility="collapsed",
        )

    with button_col:

        index_clicked = st.button(
            "Index Repository",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Index Repository
    # --------------------------------------------------------

    if index_clicked:

        repo_url = repo_url.strip()

        if not repo_url:

            st.error("Please enter a GitHub repository URL.")

        elif not is_valid_github_url(repo_url):

            st.error(
                "Please enter a valid public GitHub repository URL."
            )

        else:

            repo_name = get_repository_name(repo_url)

            if not repo_name:

                st.error(
                    "Could not determine the repository name."
                )

            else:

                st.session_state.indexing_status = "In Progress"

                # ------------------------------------------------
                # Progress section
                # ------------------------------------------------

                progress_container = st.container()

                with progress_container:

                    st.markdown(
                        """
                        <div class="card">
                            <div style="
                                font-size:16px;
                                font-weight:700;
                                color:#173F63;
                                margin-bottom:12px;
                            ">
                                🔵 Indexing in progress...
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    progress_bar = st.progress(0)

                    status_text = st.empty()

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                # ------------------------------------------------
                # Checklist placeholders
                # ------------------------------------------------

                checklist_container = st.container()

                with checklist_container:

                    st.markdown(
                        """
                        <div class="card">
                        """,
                        unsafe_allow_html=True,
                    )

                    step_clone = st.empty()
                    step_files = st.empty()
                    step_chunks = st.empty()
                    step_embeddings = st.empty()
                    step_faiss = st.empty()
                    step_bm25 = st.empty()

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                try:

                    # --------------------------------------------
                    # Step 1: Clone
                    # --------------------------------------------

                    status_text.info(
                        "Cloning repository..."
                    )

                    step_clone.markdown(
                        '<div class="check-item">'
                        '<span class="check-active">◉</span>'
                        "Cloning repository..."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    progress_bar.progress(10)

                    loader = GitHubRepositoryLoader()

                    repo_path = loader.clone_repository(
                        repo_url,
                        repo_name,
                    )

                    step_clone.markdown(
                        '<div class="check-item">'
                        '<span class="check-done">●</span>'
                        "Cloning repository"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    progress_bar.progress(20)

                    # --------------------------------------------
                    # Step 2: Find Python files
                    # --------------------------------------------

                    status_text.info(
                        "Parsing Python files..."
                    )

                    step_files.markdown(
                        '<div class="check-item">'
                        '<span class="check-active">◉</span>'
                        "Parsing Python files..."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    python_files = loader.get_python_files(
                        repo_path
                    )

                    step_files.markdown(
                        '<div class="check-item">'
                        '<span class="check-done">●</span>'
                        f"Parsing Python files ({len(python_files)} found)"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    progress_bar.progress(30)

                    # --------------------------------------------
                    # Step 3: Pipeline initialization
                    # --------------------------------------------

                    status_text.info(
                        "Creating code chunks..."
                    )

                    step_chunks.markdown(
                        '<div class="check-item">'
                        '<span class="check-active">◉</span>'
                        "Creating code chunks..."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    pipeline = RAGPipeline()

                    progress_bar.progress(40)

                    # --------------------------------------------
                    # Step 4: Index repository
                    # --------------------------------------------

                    status_text.info(
                        "Generating embeddings..."
                    )

                    step_embeddings.markdown(
                        '<div class="check-item">'
                        '<span class="check-active">◉</span>'
                        "Generating embeddings (BAAI/bge-small-en-v1.5)..."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    index_dir = (
                        Path("data")
                        / "indexes"
                        / repo_name
                    )

                    result = pipeline.index_repository(
                        repo_path=repo_path,
                        index_dir=index_dir,
                    )

                    step_chunks.markdown(
                        '<div class="check-item">'
                        '<span class="check-done">●</span>'
                        f"Creating code chunks ({result['chunks']} created)"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    step_embeddings.markdown(
                        '<div class="check-item">'
                        '<span class="check-done">●</span>'
                        "Generating embeddings (BAAI/bge-small-en-v1.5)"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    progress_bar.progress(70)

                    # --------------------------------------------
                    # FAISS
                    # --------------------------------------------

                    status_text.info(
                        "Building FAISS index..."
                    )

                    step_faiss.markdown(
                        '<div class="check-item">'
                        '<span class="check-active">◉</span>'
                        "Building FAISS index..."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # FAISS has already been created by
                    # pipeline.index_repository().
                    step_faiss.markdown(
                        '<div class="check-item">'
                        '<span class="check-done">●</span>'
                        "Building FAISS index"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    progress_bar.progress(85)

                    # --------------------------------------------
                    # BM25
                    # --------------------------------------------

                    status_text.info(
                        "Building BM25 index..."
                    )

                    step_bm25.markdown(
                        '<div class="check-item">'
                        '<span class="check-active">◉</span>'
                        "Building BM25 index..."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # BM25 is created inside RAGPipeline.
                    step_bm25.markdown(
                        '<div class="check-item">'
                        '<span class="check-done">●</span>'
                        "Building BM25 index"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    progress_bar.progress(100)

                    status_text.success(
                        "Repository indexing completed successfully."
                    )

                    # --------------------------------------------
                    # Save pipeline
                    # --------------------------------------------

                    st.session_state.pipeline = pipeline

                    st.session_state.repository_info = {
                        "name": repo_name,
                        "url": repo_url,
                        "python_files": result["python_files"],
                        "chunks": result["chunks"],
                        "vectors": result["vectors"],
                        "index_dir": str(index_dir),
                    }

                    st.session_state.indexing_status = "Ready"

                    st.success(
                        f"Successfully indexed **{repo_name}**."
                    )

                except Exception as exc:

                    st.session_state.indexing_status = "Error"

                    st.error(
                        f"Repository indexing failed: {exc}"
                    )


    # --------------------------------------------------------
    # Guidance Card
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="yellow-card">
            <div style="
                font-size:15px;
                font-weight:700;
                color:#594900;
                margin-bottom:5px;
            ">
                1. Enter GitHub Repository URL
            </div>

            <div style="
                color:#75672A;
                font-size:13px;
            ">
                Enter a public GitHub repository URL to index
                Python code.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------
    # Repository Information
    # --------------------------------------------------------

    st.markdown(
        '<div class="repo-info-title">▣ Repository Information</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.repository_info:

        info = st.session_state.repository_info

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="metric-label">Repository</div>
                    <div class="metric-value">
                        {info["name"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="metric-label">Python Files</div>
                    <div class="metric-value">
                        {info["python_files"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="metric-label">Total Chunks</div>
                    <div class="metric-value">
                        {info["chunks"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:

            status_class = (
                "status-ready"
                if st.session_state.indexing_status == "Ready"
                else "status-not-ready"
            )

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="metric-label">Index Status</div>
                    <div style="margin-top:8px;">
                        <span class="{status_class}">
                            {st.session_state.indexing_status}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                """
                <div class="info-card">
                    <div class="metric-label">Repository</div>
                    <div class="metric-value">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                """
                <div class="info-card">
                    <div class="metric-label">Python Files</div>
                    <div class="metric-value">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:

            st.markdown(
                """
                <div class="info-card">
                    <div class="metric-label">Total Chunks</div>
                    <div class="metric-value">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:

            st.markdown(
                """
                <div class="info-card">
                    <div class="metric-label">Index Status</div>

                    <div style="margin-top:8px;">
                        <span class="status-not-ready">
                            Not Indexed
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PLACEHOLDER PAGES
# ============================================================

elif st.session_state.current_page == "Ask Questions":

    st.markdown(
        '<div class="page-title">Ask Questions</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The question-answering interface will be added next."
    )


elif st.session_state.current_page == "Evaluation":

    st.markdown(
        '<div class="page-title">Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The RAG evaluation dashboard will be added later."
    )


elif st.session_state.current_page == "Settings":

    st.markdown(
        '<div class="page-title">Settings</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Pipeline and model settings will be added later."
    )