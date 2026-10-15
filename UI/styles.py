import streamlit as st


def apply_styles():
    st.markdown(
        """
        <style>

        /* =========================================================
           GLOBAL
        ========================================================= */

        .stApp {
            background-color: #FFFFFF;
        }

        .main {
            background-color: #FFFFFF;
        }

        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF;
        }

        [data-testid="stHeader"] {
            background-color: #FFFFFF;
        }

        /* Remove excessive top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }


        /* =========================================================
           SIDEBAR
        ========================================================= */

        [data-testid="stSidebar"] {
            background-color: #F4F8FC;
            border-right: 1px solid #D9E5F2;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        .sidebar-logo {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
        }

        .sidebar-logo-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background-color: #0B4F8A;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 700;
        }

        .sidebar-logo-text {
            font-size: 21px;
            font-weight: 700;
            color: #123B5D;
        }

        .sidebar-description {
            color: #66788A;
            font-size: 12px;
            line-height: 1.5;
            margin-bottom: 22px;
        }


        /* =========================================================
           PAGE TITLE
        ========================================================= */

        .page-title {
            font-size: 30px;
            font-weight: 750;
            color: #173F63;
            margin-bottom: 3px;
        }

        .page-subtitle {
            font-size: 14px;
            color: #718096;
            margin-bottom: 20px;
        }


        /* =========================================================
           CARDS
        ========================================================= */

        .card {
            background-color: #FFFFFF;
            border: 1px solid #DCE7F1;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(21, 61, 94, 0.05);
        }

        .info-card {
            background-color: #F7FAFD;
            border: 1px solid #DCE7F1;
            border-radius: 12px;
            padding: 18px;
        }

        .yellow-card {
            background-color: #FFF5CF;
            border: 1px solid #F3E3A3;
            border-radius: 12px;
            padding: 16px;
        }


        /* =========================================================
           STEP CARD
        ========================================================= */

        .step-number {
            display: inline-flex;
            width: 30px;
            height: 30px;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background-color: #0B4F8A;
            color: white;
            font-weight: 700;
            margin-right: 8px;
        }

        .step-title {
            font-size: 15px;
            font-weight: 700;
            color: #173F63;
        }


        /* =========================================================
           REPOSITORY INFORMATION
        ========================================================= */

        .repo-info-title {
            color: #173F63;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 15px;
        }

        .metric-label {
            color: #718096;
            font-size: 12px;
        }

        .metric-value {
            color: #173F63;
            font-size: 17px;
            font-weight: 700;
        }


        /* =========================================================
           STATUS
        ========================================================= */

        .status-ready {
            display: inline-block;
            background-color: #DDF5E5;
            color: #18733B;
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .status-progress {
            display: inline-block;
            background-color: #FFF0B8;
            color: #806600;
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .status-not-ready {
            display: inline-block;
            background-color: #EEF2F6;
            color: #66788A;
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: 600;
        }


        /* =========================================================
           STREAMLIT INPUTS
        ========================================================= */

        .stTextInput label {
            color: #173F63 !important;
            font-weight: 600;
        }

        .stTextInput input {
            border: 1px solid #C9D8E6;
            border-radius: 8px;
            background-color: #FFFFFF;
            color: #173F63;
        }

        .stTextInput input:focus {
            border-color: #2878B5;
            box-shadow: 0 0 0 1px #2878B5;
        }


        /* =========================================================
           PRIMARY BUTTON
        ========================================================= */

        .stButton > button {
            background-color: #1769AA;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 9px 18px;
            font-weight: 650;
            min-height: 42px;
        }

        .stButton > button:hover {
            background-color: #0B4F8A;
            color: white;
        }


        /* =========================================================
           PROGRESS BAR
        ========================================================= */

        [data-testid="stProgressBar"] > div > div > div {
            background-color: #2478B8;
        }


        /* =========================================================
           CHECKLIST
        ========================================================= */

        .check-item {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
            color: #425466;
            font-size: 14px;
        }

        .check-done {
            color: #19A15F;
            font-weight: 700;
        }

        .check-active {
            color: #2478B8;
            font-weight: 700;
        }

        .check-pending {
            color: #A4B0BC;
            font-weight: 700;
        }


        /* =========================================================
           DIVIDER
        ========================================================= */

        hr {
            border: none;
            border-top: 1px solid #E2EAF2;
            margin: 20px 0;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )