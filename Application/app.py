from streamlit_option_menu import option_menu
from .Agents.agent.py import run_robofund_agent
from .Helpers.helper_functions.py import get_data
from streamlit_chat import message
import sqlite3
import streamlit as st
import pandas as pd
import plotly.express as px
from tefas import Crawler
from datetime import datetime, timedelta
from streamlit_dynamic_filters import DynamicFilters


st.set_page_config(
    page_title="LLM-RoboFund",
    layout="wide",
    initial_sidebar_state="auto",
)

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Info", "LLM-RoboFund", "Dashboard"],
        icons=["info-circle", "robot", "bar-chart-line-fill"],
        menu_icon="cast",
        default_index=1,
    )

# Index-0; Project Info
if selected == "Info":
    st.markdown(
        "<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True
    )


# Index-1; LLM Robofund Chatbot
if selected == "LLM-RoboFund":
    st.markdown("# Accelerate Your Investment Research")
    st.markdown(
        "<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True
    )
    prompt = st.text_input(
        "Ask your question to Chatbot", placeholder="Enter your question here..."
    )
    if (
        "user_prompt_history" not in st.session_state
        and "chat_answer_history" not in st.session_state
    ):
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answer_history"] = []

    if prompt:
        with st.spinner("Please Wait While Your Response Is Being Generated..."):
            generated_response = run_robofund_agent(prompt)

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answer_history"].append(generated_response)

    if st.session_state["chat_answer_history"]:
        for user_query, answer in zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answer_history"],
        ):
            message(user_query, is_user=True, allow_html=True)
            message(answer)


# Index-2; TEFAS Database
if selected == "Dashboard":
    conn = sqlite3.connect("Data/SQLiteDB/tefas.db")
    curr = conn.cursor()
    datatable = get_data(cursor=curr, table_name="tefastable")
    dataframe = pd.DataFrame(
        data=datatable,
        columns=[
            "Fund_Code",
            "Fund_Name",
            "Rainbow_Fund_Type",
            "monthly_return",
            "monthly_3_return",
            "monthly_6_return",
            "since_jan",
            "annual_1_return",
            "annual_3_return",
            "annual_5_return",
            "applied_management_fee",
            "bylaw_management_fee",
            "annual_realized_return_rate",
            "max_total_expense_ratio",
            "init_fund_size",
            "current_fund_size",
            "portfolio_size_change",
            "init_out_shares",
            "current_out_shares",
            "change_in_nshares",
            "realized_return_rate",
        ],
    )

    st.title(" :chart_with_upwards_trend: TEFAS Dashboard")
    st.markdown(
        "<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True
    )

    # monthly_return value
    best_performer = (
        dataframe[["Fund_Code", "monthly_return"]]
        .sort_values(by="monthly_return", ascending=False)
        .iloc[:1, 1]
        .values
    )
    # best fund_code
    best_fund = (
        dataframe[["Fund_Code", "monthly_return"]]
        .sort_values(by="monthly_return", ascending=False)
        .iloc[0, 0]
    )
    # Return Value
    worst_performer = (
        dataframe[["Fund_Code", "monthly_return"]]
        .sort_values(by="monthly_return", ascending=True)
        .iloc[:1, 1]
    )
    # worst fund_code
    worst_fund = (
        dataframe[["Fund_Code", "monthly_return"]]
        .sort_values(by="monthly_return", ascending=True)
        .iloc[0, 0]
    )

    # init tefas-crawler
    tefas = Crawler()
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = datetime.today() - timedelta(weeks=12)  # Available data up to 3 months

    # Best Fund Data
    metric_2_data = tefas.fetch(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date,
        name=f"{best_fund}",
        columns=["date", "price"],
    )
    # Worst Fund Data
    metric_4_data = tefas.fetch(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date,
        name=f"{worst_fund}",
        columns=["date", "price"],
    )

    # Metric Columns
    metric_1, metric_2, metric_3, metric_4 = st.columns(
        spec=[0.15, 0.35, 0.15, 0.35], gap="small"
    )
    # Metric 1: Best Performing Fund Monthly Return Rate (%)
    metric_1.metric(f"Best Fund ({best_fund})", best_performer, "Monthly Return (%)")
    # Metric 2: Best Performing Fund 3-Month Line Graph
    metric_2.plotly_chart(
        figure_or_data=px.line(
            metric_2_data,
            x="date",
            y="price",
            title=f"{best_fund} Fund 3-Month Performance",
            width=300,
            height=300,
        ),
        use_container_width=True,
    )
    # Metric 3: Worst Performing Fund Monthly Return Rate (%)
    metric_3.metric(
        f"Worst Fund ({worst_fund})",
        worst_performer,
        "- Monthly Return (%)",
    )
    # Metric 4: Worst Performing Fund Monthly Return Rate (%)
    metric_4.plotly_chart(
        figure_or_data=px.line(
            metric_4_data,
            x="date",
            y="price",
            title=f"{worst_fund} Fund 3-Month Performance",
            width=300,
            height=300,
        ),
        use_container_width=True,
    )
    # Bar & Pie Charts
    top_fund_columns, search_column = st.columns(spec=[0.5, 0.5], gap="large")

    # Bart Chart Column
    with top_fund_columns:
        with st.expander(
            "## Gainers (%) & Losers (%) in Monthly Returns (%)", expanded=True
        ):
            # Bar Chart 1: TOP 5 Funds
            st.bar_chart(
                data=dataframe[["Fund_Code", "monthly_return"]]
                .sort_values(by="monthly_return", ascending=False)
                .iloc[:5, :],
                x="Fund_Code",
                y="monthly_return",
                color="#34eb83",
                height=250,
                width=200,
            )
            # Bar Chart 2: Worst 5 Funds
            st.bar_chart(
                data=dataframe[["Fund_Code", "monthly_return"]]
                .sort_values(by="monthly_return", ascending=True)
                .iloc[:5, :],
                x="Fund_Code",
                y="monthly_return",
                color="#eb3434",
                height=250,
                width=200,
            )

    # Pie Chart Column
    with search_column:
        with st.expander("TEFAS Fund Size Distribution (%) ", expanded=True):
            fig_2 = px.pie(
                data_frame=dataframe,
                values="current_fund_size",
                names="Rainbow_Fund_Type",
                hole=0.4,
                height=521,
            )
            fig_2.update_traces(
                text=dataframe["Rainbow_Fund_Type"], textposition="outside"
            )
            st.plotly_chart(fig_2, use_container_width=True)

    # Datatable with fund type filter
    st.subheader("TEFAS Fund Datatable")

    data_filter = DynamicFilters(df=dataframe, filters=["Rainbow_Fund_Type"])
    with st.sidebar:
        data_filter.display_filters()
    data_filter.display_df()

    # Datatable with fund details
    st.subheader("Fund Details")
    conn2 = sqlite3.connect("Data/SQLiteDB/funddetails.db")
    curr2 = conn2.cursor()
    detail_data = get_data(cursor=curr2, table_name="detailtable")
    detaildf = pd.DataFrame(
        data=detail_data,
        columns=[
            "Date",
            "Fund Code",
            "Fund Title",
            "Stock (%)",
            "Government Bond (%)",
            "Treasury Bill (%)",
            "Government Currency Debt Sec. (%)",
            "Commercial Paper (%)",
            "Private Sector Bond (%)",
            "Asset-Backed Securities (%)",
            "Government Bonds and Bills (FX) (%)",
            "International Corporate Debt Sec. (%)",
            "Takasbank Money Market (%)",
            "Government Lease Certificate (TL) (%)",
            "Government Lease Certificate (FC) (%)",
            "Private Sector Lease Certificates (%)",
            "International Gov. Lease Certificates (%)",
            "International Cor. Lease Certificates (%)",
            "Deposit Account (Turkish Lira) (%)",
            "Deposit Account (FC) (%)",
            "Deposit Account (Gold) (%)",
            "Participation Account (TL) (%)",
            "Participation Account (FC) (%)",
            "Participation Account (Gold) (%)",
            "Repo (%)",
            "Reverse-Repo (%)",
            "Precious Metals (%)",
            "ETF Issued in Precious Metals (%)",
            "Government Debt Sec. Issued in Precious Metal (%)",
            "Gov. Lease Certificates Issued in Precious Metal (%)",
            "International Government Debt Sec. (%)",
            "Foreign Equity (%)",
            "International ETF (%)",
            "Investment Funds Participation Share (%)",
            "Exchange Traded Fund Participation Share (%)",
            "Real Estate I. Fund Participation Share (%)",
            "Venture Capital I. Fund Participation Share (%)",
            "Futures Contract Cash Collateral (%)",
        ],
    )
    st.dataframe(detaildf)
    with st.sidebar:
        selected_fund_code = st.multiselect(
            "Select Fund Code", options=detaildf["Fund Code"]
        )

    if selected_fund_code:
        filtered_df = detaildf[detaildf["Fund Code"].isin(selected_fund_code)]

        # Select relevant columns for the pie chart
        pie_chart_data = filtered_df.iloc[:, 3:]
        st.subheader(f"Investment Distribution (%) of {selected_fund_code[0]} Fund")
        fig_3 = px.pie(
            data_frame=pie_chart_data,
            names=pie_chart_data.columns,
            values=pie_chart_data.iloc[
                0, :
            ].values,  # Selecting values for the first row
            hole=0.4,
            height=521,
        )
        st.plotly_chart(fig_3, use_container_width=True)
