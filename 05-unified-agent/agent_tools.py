"""
Tools for the unified agent
===========================

`build_tools(vectorstore, dataframe)` is called on every Streamlit rerun with
a *snapshot* of the current document and dataset (see app.py for why). The
tools close over these plain Python objects and never touch st.session_state,
which makes them safe to run in LangGraph's worker threads and easy to test.
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool, tool

from llm_series.config import tavily_available
from llm_series.documents import format_context
from llm_series.safe_math import calculate

TOP_K = 4
# Hard cap on rows returned to the model. A `value_counts` on a column with
# thousands of distinct values would otherwise flood the context window.
MAX_OUTPUT_ROWS = 50

Operation = Literal[
    "columns", "shape", "head", "describe",
    "mean", "median", "std", "min", "max", "sum", "count", "nunique",
    "value_counts", "groupby_mean",
]
FilterOperator = Literal["==", "!=", ">", "<", ">=", "<=", "contains"]

_AGGREGATIONS = ("mean", "median", "std", "min", "max", "sum", "count", "nunique")
_NUMERIC_ONLY = ("mean", "median", "std", "sum")


def _limit(obj: pd.Series | pd.DataFrame) -> str:
    """Render a pandas object, truncated to MAX_OUTPUT_ROWS with a note."""
    if len(obj) > MAX_OUTPUT_ROWS:
        return obj.head(MAX_OUTPUT_ROWS).to_string() + (
            f"\n... showing the first {MAX_OUTPUT_ROWS} of {len(obj)} rows."
        )
    return obj.to_string()


def _missing_column(name: str, df: pd.DataFrame) -> str:
    # Listing the real columns lets the model fix its argument and retry.
    return f"Column '{name}' not found. Available columns: {list(df.columns)}"


def _apply_filter(df: pd.DataFrame, column: str, op: str, value: str) -> pd.DataFrame | str:
    """Return the filtered frame, or an error message string."""
    if column not in df.columns:
        return _missing_column(column, df)
    series = df[column]
    target: object = value
    if op != "contains":
        # bool is checked first: pandas also counts bool columns as numeric.
        if pd.api.types.is_bool_dtype(series):
            target = value.strip().lower() in ("true", "1", "yes")
        elif pd.api.types.is_numeric_dtype(series):
            try:
                target = float(value)
            except ValueError:
                return f"Column '{column}' is numeric but the filter value '{value}' is not a number."
        elif op in (">", "<", ">=", "<="):
            return (
                f"Column '{column}' has type {series.dtype}; the operator '{op}' "
                "needs a numeric column. Use '==', '!=' or 'contains'."
            )

    if op == "==":
        mask = series == target
    elif op == "!=":
        mask = series != target
    elif op == ">":
        mask = series > target
    elif op == "<":
        mask = series < target
    elif op == ">=":
        mask = series >= target
    elif op == "<=":
        mask = series <= target
    else:  # contains
        mask = series.astype(str).str.contains(str(value), case=False, regex=False, na=False)
    return df[mask]


def build_tools(vectorstore, dataframe: pd.DataFrame | None) -> list[BaseTool]:
    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression and return the result.

        Supports + - * / // % and ** (or ^) for powers, parentheses, pi, e and
        the functions sqrt, log, log10, exp, sin, cos, tan, abs, round, floor, ceil."""
        return calculate(expression)

    @tool
    def document_search(query: str) -> str:
        """Search the PDF document the user uploaded and return the most
        relevant passages with their page numbers. Use it for questions about
        the content, data or text of that document. Write the query as a
        complete, standalone search phrase."""
        if vectorstore is None:
            return "No PDF document has been uploaded yet."
        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
        )
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant passage was found in the document."
        return format_context(docs)

    @tool
    def analyze_data(
        operation: Operation,
        column: Optional[str] = None,
        group_by: Optional[str] = None,
        n: int = 5,
        filter_column: Optional[str] = None,
        filter_operator: Optional[FilterOperator] = None,
        filter_value: Optional[str] = None,
    ) -> str:
        """Run a safe, predefined operation on the CSV dataset the user uploaded.
        It never executes arbitrary code.

        operation:
        - "columns": list columns and their types
        - "shape": number of rows and columns
        - "head": first n rows
        - "describe": statistical summary of numeric columns
        - "mean" / "median" / "std" / "min" / "max" / "sum" / "count" / "nunique":
          aggregate the given column
        - "value_counts": frequency of each value in column
        - "groupby_mean": mean of column for each value of group_by

        Optional filter, applied before the operation: filter_column,
        filter_operator and filter_value. Start with "columns" if you do not
        know the column names."""
        if dataframe is None:
            return "No CSV dataset has been uploaded yet."
        df = dataframe

        filter_args = (filter_column, filter_operator, filter_value)
        if any(a is not None for a in filter_args):
            if not all(a is not None for a in filter_args):
                return "A filter needs all three of filter_column, filter_operator and filter_value."
            filtered = _apply_filter(df, filter_column, filter_operator, filter_value)
            if isinstance(filtered, str):
                return filtered
            if filtered.empty:
                return "The filter matched no rows."
            df = filtered

        try:
            if operation == "columns":
                return "\n".join(f"{c}: {t}" for c, t in zip(df.columns, df.dtypes))
            if operation == "shape":
                return f"{df.shape[0]} rows, {df.shape[1]} columns"
            if operation == "head":
                return df.head(max(1, min(n, MAX_OUTPUT_ROWS))).to_string()
            if operation == "describe":
                return df.describe().to_string()

            if operation in _AGGREGATIONS or operation == "value_counts":
                if column is None:
                    return f"The '{operation}' operation needs a 'column'."
                if column not in df.columns:
                    return _missing_column(column, df)
                if operation == "value_counts":
                    counts = df[column].value_counts()
                    return _limit(counts) + f"\n({counts.size} distinct values)"
                if operation in _NUMERIC_ONLY and not pd.api.types.is_numeric_dtype(df[column]):
                    return f"Column '{column}' is not numeric ({df[column].dtype}); '{operation}' needs numbers."
                # Safe despite getattr: `operation` was checked against a fixed list.
                return str(getattr(df[column], operation)())

            if operation == "groupby_mean":
                if column is None or group_by is None:
                    return "'groupby_mean' needs both 'column' and 'group_by'."
                for name in (group_by, column):
                    if name not in df.columns:
                        return _missing_column(name, df)
                if not pd.api.types.is_numeric_dtype(df[column]):
                    return f"Column '{column}' is not numeric; its mean cannot be computed."
                return _limit(df.groupby(group_by)[column].mean())

            return f"Unknown operation: '{operation}'"
        except Exception as exc:
            return f"Error while running '{operation}': {exc}"

    tools: list[BaseTool] = [
        calculator,
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)),
        document_search,
        analyze_data,
    ]
    if tavily_available():
        from langchain_tavily import TavilySearch

        tools.insert(1, TavilySearch(max_results=2))
    return tools
