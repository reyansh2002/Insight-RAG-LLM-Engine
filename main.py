import pandas as pd
import numpy as np
import os
import re
import json
import time
import requests
from datetime import timedelta, date, datetime
import streamlit as st
import plotly.express as px
import importlib
from collections import Counter
import calendar

# --- 1. MOCKS & DEPENDENCIES ---
try:
    faiss_spec = importlib.util.find_spec('faiss')
    torch_spec = importlib.util.find_spec('torch')
    transformers_spec = importlib.util.find_spec('transformers')

    if faiss_spec and torch_spec and transformers_spec:
        import faiss
        import torch
        from transformers import AutoTokenizer, AutoModel
    else:
        raise ModuleNotFoundError("Using Mocks.")
except Exception:
    class MockFaissIndex:
        def __init__(self, dimension): self.dimension = dimension
        def add(self, vectors): pass
        def search(self, query, k): return np.array([]), np.array([[]])

    class MockEmbeddingModel:
        def encode(self, texts, normalize_embeddings=True): return np.array([])

    faiss = type('faiss', (object,), {'IndexFlatL2': MockFaissIndex})()
    AutoTokenizer = lambda name: None
    AutoModel = lambda name: MockEmbeddingModel()

# --- 2. CONFIGURATION ---
KPI_DATA_FILE = 'daily_kpi_data.csv'
ALERTS_LOG_FILE = 'alerts_log.csv'
RECOMMENDATIONS_FILE = 'recommendations.csv'

DEVIATION_THRESHOLD = -0.15 
DEMO_DEVIATION_DATE_STR = '2025-01-17'

API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBatDpaxYU_jdZC62qLpPJYvmkApk-xPPU")
GEMINI_MODEL = "gemini-2.0-flash" 
API_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
MAX_RETRIES = 3

CAUSAL_VARS = {
    'Rating': True, 'Discount': False, 'M_Spend': True,
    'Supply_Chain_E': True, 'Market_T': True
}

# --- 3. GEMINI API FUNCTIONS ---

def call_gemini_filter_extraction(user_query, schema_str, current_date):
    if not API_KEY: return {}
    today_str = current_date.strftime('%Y-%m-%d')
    system_prompt = (
        "You are a Data Query Parser. Extract structured filters to JSON. "
        f"Columns: {schema_str}. Today: {today_str}. "
        "Keys: start_date, end_date, min_Col, max_Col, or exact ColName. "
        "If list requested, value is 'LIST_ALL'."
    )
    payload = {
        "contents": [{"parts": [{"text": f"Query: {user_query}"}]}],
        "generationConfig": {"responseMimeType": "application/json"},
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    url = f"{API_BASE_URL}?key={API_KEY}"
    for _ in range(3):
        try:
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "{}")
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
                elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    return parsed[0]
                else:
                    return {}
        except Exception: pass
    return {}

def call_gemini_rag_query(user_query, context_data):
    if not API_KEY: return "Error: API Key missing."
    
    # SENDING FULL CONTEXT AS REQUESTED
    full_prompt = (
        "You are an expert KPI Analyst. The user has provided the COMPLETE dataset below.\n"
        "Use this full context to answer the question accurately. Do not make up data.\n"
        "Analyze the rows carefully to answer questions about product lists, specific dates, or performance.\n"
        f"QUESTION: {user_query}\n\nFULL DATA CONTEXT:\n{context_data} and {pd.read_csv(KPI_DATA_FILE)}"
        f"\n\n Also give the recomendations to improve the KPI metrics."
    )
    
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "tools": [{"google_search": {}}], 
        "systemInstruction": {"parts": [{"text": "Helpful business analyst."}]}
    }
    url = f"{API_BASE_URL}?key={API_KEY}"
    
    for attempt in range(MAX_RETRIES):
        try:
            headers = {'Content-Type': 'application/json'}
            # Extended timeout for large context
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            
            if resp.status_code == 200:
                result = resp.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Error")
            elif resp.status_code in [429, 503]: 
                time.sleep(2 ** attempt)
            else: 
                return f"Error: {resp.status_code}"
        except Exception as e: 
            if attempt == MAX_RETRIES - 1:
                return f"Error: Timeout or API Issue ({e})"
    return "Timeout"

def call_gemini_recommendation(prompt):
    if not API_KEY: return None
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
        "systemInstruction": {"parts": [{"text": "Return JSON: {subject, action_plan}"}]}
    }
    url = f"{API_BASE_URL}?key={API_KEY}"
    for attempt in range(MAX_RETRIES):
        try:
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            if resp.status_code == 200:
                result = resp.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "{}")
                return json.loads(text)
            elif resp.status_code in [429, 503]: time.sleep(2 ** attempt)
            else: break
        except Exception: pass
    return None

# --- 4. KPI AGENT CLASS ---

class KPIAgent:
    def __init__(self, data_file, uploaded_file=None):
        self.data_file = data_file
        self.uploaded_file = uploaded_file
        self.df_granular = None
        self.df_daily = None
        self.latest_data_date = None
        self.min_date = None
        self.max_date = None
        
        self.load_data()

    def load_data(self):
        try:
            if self.uploaded_file:
                self.df_granular = pd.read_csv(self.uploaded_file)
            elif os.path.exists(self.data_file):
                self.df_granular = pd.read_csv(self.data_file)
            else:
                self.df_granular = self._create_mock_data()

            if 'Date' not in self.df_granular.columns:
                for col in self.df_granular.columns:
                    if 'date' in col.lower():
                        self.df_granular.rename(columns={col: 'Date'}, inplace=True)
                        break
            
            if 'Date' in self.df_granular.columns:
                self.df_granular['Date'] = pd.to_datetime(self.df_granular['Date']).dt.normalize()
                self.latest_data_date = self.df_granular['Date'].max()
                self.min_date = self.df_granular['Date'].min()
                self.max_date = self.df_granular['Date'].max()
            
            if 'Revenue_d' in self.df_granular.columns:
                self._calculate_kpis()

        except Exception as e:
            print(f"Error: {e}")
            self.df_granular = pd.DataFrame()

    def _create_mock_data(self):
        dates = pd.date_range(start='2025-01-01', periods=30)
        data = {
            'Date': np.repeat(dates, 5),
            'Revenue_d': np.random.randint(50000, 150000, 150),
            'Category': np.tile(['Electronics', 'Footwear', 'Kitchen', 'Sports', 'Home'], 30),
            'Rating': np.random.uniform(3.5, 5.0, 150),
            'Discount': np.random.randint(0, 30, 150),
            'M_Spend': np.random.randint(5000, 15000, 150),
            'Supply_Chain_E': np.random.randint(60, 95, 150),
            'Market_T': np.random.randint(50, 90, 150),
        }
        df = pd.DataFrame(data)
        dev_date = pd.to_datetime(DEMO_DEVIATION_DATE_STR)
        if dev_date in df['Date'].values:
            df.loc[df['Date'] == dev_date, 'Revenue_d'] *= 0.6
            df.loc[df['Date'] == dev_date, 'Rating'] = 2.5
        return df

    def _calculate_kpis(self):
        self.df_daily = self.df_granular.groupby('Date')['Revenue_d'].sum().reset_index()
        self.df_daily.rename(columns={'Revenue_d': 'Overall_Revenue'}, inplace=True)
        self.df_daily['Rolling_Avg_7d'] = self.df_daily['Overall_Revenue'].shift(1).rolling(7).mean()
        self.df_daily['Pct_Change_vs_7d'] = (self.df_daily['Overall_Revenue'] / self.df_daily['Rolling_Avg_7d']) - 1

    def check_date(self, check_date):
        """Return (is_dev, details_dict, msg)."""
        d = pd.to_datetime(check_date).normalize()
        if self.df_daily is None or self.df_daily.empty:
            self._calculate_kpis()
        row = self.df_daily[self.df_daily['Date'] == d]
        current = float(row['Overall_Revenue'].sum()) if not row.empty else 0.0
        prev_start = d - pd.Timedelta(days=7)
        prev = self.df_daily[(self.df_daily['Date'] >= prev_start) & (self.df_daily['Date'] < d)]
        rolling_avg = float(prev['Overall_Revenue'].mean()) if not prev.empty else 0.0
        pct = (current / rolling_avg - 1) if rolling_avg and rolling_avg != 0 else 0.0
        is_dev = pct <= DEVIATION_THRESHOLD
        details = {
            'Date': d,
            'Current_Value': current,
            'Rolling_Avg_7d': rolling_avg,
            'Pct_Deviation': pct
        }
        msg = "Deviation" if is_dev else "OK"
        return is_dev, details, msg

    def causal_analysis(self, deviation_date):
        """Lightweight heuristic causal analysis returning list of drivers."""
        d = pd.to_datetime(deviation_date).normalize()
        drivers = []
        # Use CAUSAL_VARS mapping where available
        for var, consider in CAUSAL_VARS.items():
            if not consider or var not in self.df_granular.columns:
                continue
            # value on date and 7-day prior average
            val_on_date = self.df_granular[self.df_granular['Date'] == d][var]
            if val_on_date.empty:
                continue
            val_on_date = float(val_on_date.mean())
            prior = self.df_granular[(self.df_granular['Date'] < d) & (self.df_granular['Date'] >= d - pd.Timedelta(days=7))][var]
            prior_avg = float(prior.mean()) if not prior.empty else None
            if prior_avg is None or prior_avg == 0:
                change_pct = None
            else:
                change_pct = (val_on_date - prior_avg) / prior_avg
            # flag as potential driver if change magnitude > 5%
            if change_pct is not None and abs(change_pct) >= 0.05:
                drivers.append({'Variable': var, 'Value_On_Date': val_on_date, 'Prior_Avg': prior_avg, 'Change_Pct': change_pct})
        return drivers

    def generate_aggregate_recommendation(self, anomalies):
        """
        Generates a detailed strategic recommendation using Gemini based on the provided anomalies.
        Logs the strategy to recommendations.csv.
        """
        if not anomalies:
            return None
            
        # 1. Construct Context for AI
        anomaly_descriptions = []
        for a in anomalies:
            date_str = a['date'].strftime('%Y-%m-%d') if hasattr(a['date'], 'strftime') else str(a['date'])
            drivers = a.get('drivers', [])
            if drivers:
                driver_text = ", ".join([f"{d['Variable']} (changed {d['Change_Pct']:.1%})" for d in drivers])
            else:
                driver_text = "No specific metric drivers flagged (check external factors)."
            anomaly_descriptions.append(f"- Date: {date_str} | Deviation: {a['pct']:.1%} | Drivers: {driver_text}")
        
        context_str = "\n".join(anomaly_descriptions)
        
        prompt = (
            "You are a Senior Strategy Consultant. Analyze these revenue anomalies detected in the system:\n"
            f"{context_str}\n\n"
            "Based on the drivers identified (e.g., Rating drops, Marketing Spend changes, Supply Chain issues), "
            "formulate a detailed strategic response.\n"
            "Requirements:\n"
            "1. 'subject': A professional, executive summary title (e.g., 'Strategic Pivot required for Inventory...').\n"
            "2. 'action_plan': A structured, step-by-step plan. Use bullet points or numbered lists in the text. "
            "Explain WHY these actions will fix the specific driver."
        )

        # 2. Call AI
        response = call_gemini_recommendation(prompt)
        
        # 3. Fallback if AI fails
        if not response:
            subjects = [str(a['date'])[:10] for a in anomalies]
            subject = f"Manual Review: Anomalies on {', '.join(subjects[:2])}..."
            action_plan = "The AI service could not generate a strategy. Please review the 'Visual Analytics' tab for the specific dates and check Marketing/Inventory logs manually."
            response = {'subject': subject, 'action_plan': action_plan}

        # 4. Log to CSV
        log_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Subject": response.get('subject', 'No Subject'),
            "Action_Plan": response.get('action_plan', 'No Plan'),
            "Anomalies_Found": len(anomalies)
        }
        self._log_to_csv(RECOMMENDATIONS_FILE, log_entry)

        return response

    def _log_to_csv(self, filename, data_dict):
        try:
            file_exists = os.path.isfile(filename)
            df_log = pd.DataFrame([data_dict])
            df_log.to_csv(filename, mode='a', header=not file_exists, index=False)
        except Exception as e:
            print(f"Failed to log to {filename}: {e}")

    def smart_query_handler(self, user_query):
        schema_info = []
        for col in self.df_granular.columns:
            if pd.api.types.is_numeric_dtype(self.df_granular[col]):
                schema_info.append(f"- {col} (Numeric)")
            else:
                schema_info.append(f"- {col} (Text)")
        schema_str = "\n".join(schema_info)

        filters = call_gemini_filter_extraction(user_query, schema_str, self.latest_data_date)
        source = "Semantic Search (Full Data)"
        df_filtered = self.df_granular.copy()
        
        # Apply filters to reduce context size if possible
        if filters and isinstance(filters, dict):
            for key, value in filters.items():
                if key == 'start_date' and 'Date' in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered['Date'] >= pd.to_datetime(value)]
                elif key == 'end_date' and 'Date' in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered['Date'] <= pd.to_datetime(value)]
                elif key in df_filtered.columns:
                    if value != 'LIST_ALL':
                        df_filtered = df_filtered[df_filtered[key].astype(str).str.contains(str(value), case=False, na=False)]

        # Convert FULL Data to CSV string
        context_str = df_filtered.to_csv(index=False)
        
        # Guard against hard limits (approx 100k tokens safety), though Gemini Flash is 1M+
        if len(context_str) > 3000000:
            context_str = context_str[:3000000] + "\n...(Data Truncated)..."

        gemini_response = call_gemini_rag_query(user_query, context_str)
        

        # If API timed out or errored, use extended local deterministic fallback
        if isinstance(gemini_response, str) and (gemini_response.lower().startswith("error") or "timeout" in gemini_response.lower()):
            local_resp = self._local_fallback_answer(user_query, df_filtered)
            if local_resp is not None:
                return local_resp, "Local Aggregation Fallback"

        return gemini_response, source

    def _local_fallback_answer(self, user_query, df_filtered):
        """
        Extended fallback: detects intents and returns local aggregations for:
         - revenue by month range (and category)
         - top N categories/products by revenue
         - list products sold on a specific date
         - daily breakdown for a date range
         - compare revenue between two dates
        Returns a string or None if unable to handle.
        """
        q = user_query.lower()

        # If no revenue field, cannot answer numeric asks
        if 'revenue' in q and 'Revenue_d' not in df_filtered.columns:
            return None

        # Helper: parse numbers (N) and years
        def parse_int(nstr, default=None):
            try:
                return int(nstr)
            except:
                return default

        # 1) Revenue by month-range (existing logic)
        rev_ans = self._answer_revenue_by_range(q, df_filtered)
        if rev_ans: return rev_ans

        # 2) Top N by revenue (category or product)
        topn_ans = self._answer_top_n(q, df_filtered)
        if topn_ans: return topn_ans

        # 3) List products/items sold on a specific date
        list_ans = self._answer_list_on_date(q, df_filtered)
        if list_ans: return list_ans

        # 4) Daily breakdown for a range
        daily_ans = self._answer_daily_breakdown(q, df_filtered)
        if daily_ans: return daily_ans

        # 5) Compare two dates
        cmp_ans = self._answer_compare_dates(q, df_filtered)
        if cmp_ans: return cmp_ans

        return None

    def _answer_revenue_by_range(self, q, df):
        # look for 'revenue' and month tokens
        if 'revenue' not in q: return None

        months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
        months_abbr = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
        months.update(months_abbr)

        found_months = []
        for name, num in months.items():
            if re.search(r'\b' + re.escape(name) + r'\b', q):
                found_months.append(num)

        # patterns like 'jan-mar' or 'jan to mar'
        if not found_months:
            mrange = re.search(r'([a-z]{3,9})\s*(?:-|to|through)\s*([a-z]{3,9})', q)
            if mrange:
                a = mrange.group(1).lower(); b = mrange.group(2).lower()
                if a in months and b in months:
                    start_m = months[a]; end_m = months[b]
                    if start_m <= end_m:
                        found_months = list(range(start_m, end_m + 1))
                    else:
                        found_months = list(range(start_m, 13)) + list(range(1, end_m + 1))

        if not found_months:
            # numeric month tokens
            mnums = re.findall(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2})\b', q)
            for token in mnums:
                t = token.lower()
                if t.isdigit():
                    v = int(t)
                    if 1 <= v <= 12: found_months.append(v)
                else:
                    if t in months: found_months.append(months[t])

        if not found_months: return None

        # find year if present, else latest year
        years = re.findall(r'\b(20\d{2})\b', q)
        year = int(years[0]) if years else self.latest_data_date.year

        start_m = min(found_months); end_m = max(found_months)
        start_dt = pd.to_datetime(f"{year}-{start_m:02d}-01")
        end_dt = pd.to_datetime(f"{year}-{end_m:02d}-01") + pd.offsets.MonthEnd(0)

        dfq = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]

        # optional category/product filter
        cat_col, cat_val = None, None
        for c in dfq.select_dtypes(include=['object', 'category']).columns:
            for v in dfq[c].dropna().astype(str).unique():
                if v.lower() in q:
                    cat_col, cat_val = c, v
                    break
            if cat_col: break
        if cat_col:
            dfq = dfq[dfq[cat_col].astype(str).str.lower() == cat_val.lower()]

        total_rev = dfq['Revenue_d'].sum()
        start_label = f"{calendar.month_name[start_m]} {start_dt.year}"
        end_label = f"{calendar.month_name[end_m]} {end_dt.year}"

        ans = f"Total Revenue {start_label} - {end_label}"
        if cat_col:
            ans += f" for {cat_val} ({cat_col})"
        ans += f": ${total_rev:,.0f} (calculated locally due to API issue)"
        return ans

    def _answer_top_n(self, q, df):
        # look for "top N" or "top N categories/products"
        m = re.search(r'top\s+(\d+)', q)
        if not m: 
            # also support "top products" without number => default 5
            if 'top ' not in q: return None
            n = 5
        else:
            n = int(m.group(1))

        # determine dimension: category or product column
        cand_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c.lower() not in ['date']]
        if not cand_cols: return None

        # Prefer 'Product' or 'Category' if present
        pref = None
        for p in ['product', 'category', 'item', 'sku', 'name']:
            for c in cand_cols:
                if p in c.lower():
                    pref = c; break
            if pref: break
        dim = pref if pref else cand_cols[0]

        agg = df.groupby(dim)['Revenue_d'].sum().reset_index().sort_values('Revenue_d', ascending=False).head(n)
        lines = [f"{i+1}. {row[dim]} — ${row['Revenue_d']:,.0f}" for i, row in agg.reset_index(drop=True).iterrows()]
        ans = f"Top {len(lines)} by Revenue ({dim}):\n" + "\n".join(lines) + "\n(calculated locally due to API issue)"
        return ans

    def _answer_list_on_date(self, q, df):
        # detect phrases like 'list products on Jan 15' or 'products sold on 2025-01-15'
        ph = re.search(r'on\s+([0-9]{4}-[0-9]{2}-[0-9]{2})', q)
        date_token = None
        if ph:
            date_token = ph.group(1)
        else:
            m = re.search(r'on\s+([a-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?)', q)
            if m:
                try:
                    date_token = str(pd.to_datetime(m.group(1)).date())
                except:
                    date_token = None

        if not date_token:
            return None

        d = pd.to_datetime(date_token).normalize()
        dfq = df[df['Date'] == d]
        if dfq.empty: return f"No records found on {d.strftime('%Y-%m-%d')} (checked locally)."

        # choose a product-like column
        prod_col = None
        for c in dfq.select_dtypes(include=['object', 'category']).columns:
            if c.lower() not in ['category'] and c.lower() not in ['date']:
                prod_col = c; break
        if not prod_col:
            # fallback to Category listing
            prod_col = 'Category' if 'Category' in dfq.columns else dfq.columns[0]

        items = dfq[prod_col].astype(str).unique().tolist()
        ans = f"Items on {d.strftime('%Y-%m-%d')} ({len(items)}): " + ", ".join(items[:50])
        if len(items) > 50: ans += " ... (truncated)"
        ans += " (calculated locally due to API issue)"
        return ans

    def _answer_daily_breakdown(self, q, df):
        # detect "daily breakdown" or "daily revenue" for a range
        if 'daily' not in q and 'per day' not in q and 'day by day' not in q:
            return None

        # extract date range tokens
        dates = re.findall(r'([0-9]{4}-[0-9]{2}-[0-9]{2})', q)
        if len(dates) >= 2:
            start_dt = pd.to_datetime(dates[0]).normalize()
            end_dt = pd.to_datetime(dates[1]).normalize()
        else:
            # fallback to month parsing like Jan-Mar
            months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
            mrange = re.search(r'([a-z]{3,9})\s*(?:-|to|through)\s*([a-z]{3,9})', q)
            if not mrange: return None
            year = self.latest_data_date.year
            start_dt = pd.to_datetime(f"{year}-{months[mrange.group(1).lower()]:02d}-01")
            end_dt = pd.to_datetime(f"{year}-{months[mrange.group(2).lower()]:02d}-01") + pd.offsets.MonthEnd(0)

        dfq = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
        if dfq.empty: return f"No data between {start_dt.date()} and {end_dt.date()} (checked locally)."

        daily = dfq.groupby(dfq['Date'])['Revenue_d'].sum().reset_index()
        lines = [f"{row['Date'].date()}: ${row['Revenue_d']:,.0f}" for _, row in daily.iterrows()]
        ans = "Daily Revenue:\n" + "\n".join(lines[:200])  # cap output
        if len(lines) > 200: ans += "\n... (truncated)"
        ans += "\n(calculated locally due to API issue)"
        return ans

    def _answer_compare_dates(self, q, df):
        # detect patterns like 'compare Jan 5 and Jan 12' or 'compare 2025-01-05 to 2025-01-12'
        m = re.findall(r'([0-9]{4}-[0-9]{2}-[0-9]{2})', q)
        if len(m) >= 2:
            d1 = pd.to_datetime(m[0]).normalize(); d2 = pd.to_datetime(m[1]).normalize()
        else:
            m = re.findall(r'([a-z]{3,9}\s+\d{1,2})', q)
            if len(m) >= 2:
                try:
                    d1 = pd.to_datetime(m[0]).normalize(); d2 = pd.to_datetime(m[1]).normalize()
                except:
                    return None
            else:
                return None

        r1 = df.loc[df['Date'] == d1, 'Revenue_d'].sum() if not df.loc[df['Date'] == d1].empty else 0
        r2 = df.loc[df['Date'] == d2, 'Revenue_d'].sum() if not df.loc[df['Date'] == d2].empty else 0
        diff = r2 - r1
        pct = (diff / r1) if r1 != 0 else None

        ans = f"Revenue on {d1.date()}: ${r1:,.0f}\nRevenue on {d2.date()}: ${r2:,.0f}\nDifference: ${diff:,.0f}"
        if pct is not None:
            ans += f" ({pct:.1%} change)"
        else:
            ans += " (baseline zero, percent change N/A)"
        ans += "\n(calculated locally due to API issue)"
        return ans
# --- 5. STREAMLIT UI ---

def main():
    st.set_page_config(page_title="KPI Agent", layout="wide")
    st.title("📊 Intelligent KPI Monitor & Agent")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload KPI CSV", type=['csv'])
        
        agent = KPIAgent(KPI_DATA_FILE, uploaded_file)
        if agent.df_granular is None or agent.df_granular.empty:
            st.error("No data available.")
            st.stop()

        st.divider()
        st.subheader("🚨 Agent Monitor")
        st.caption("Select Date Range to Scan")
        
        min_dt = agent.min_date.date()
        max_dt = agent.max_date.date()
        
        col_s1, col_s2 = st.columns(2)
        with col_s1: start_scan = st.date_input("Start", value=min_dt, min_value=min_dt, max_value=max_dt)
        with col_s2: end_scan = st.date_input("End", value=max_dt, min_value=min_dt, max_value=max_dt)
            
        run_check = st.button("Run KPI Scan")

    # --- Metrics ---
    latest_date = agent.latest_data_date
    latest_data = agent.df_granular[agent.df_granular['Date'] == latest_date]
    
    st.markdown(f"### Snapshot: {latest_date.strftime('%Y-%m-%d')}")
    col1, col2, col3, col4 = st.columns(4)
    
    total_rev = latest_data['Revenue_d'].sum() if 'Revenue_d' in latest_data.columns else 0
    col1.metric("Total Revenue", f"${total_rev:,.0f}")
    
    if run_check:
        st.divider()
        st.subheader(f"🔍 Agent Analysis Results ({start_scan} to {end_scan})")
        with st.spinner("Analyzing date range..."):
            anomalies = []
            current_loop_date = pd.to_datetime(start_scan)
            end_loop_date = pd.to_datetime(end_scan)
            
            while current_loop_date <= end_loop_date:
                is_dev, details, msg = agent.check_date(current_loop_date)
                if is_dev:
                    factors = agent.causal_analysis(details['Date'])
                    anomalies.append({'date': details['Date'], 'pct': details['Pct_Deviation'], 'current': details['Current_Value'], 'drivers': factors})
                current_loop_date += timedelta(days=1)
            
            if anomalies:
                st.warning(f"⚠️ Found {len(anomalies)} deviation(s).")
                st.info(f"Logged to `{ALERTS_LOG_FILE}`.")
                
                table_data = []
                for a in anomalies:
                    drivers_str = ", ".join([d['Variable'] for d in a['drivers']]) if a['drivers'] else "Unknown"
                    table_data.append({"Date": a['date'].strftime('%Y-%m-%d'), "Revenue": f"${a['current']:,.0f}", "Deviation": f"{a['pct']:.1%}", "Key Drivers": drivers_str})
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                
                st.markdown("#### 💡 Final Strategic Recommendation")
                with st.spinner("Generating consolidated strategy..."):
                    rec = agent.generate_aggregate_recommendation(anomalies)
                    if rec:
                        st.success(f"Strategy logged to `{RECOMMENDATIONS_FILE}`")
                        st.info(f"**Subject: {rec.get('subject')}**")
                        st.write(rec.get('action_plan'))
            else:
                st.success("✅ No critical deviations detected in the selected range.")

    st.divider()

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["📈 Visual Analytics", "💬 Smart Chat", "📋 Raw Data"])

    with tab1:
        st.subheader("Trends & Insights")
        cat_cols = agent.df_granular.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [c for c in cat_cols if c != 'Date']
        primary_cat = 'Category' if 'Category' in cat_cols else (cat_cols[0] if cat_cols else None)

        if 'Overall_Revenue' in agent.df_daily.columns:
            mask = (agent.df_daily['Date'].dt.date >= start_scan) & (agent.df_daily['Date'].dt.date <= end_scan)
            chart_df = agent.df_daily[mask] if not agent.df_daily[mask].empty else agent.df_daily
            fig_rev = px.line(chart_df, x='Date', y='Overall_Revenue', title='Revenue Trend')
            st.plotly_chart(fig_rev, use_container_width=True)

    with tab3:
        st.dataframe(agent.df_granular)

    # --- CHAT TAB (Clean ChatGPT-like Interface) ---
    with tab2:
        st.caption("Ask questions about your data (e.g., 'List products sold on Jan 15', 'Why did revenue drop?')")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render stored history only (input moved out so the input stays pinned)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # top-level chat input so it stays pinned at the bottom of the app
        if prompt := st.chat_input("Ask the Agent..."):
            # Append user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Call agent and append assistant response (the tab will re-render and show these)
            with st.spinner("Analyzing Full Data Context..."):
                response, source = agent.smart_query_handler(prompt)
                final_msg = f"**[{source}]**\n\n{response}"
            st.session_state.messages.append({"role": "assistant", "content": final_msg})
            # Optionally show the messages immediately in this run as well
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(final_msg)

if __name__ == '__main__':
    main()