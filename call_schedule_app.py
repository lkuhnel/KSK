# call_schedule_app.py

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime as dt_type, date as date_type, timedelta
import json
import os
from scheduling_engine import run_scheduling_engine
from run_formatter import format_schedule
from openpyxl import Workbook
import io
from gmail_fetcher import fetch_requests_from_gmail, ensure_date
import traceback
import copy

# After imports, add:
def norm_name(name):
    return str(name).strip().lower()

def is_within_block(start, end, block_start, block_end):
    from datetime import datetime
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    block_start = datetime.strptime(block_start, "%Y-%m-%d")
    block_end = datetime.strptime(block_end, "%Y-%m-%d")
    return not (end < block_start or start > block_end)

# Initialize session state variables
if 'pto_requests' not in st.session_state:
    st.session_state.pto_requests = {}
if 'previous_call_counts' not in st.session_state:
    st.session_state.previous_call_counts = None
if 'schedule_df' not in st.session_state:
    st.session_state.schedule_df = None
if 'residents' not in st.session_state:
    st.session_state.residents = None
if 'block_start_date' not in st.session_state:
    st.session_state.block_start_date = None
if 'block_end_date' not in st.session_state:
    st.session_state.block_end_date = None
if 'block_number' not in st.session_state:
    st.session_state.block_number = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Hard Constraints"
if 'last_modified_resident' not in st.session_state:
    st.session_state.last_modified_resident = None
if 'pending_pto_action' not in st.session_state:
    st.session_state.pending_pto_action = None
if 'pending_soft_action' not in st.session_state:
    st.session_state.pending_soft_action = None

# Initialize all session state variables
if 'residents_data_by_block' not in st.session_state:
    st.session_state.residents_data_by_block = {}
if 'holiday_assignments_by_block' not in st.session_state:
    st.session_state.holiday_assignments_by_block = {}
if 'pto_requests_by_block' not in st.session_state:
    st.session_state.pto_requests_by_block = {}
if 'soft_constraints_by_block' not in st.session_state:
    st.session_state.soft_constraints_by_block = {}
if 'previous_assignments_by_block' not in st.session_state:
    st.session_state.previous_assignments_by_block = {}
if 'previous_call_counts_by_block' not in st.session_state:
    st.session_state.previous_call_counts_by_block = {}
if 'removed_residents_by_block' not in st.session_state:
    st.session_state.removed_residents_by_block = {}
if 'removed_holidays_by_block' not in st.session_state:
    st.session_state.removed_holidays_by_block = {}
if 'removed_pto_by_block' not in st.session_state:
    st.session_state.removed_pto_by_block = {}
if 'removed_soft_constraints_by_block' not in st.session_state:
    st.session_state.removed_soft_constraints_by_block = {}
if 'block_dates' not in st.session_state:
    st.session_state.block_dates = None
if 'resident_count_by_block' not in st.session_state:
    st.session_state.resident_count_by_block = {}
if 'holiday_count_by_block' not in st.session_state:
    st.session_state.holiday_count_by_block = {}
if 'pto_count_by_block' not in st.session_state:
    st.session_state.pto_count_by_block = {}
if 'soft_constraint_count_by_block' not in st.session_state:
    st.session_state.soft_constraint_count_by_block = {}
if 'current_academic_year_by_block' not in st.session_state:
    st.session_state.current_academic_year_by_block = {}
if 'loaded_residents_by_block' not in st.session_state:
    st.session_state.loaded_residents_by_block = {}
if 'disable_holidays_by_block' not in st.session_state:
    st.session_state.disable_holidays_by_block = {}
if 'disable_pto_by_block' not in st.session_state:
    st.session_state.disable_pto_by_block = {}
if 'disable_soft_constraints_by_block' not in st.session_state:
    st.session_state.disable_soft_constraints_by_block = {}

def safe_int(val):
    return 0 if pd.isna(val) else int(val)

def safe_int_nan(val):
    if pd.isna(val):
        return 0
    try:
        return int(val)
    except Exception:
        return 0

def save_data(academic_year):
    """Save only resident information for an academic year"""
    if not os.path.exists('saved_data'):
        os.makedirs('saved_data')
    
    data = {
        'residents': st.session_state.residents_data_by_block[academic_year]
    }
    
    filename = f'saved_data/academic_year_{academic_year.replace("-", "_")}.json'
    with open(filename, 'w') as f:
        json.dump(data, f)
    return filename

def load_data(academic_year):
    """Load only resident information for an academic year"""
    filename = f'saved_data/academic_year_{academic_year.replace("-", "_")}.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data.get('residents', [])
    return []

def calculate_call_distribution(schedule_df, block_end_dt=None):
    """Calculate call distribution statistics for each resident, using their final PGY for the block."""
    stats = {1: [], 2: [], 3: [], 4: []}
    all_residents = set(schedule_df['Call'].unique()) | set(schedule_df['Backup'].unique()) | set(schedule_df['Intern'].unique())

    # Build a lookup for each resident's final PGY at the end of the block
    final_pgy_lookup = {}
    for res in st.session_state.residents_data_by_block[block_choice]:
        pgy = int(res['PGY'])
        transition_date = res.get('Transition_Date')
        final_pgy = pgy
        if transition_date and pd.notna(transition_date):
            try:
                tdate = dt_type.strptime(str(transition_date), "%Y-%m-%d")
                if block_end_dt and tdate <= block_end_dt:
                    final_pgy = min(pgy + 1, 4)
            except Exception:
                pass
        final_pgy_lookup[res['Name']] = final_pgy

    for resident in all_residents:
        call_assignments = schedule_df[schedule_df['Call'] == resident]
        backup_assignments = schedule_df[schedule_df['Backup'] == resident]
        intern_assignments = schedule_df[schedule_df['Intern'] == resident]
        final_pgy = final_pgy_lookup.get(resident, None)
        if final_pgy is None:
            continue
        # PGY-1: Interns
        if final_pgy == 1:
            all_intern_dates = pd.concat([intern_assignments['Date'], call_assignments['Date']]).unique()
            weekday_calls = sum(1 for date in all_intern_dates if dt_type.strptime(str(date), "%Y-%m-%d").weekday() not in [5, 6])
            saturday_calls = sum(1 for date in all_intern_dates if dt_type.strptime(str(date), "%Y-%m-%d").weekday() == 5)
            total_calls = len(all_intern_dates)
            stats[1].append({
                'Resident': resident,
                'Weekday': weekday_calls,
                'Saturday': saturday_calls,
                'Total': total_calls
            })
        # PGY-2: Weekday, Friday, Sunday, Total
        elif final_pgy == 2:
            weekday_calls = len(call_assignments[call_assignments['Date'].apply(lambda x: dt_type.strptime(str(x), "%Y-%m-%d").weekday() in [0,1,2,3])])
            friday_calls = len(call_assignments[call_assignments['Date'].apply(lambda x: dt_type.strptime(str(x), "%Y-%m-%d").weekday() == 4)])
            sunday_calls = len(call_assignments[call_assignments['Date'].apply(lambda x: dt_type.strptime(str(x), "%Y-%m-%d").weekday() == 6)])
            total_calls = len(call_assignments)
            stats[2].append({
                'Resident': resident,
                'Weekday': weekday_calls,
                'Fridays': friday_calls,
                'Sunday': sunday_calls,
                'Total': total_calls
            })
        # PGY-3: Weekday, Saturday, Total
        elif final_pgy == 3:
            weekday_calls = len(call_assignments[call_assignments['Date'].apply(lambda x: dt_type.strptime(str(x), "%Y-%m-%d").weekday() in [0,1,2,3])])
            saturday_calls = len(call_assignments[call_assignments['Date'].apply(lambda x: dt_type.strptime(str(x), "%Y-%m-%d").weekday() == 5)])
            total_calls = len(call_assignments)
            stats[3].append({
                'Resident': resident,
                'Weekday': weekday_calls,
                'Saturday': saturday_calls,
                'Total': total_calls
            })
        # PGY-4: Total only
        elif final_pgy == 4:
            total_calls = len(call_assignments)
            stats[4].append({
                'Resident': resident,
                'Total': total_calls
            })
    all_stats = []
    for pgy in [1, 2, 3, 4]:
        if stats[pgy]:
            df = pd.DataFrame(stats[pgy])
            df['PGY'] = pgy
            all_stats.append(df)
    if all_stats:
        return pd.concat(all_stats, ignore_index=True)
    return pd.DataFrame()

st.set_page_config(page_title="Kall Scheduler Kuhnel (KSK)", layout="wide")

st.markdown("""
<h2 style='text-align: center;'>Kall Scheduler Kuhnel (KSK)</h2>
""", unsafe_allow_html=True)

# Academic Year Selection
current_year = dt_type.now().year
year_options = [f"{year}-{year+1}" for year in range(2025, current_year+10)]
academic_year = st.selectbox("Select Academic Year:", year_options)
start_year = int(academic_year.split('-')[0])

# Initialize or update block dates based on academic year
if st.session_state.block_dates is None or st.session_state.current_academic_year_by_block.get(academic_year) != academic_year:
    st.session_state.block_dates = {
        "Block 1": {
            "start": f"{start_year}-07-01",
            "end": f"{start_year}-10-31"
        },
        "Block 2": {
            "start": f"{start_year}-11-01",
            "end": f"{start_year+1}-02-28"
        },
        "Block 3": {
            "start": f"{start_year+1}-03-01",
            "end": f"{start_year+1}-06-30"
        }
    }
    st.session_state.current_academic_year_by_block[academic_year] = academic_year
    # Reset holiday assignments when academic year changes
    st.session_state.holiday_assignments_by_block[academic_year] = []
    st.session_state.holiday_count_by_block[academic_year] = 1
    # Reset PTO requests when academic year changes
    st.session_state.pto_requests_by_block[academic_year] = {}
    st.session_state.pto_count_by_block[academic_year] = 1
    st.session_state.removed_pto_by_block[academic_year] = set()
    # Reset soft constraints when academic year changes
    st.session_state.soft_constraints_by_block[academic_year] = []
    st.session_state.soft_constraint_count_by_block[academic_year] = 1
    st.session_state.removed_soft_constraints_by_block[academic_year] = set()

# Load saved data when academic year changes
if st.session_state.current_academic_year_by_block.get(academic_year) != academic_year:
    loaded_residents = load_data(academic_year)
    if loaded_residents:
        st.session_state.loaded_residents_by_block[academic_year] = loaded_residents
        st.session_state.resident_count_by_block[academic_year] = len(loaded_residents)
    st.session_state.current_academic_year_by_block[academic_year] = academic_year

# Block selector with date ranges
block_info = {
    "Block 1": {
        "date_range": f"July - October {start_year}",
        "default_start": f"{start_year}-07-01",
        "default_end": f"{start_year}-10-31",
        "requires_previous": False
    },
    "Block 2": {
        "date_range": f"November {start_year} - February {start_year+1}",
        "default_start": f"{start_year}-11-01",
        "default_end": f"{start_year+1}-02-28",
        "requires_previous": True
    },
    "Block 3": {
        "date_range": f"March - June {start_year+1}",
        "default_start": f"{start_year+1}-03-01",
        "default_end": f"{start_year+1}-06-30",
        "requires_previous": True
    }
}

# Move PGY-4 cap input to the right of block selection
col_block_radio, col_block_cap = st.columns([3, 1])
with col_block_radio:
    block_choice = st.radio("Select block to generate:", list(block_info.keys()), horizontal=True)
with col_block_cap:
    pgy4_cap = st.number_input(
        "Desired PGY-4 call cap for this block:",
        min_value=1,
        max_value=100,
        value=st.session_state.get('pgy4_cap', 2),
        step=1,
        key='pgy4_cap'
    )

# Display block information and date selection
st.info(f"📅 Selected: {block_choice}")

# Add date selection
col1, col2 = st.columns(2)
with col1:
    block_start = st.date_input(
        "Block Start Date",
        value=dt_type.strptime(st.session_state.block_dates[block_choice]["start"], "%Y-%m-%d").date(),
        min_value=dt_type(start_year, 7, 1).date(),
        max_value=dt_type(start_year+1, 6, 30).date(),
        key=f"block_start_{block_choice}"
    )
with col2:
    block_end = st.date_input(
        "Block End Date",
        value=dt_type.strptime(st.session_state.block_dates[block_choice]["end"], "%Y-%m-%d").date(),
        min_value=dt_type(start_year, 7, 1).date(),
        max_value=dt_type(start_year+1, 6, 30).date(),
        key=f"block_end_{block_choice}"
    )

# Add this after the block start/end date inputs
fairness_weight = st.slider(
    "Fairness vs. Soft Constraint Weighting (Fairness %)",
    min_value=0,
    max_value=100,
    value=75,
    step=1,
    help="Set to 100 for pure fairness, 0 for pure soft constraint fulfillment."
) / 100.0
soft_constraint_weight = 1.0 - fairness_weight

# Convert to datetime objects for comparison
block_start_dt = dt_type.combine(block_start, dt_type.min.time())
block_end_dt = dt_type.combine(block_end, dt_type.min.time())

# Validate dates
if block_end_dt <= block_start_dt:
    st.error("End date must be after start date")
elif block_choice == "Block 1" and block_start != dt_type.strptime(f"{start_year}-07-01", "%Y-%m-%d").date():
    st.error("Block 1 must start on July 1st")
elif block_choice == "Block 3" and block_end != dt_type.strptime(f"{start_year+1}-06-30", "%Y-%m-%d").date():
    st.error("Block 3 must end on June 30th")

# Store the selected dates in session state
st.session_state.block_dates[block_choice] = {
    "start": block_start.strftime("%Y-%m-%d"),
    "end": block_end.strftime("%Y-%m-%d")
}

# After block_choice is set
# Ensure all per-block session state values are initialized for the current block
per_block_defaults = {
    'residents_data_by_block': [],
    'holiday_assignments_by_block': [],
    'pto_requests_by_block': {},
    'soft_constraints_by_block': {},
    'previous_assignments_by_block': [],
    'previous_call_counts_by_block': {},
    'removed_residents_by_block': set(),
    'removed_holidays_by_block': set(),
    'removed_pto_by_block': set(),
    'removed_soft_constraints_by_block': set(),
    'resident_count_by_block': 1,
    'holiday_count_by_block': 1,
    'pto_count_by_block': 1,
    'soft_constraint_count_by_block': 1,
    'loaded_residents_by_block': [],
    'disable_holidays_by_block': False,
    'disable_pto_by_block': False,
    'disable_soft_constraints_by_block': False,
}
for key, default in per_block_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = {}
    if block_choice not in st.session_state[key]:
        st.session_state[key][block_choice] = default

# --- Tab state management ---
tab_labels = ["Residents", "Holiday Assignments", "Hard Constraints", "Soft Constraints", "Previous Block", "Previous Call Counts", "Generate & Review"]
tabs = st.tabs(tab_labels)

def add_pto_for_resident(resident):
    if not isinstance(st.session_state.pto_requests_by_block[block_choice], dict):
        st.session_state.pto_requests_by_block[block_choice] = {}
    st.session_state.pto_requests_by_block[block_choice].setdefault(resident, []).append({
        "Start_Date": "",
        "End_Date": ""
    })

def remove_pto_for_resident(resident, idx):
    if resident in st.session_state.pto_requests_by_block[block_choice] and 0 <= idx < len(st.session_state.pto_requests_by_block[block_choice][resident]):
        st.session_state.pto_requests_by_block[block_choice][resident].pop(idx)
        if not st.session_state.pto_requests_by_block[block_choice][resident]:
            del st.session_state.pto_requests_by_block[block_choice][resident]

with tabs[0]:
    st.subheader("Resident Information")
    
    def add_resident():
        st.session_state.resident_count_by_block[block_choice] += 1

    def remove_resident(index):
        # Remove the resident at the given index from the list for this block
        if 0 <= index < len(st.session_state.residents_data_by_block[block_choice]):
            st.session_state.residents_data_by_block[block_choice].pop(index)
            st.session_state.resident_count_by_block[block_choice] = len(st.session_state.residents_data_by_block[block_choice])
            st.rerun()

    # Use st.data_editor for batch resident editing (Streamlit 1.22+)
    if st.session_state.residents_data_by_block.get(block_choice):
        df = pd.DataFrame(st.session_state.residents_data_by_block[block_choice])
    else:
        df = pd.DataFrame(columns=["Name", "PGY", "Transition_Date"])

    # Convert Transition_Date to datetime.date for compatibility with DateColumn
    if "Transition_Date" in df.columns:
        df["Transition_Date"] = pd.to_datetime(df["Transition_Date"], errors="coerce").dt.date

    st.markdown("### Resident List")
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Name": st.column_config.TextColumn(
                "Resident Name",
                width="large"
            ),
            "PGY": st.column_config.SelectboxColumn(
                "PGY Level",
                options=[1, 2, 3, 4],
                required=True,
                width="small"
            ),
            "Transition_Date": st.column_config.DateColumn(
                "Off Cycle Date",
                format="YYYY-MM-DD",
                min_value=dt_type(start_year, 7, 1),
                max_value=dt_type(start_year+1, 6, 30),
                width="medium"
            ),
        },
        key=f"residents_data_editor_{block_choice}"
    )
    st.session_state.residents_data_by_block[block_choice] = copy.deepcopy(edited_df.to_dict('records'))
    st.session_state.resident_count_by_block[block_choice] = len(st.session_state.residents_data_by_block[block_choice])

    # Remove resident via selectbox and button
    resident_names = [row['Name'] for row in st.session_state.residents_data_by_block[block_choice]]
    if resident_names:
        selected_to_remove = st.selectbox("Select resident to remove:", resident_names, key=f"remove_resident_select_{block_choice}")
        if st.button("Remove Selected Resident", key=f"remove_resident_btn_{block_choice}"):
            idx = resident_names.index(selected_to_remove)
            remove_resident(idx)

    st.button("Add Another Resident", on_click=add_resident)

    # Save Residents button
    if st.button("Save Residents"):
        if st.session_state.residents_data_by_block[block_choice]:
            # Export residents data as CSV
            residents_df = pd.DataFrame(st.session_state.residents_data_by_block[block_choice])
            csv = residents_df.to_csv(index=False)
            st.download_button(
                label="Download Residents CSV",
                data=csv,
                file_name="residents.csv",
                mime="text/csv"
            )
            st.success(f"Residents data exported as CSV for {academic_year}")
        else:
            st.warning("No resident data to save")

    # --- CSV upload: ensure all rows are loaded ---
    uploaded_file = st.file_uploader("Upload Residents CSV", type=["csv"], key=f"residents_csv_upload_{block_choice}")
    if uploaded_file is not None and not st.session_state.get(f"residents_csv_uploaded_{block_choice}", False):
        try:
            residents_df = pd.read_csv(uploaded_file, dtype=str).fillna("")
            st.session_state.residents_data_by_block[block_choice] = copy.deepcopy(residents_df.to_dict('records'))
            st.session_state.resident_count_by_block[block_choice] = len(st.session_state.residents_data_by_block[block_choice])
            st.session_state[f"residents_csv_uploaded_{block_choice}"] = True  # Mark as processed
            st.success("Residents data uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
    elif uploaded_file is None and st.session_state.get(f"residents_csv_uploaded_{block_choice}", False):
        st.session_state[f"residents_csv_uploaded_{block_choice}"] = False  # Reset for next upload

with tabs[1]:
    st.subheader("Holiday Call Assignments")
    
    def add_holiday():
        st.session_state.holiday_count_by_block[block_choice] += 1

    def remove_holiday(index):
        st.session_state.removed_holidays_by_block[block_choice].add(index)
        st.rerun()

    # Add disable holidays checkbox
    st.session_state.disable_holidays_by_block[block_choice] = st.checkbox("Disable Holiday Assignments", value=st.session_state.disable_holidays_by_block.get(block_choice, False))

    if not st.session_state.disable_holidays_by_block[block_choice]:
        # Initialize holiday_assignments_list if not already initialized
        if 'holiday_assignments_by_block' not in st.session_state:
            st.session_state.holiday_assignments_by_block = {}
        
        # Filter PGY-2 residents for holiday assignments
        pgy2_residents = [res["Name"] for res in st.session_state.residents_data_by_block[block_choice] if int(res["PGY"]) == 2]
        
        # Temporary list to collect new holiday assignments
        new_holiday_assignments = []
        
        for i in range(st.session_state.holiday_count_by_block[block_choice]):
            if i in st.session_state.removed_holidays_by_block[block_choice]:
                continue
                
            st.markdown(f"### Holiday #{i+1}")
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 0.5])
            
            with col1:
                # Get existing holiday name if available
                default_name = st.session_state.holiday_assignments_by_block[block_choice][i].get('Name', "") if i < len(st.session_state.holiday_assignments_by_block[block_choice]) else ""
                holiday_name = st.text_input("Holiday Name", value=default_name, key=f"holiday_name_{i}")
            with col2:
                min_date = block_start
                max_date = block_end
                # Get existing holiday date if available, otherwise use block start date
                if i < len(st.session_state.holiday_assignments_by_block[block_choice]):
                    raw_date = st.session_state.holiday_assignments_by_block[block_choice][i]['Date']
                    try:
                        default_date = dt_type.strptime(raw_date, "%Y-%m-%d").date()
                    except Exception:
                        default_date = min_date
                else:
                    default_date = min_date
                # Ensure default_date is within the allowed range
                if not (min_date <= default_date <= max_date):
                    default_date = min_date
                holiday_date = st.date_input(
                    "Date",
                    value=default_date,
                    min_value=min_date,
                    max_value=max_date,
                    key=f"holiday_date_{i}"
                )
            with col3:
                # Get existing call assignment if available
                default_call = st.session_state.holiday_assignments_by_block[block_choice][i].get('Call', "") if i < len(st.session_state.holiday_assignments_by_block[block_choice]) else ""
                call = st.selectbox(
                    "Call Assignment",
                    [""] + pgy2_residents if pgy2_residents else [""],
                    index=0 if default_call == "" else pgy2_residents.index(default_call) + 1 if default_call in pgy2_residents else 0,
                    key=f"holiday_call_{i}"
                )
            with col4:
                # Get existing backup assignment if available
                default_backup = st.session_state.holiday_assignments_by_block[block_choice][i].get('Backup', "") if i < len(st.session_state.holiday_assignments_by_block[block_choice]) else ""
                backup = st.selectbox(
                    "Backup Assignment",
                    [""] + pgy2_residents if pgy2_residents else [""],
                    index=0 if default_backup == "" else pgy2_residents.index(default_backup) + 1 if default_backup in pgy2_residents else 0,
                    key=f"holiday_backup_{i}"
                )
            with col5:
                st.button("❌", key=f"remove_hol_{i}", on_click=remove_holiday, args=(i,))
            
            if holiday_name and call != "" and backup != "":  # Only add if all fields are filled
                new_holiday_assignments.append({
                    "Name": holiday_name,
                    "Date": holiday_date.strftime("%Y-%m-%d"),
                    "Call": call,
                    "Backup": backup
                })
        
        # Update holiday_assignments_list with new data
        st.session_state.holiday_assignments_by_block[block_choice] = new_holiday_assignments

        col1, col2 = st.columns([6, 1])
        with col1:
            st.button("Add Another Holiday", on_click=add_holiday)
        with col2:
            if st.button("Clear Holidays"):
                st.session_state.holiday_count_by_block[block_choice] = 1
                st.session_state.holiday_assignments_by_block[block_choice] = []
                st.session_state.removed_holidays_by_block[block_choice] = set()
                st.rerun()

with tabs[2]:
    st.session_state.current_tab = "Hard Constraints"
    st.subheader("Hard Constraints")
    st.info("Use this section to specify PTO requests and rotation blocks where residents cannot be assigned call. These are high-priority constraints that will be strictly enforced in the schedule.")

    # --- Move the fetch button to the top of the tab ---
    if st.button("Fetch Requests from Gmail"):
        pto_requests, _ = fetch_requests_from_gmail()  # Only use PTO requests here
        # Only keep requests with Reason == 'PTO' and within block date range
        block_start_str = block_start.strftime("%Y-%m-%d")
        block_end_str = block_end.strftime("%Y-%m-%d")
        pto_only = [req for req in pto_requests if req.get("Reason", "").lower() == "pto" and is_within_block(req["Start_Date"], req["End_Date"], block_start_str, block_end_str)]
        print('DEBUG PTO ONLY:', pto_only)
        # Match by last name only (case-insensitive)
        valid_residents = {res["Name"].strip().lower(): res["Name"] for res in st.session_state.residents_data_by_block[block_choice]}
        last_name_map = {res["Name"].strip().lower(): res["Name"] for res in st.session_state.residents_data_by_block[block_choice]}
        grouped_pto = {}
        for req in pto_only:
            # Extract last name from email resident name
            req_last = req["Resident"].strip().split()[-1].lower()
            # Find a matching resident by last name
            match = None
            for lname, canonical in last_name_map.items():
                if req_last == lname:
                    match = canonical
                    break
            if match:
                if match not in grouped_pto:
                    grouped_pto[match] = []
                grouped_pto[match].append({
                    "Start_Date": req["Start_Date"],
                    "End_Date": req["End_Date"]
                })
        # Merge with existing PTO requests instead of replacing them
        if not isinstance(st.session_state.pto_requests_by_block[block_choice], dict):
            st.session_state.pto_requests_by_block[block_choice] = {}
        for resident, requests in grouped_pto.items():
            if resident not in st.session_state.pto_requests_by_block[block_choice]:
                st.session_state.pto_requests_by_block[block_choice][resident] = []
            # Add new requests while avoiding duplicates
            existing_dates = {(req["Start_Date"], req["End_Date"]) for req in st.session_state.pto_requests_by_block[block_choice][resident]}
            for req in requests:
                if (req["Start_Date"], req["End_Date"]) not in existing_dates:
                    st.session_state.pto_requests_by_block[block_choice][resident].append(req)
        msg = f"Fetched and merged PTO requests for {len(grouped_pto)} residents from Gmail."
        st.success(msg)

    st.session_state.disable_pto_by_block[block_choice] = st.checkbox("Disable PTO Requests", value=st.session_state.disable_pto_by_block.get(block_choice, False))

    if not st.session_state.disable_pto_by_block[block_choice]:
        if 'pto_requests_by_block' not in st.session_state:
            st.session_state.pto_requests_by_block = {}
        
        # Keep original resident order
        for resident in [res["Name"] for res in st.session_state.residents_data_by_block[block_choice]]:
            # Robustly handle both dict and list types for PTO requests
            if isinstance(st.session_state.pto_requests_by_block[block_choice], dict):
                requests = st.session_state.pto_requests_by_block[block_choice].get(resident, [])
            elif isinstance(st.session_state.pto_requests_by_block[block_choice], list):
                requests = [req for req in st.session_state.pto_requests_by_block[block_choice] if req.get('Resident') == resident]
            else:
                requests = []
            
            st.markdown(f"### {resident} Hard Constraints")
            for idx, req in enumerate(requests):
                col1, col2, col3 = st.columns([1, 1, 0.2])
                min_date = block_start
                max_date = block_end
                raw_start = req.get('Start_Date', min_date)
                raw_end = req.get('End_Date', min_date)
                # Convert to date objects before comparison
                raw_start = ensure_date(raw_start, min_date)
                raw_end = ensure_date(raw_end, min_date)
                # Now safe to compare since both are date objects
                safe_start = raw_start if min_date <= raw_start <= max_date else min_date
                safe_end = raw_end if min_date <= raw_end <= max_date else min_date
                with col1:
                    start_date = st.date_input(
                        f"Start Date for {resident} #{idx+1}",
                        value=safe_start,
                        min_value=min_date,
                        max_value=max_date,
                        key=f"pto_start_{resident}_{idx}"
                    )
                with col2:
                    end_date = st.date_input(
                        f"End Date for {resident} #{idx+1}",
                        value=safe_end,
                        min_value=min_date,
                        max_value=max_date,
                        key=f"pto_end_{resident}_{idx}"
                    )
                with col3:
                    if st.button("❌", key=f"remove_pto_{resident}_{idx}"):
                        st.session_state.pending_pto_action = ("remove", resident, idx)
                # Update the request in session state
                if isinstance(st.session_state.pto_requests_by_block[block_choice], dict):
                    st.session_state.pto_requests_by_block[block_choice][resident][idx] = {
                        "Start_Date": start_date.strftime("%Y-%m-%d"),
                        "End_Date": end_date.strftime("%Y-%m-%d")
                    }
            if st.button(f"Add Another Hard Constraint for {resident}"):
                st.session_state.pending_pto_action = ("add", resident)

# Process any pending PTO actions after the UI is rendered
if st.session_state.pending_pto_action:
    action_type, resident, *args = st.session_state.pending_pto_action
    if action_type == "add":
        add_pto_for_resident(resident)
    elif action_type == "remove":
        remove_pto_for_resident(resident, args[0])
    st.session_state.pending_pto_action = None
    st.rerun()

with tabs[3]:
    st.subheader("Soft Constraints")
    st.info("Use this section to specify requests that are preferred but not strictly required.")

    # --- Add a fetch button for Non-PTO requests only ---
    if st.button("Fetch Non-PTO Requests from Gmail"):
        _, non_pto_requests = fetch_requests_from_gmail()  # Only use Non-PTO requests here
        print('DEBUG NON-PTO (button):', non_pto_requests)
        # Only keep requests with Reason == 'Non-call' or 'Non-PTO' and within block date range
        block_start_str = block_start.strftime("%Y-%m-%d")
        block_end_str = block_end.strftime("%Y-%m-%d")
        non_pto_only = [req for req in non_pto_requests if req.get("Reason", "").lower() in ["non-call", "non-pto"] and is_within_block(req["Start_Date"], req["End_Date"], block_start_str, block_end_str)]
        # Match by last name only (case-insensitive)
        valid_residents = {res["Name"].strip().lower(): res["Name"] for res in st.session_state.residents_data_by_block[block_choice]}
        last_name_map = {res["Name"].strip().lower(): res["Name"] for res in st.session_state.residents_data_by_block[block_choice]}
        grouped_soft = {}
        skipped_non_pto_names = []
        for req in non_pto_only:
            req_last = req["Resident"].strip().split()[-1].lower()
            match = None
            for lname, canonical in last_name_map.items():
                if req_last == lname:
                    match = canonical
                    break
            if match:
                grouped_soft.setdefault(match, []).append({
                    "Start_Date": req["Start_Date"],
                    "End_Date": req["End_Date"]
                })
            else:
                skipped_non_pto_names.append(req["Resident"])
        # Merge with existing soft constraints instead of replacing them
        if not isinstance(st.session_state.soft_constraints_by_block[block_choice], dict):
            st.session_state.soft_constraints_by_block[block_choice] = {}
        for resident, requests in grouped_soft.items():
            if resident not in st.session_state.soft_constraints_by_block[block_choice]:
                st.session_state.soft_constraints_by_block[block_choice][resident] = []
            # Add new requests while avoiding duplicates
            existing_dates = {(req["Start_Date"], req["End_Date"]) for req in st.session_state.soft_constraints_by_block[block_choice][resident]}
            for req in requests:
                if (req["Start_Date"], req["End_Date"]) not in existing_dates:
                    st.session_state.soft_constraints_by_block[block_choice][resident].append(req)
        msg = f"Fetched and merged Non-PTO requests for {len(grouped_soft)} residents from Gmail."
        if skipped_non_pto_names:
            msg += f" Skipped Non-PTO requests for: {', '.join(skipped_non_pto_names)}."
        st.success(msg)

    def add_soft_constraint_for_resident(resident):
        st.session_state.soft_constraints_by_block[block_choice].setdefault(resident, []).append({
            "Start_Date": "",
            "End_Date": ""
        })

    def remove_soft_constraint_for_resident(resident, idx):
        if resident in st.session_state.soft_constraints_by_block[block_choice] and 0 <= idx < len(st.session_state.soft_constraints_by_block[block_choice][resident]):
            st.session_state.soft_constraints_by_block[block_choice][resident].pop(idx)
            if not st.session_state.soft_constraints_by_block[block_choice][resident]:
                del st.session_state.soft_constraints_by_block[block_choice][resident]

    st.session_state.disable_soft_constraints_by_block[block_choice] = st.checkbox("Disable Soft Constraints", value=st.session_state.disable_soft_constraints_by_block.get(block_choice, False))

    if not st.session_state.disable_soft_constraints_by_block[block_choice]:
        if 'soft_constraints_by_block' not in st.session_state or not isinstance(st.session_state.soft_constraints_by_block[block_choice], dict):
            st.session_state.soft_constraints_by_block[block_choice] = {}
        for resident in [res["Name"] for res in st.session_state.residents_data_by_block[block_choice]]:
            requests = st.session_state.soft_constraints_by_block[block_choice].get(resident, [])
            st.markdown(f"### {resident} Soft Constraints")
            for idx, req in enumerate(requests):
                col1, col2, col3, col4 = st.columns([1, 0.2, 1, 0.2])
                min_date = block_start
                max_date = block_end
                raw_start = req.get('Start_Date', min_date)
                raw_end = req.get('End_Date', min_date)
                prev_start = req.get('_prev_start', raw_start)
                raw_start = ensure_date(raw_start, min_date)
                raw_end = ensure_date(raw_end, min_date)
                prev_start = ensure_date(prev_start, min_date)
                safe_start = raw_start if min_date <= raw_start <= max_date else min_date
                safe_end = raw_end if min_date <= raw_end <= max_date else safe_start
                with col1:
                    start_date = st.date_input(
                        f"Start Date for {resident} #{idx+1}",
                        value=safe_start,
                        min_value=min_date,
                        max_value=max_date,
                        key=f"soft_start_{resident}_{idx}"
                    )
                with col2:
                    if st.button("🔄", key=f"copy_soft_{resident}_{idx}"):
                        st.session_state.soft_constraints_by_block[block_choice][resident][idx]["End_Date"] = start_date.strftime("%Y-%m-%d")
                        st.session_state.soft_constraints_by_block[block_choice][resident][idx]["_prev_start"] = start_date
                        st.rerun()
                with col3:
                    end_date = st.date_input(
                        f"End Date for {resident} #{idx+1}",
                        value=safe_end,
                        min_value=min_date,
                        max_value=max_date,
                        key=f"soft_end_{resident}_{idx}"
                    )
                with col4:
                    if st.button("❌", key=f"remove_soft_{resident}_{idx}"):
                        st.session_state.pending_soft_action = ("remove", resident, idx)
                # Update the request in session state, and track previous start
                st.session_state.soft_constraints_by_block[block_choice][resident][idx] = {
                    "Start_Date": start_date.strftime("%Y-%m-%d"),
                    "End_Date": end_date.strftime("%Y-%m-%d"),
                    "_prev_start": safe_start  # hidden field to track previous start
                }
            if st.button(f"Add Another Soft Constraint for {resident}"):
                st.session_state.pending_soft_action = ("add", resident)

# Process any pending soft constraint actions after the UI is rendered
if st.session_state.pending_soft_action:
    action_type, resident, *args = st.session_state.pending_soft_action
    if action_type == "add":
        add_soft_constraint_for_resident(resident)
    elif action_type == "remove":
        remove_soft_constraint_for_resident(resident, args[0])
    st.session_state.pending_soft_action = None
    st.rerun()

with tabs[4]:
    st.subheader("Previous Block End Assignments")
    
    if not block_info[block_choice]['requires_previous']:
        st.info("No previous block assignments needed for Block 1.")
    else:
        st.info("Please enter the call assignments for the last 4 days of the previous block to maintain spacing rules.")
        
        # Calculate the dates (last 4 days of previous block)
        prev_block_end = block_start_dt - timedelta(days=1)
        dates = [(prev_block_end - timedelta(days=i)) for i in range(3, -1, -1)]
        
        # Initialize previous assignments in session state if not exists
        if 'previous_assignments_by_block' not in st.session_state:
            st.session_state.previous_assignments_by_block = {}
        
        # Create a list to store new assignments
        new_previous_assignments = []
        
        # Create input fields for each date
        for idx, date in enumerate(dates):
            date_str = date.strftime("%Y-%m-%d")
            st.markdown(f"**{date_str}**")
            col1, col2 = st.columns(2)
            with col1:
                default_call = st.session_state.previous_assignments_by_block[block_choice][idx]['Call'] if idx < len(st.session_state.previous_assignments_by_block[block_choice]) else ""
                call = st.selectbox(
                    "Call Resident",
                    [""] + [res["Name"] for res in st.session_state.residents_data_by_block[block_choice]],
                    index=0 if default_call == "" else [res["Name"] for res in st.session_state.residents_data_by_block[block_choice]].index(default_call) + 1 if default_call in [res["Name"] for res in st.session_state.residents_data_by_block[block_choice]] else 0,
                    key=f"prev_call_{date_str}"
                )
            with col2:
                default_backup = st.session_state.previous_assignments_by_block[block_choice][idx]['Backup'] if idx < len(st.session_state.previous_assignments_by_block[block_choice]) else ""
                backup = st.selectbox(
                    "Backup Resident",
                    [""] + [res["Name"] for res in st.session_state.residents_data_by_block[block_choice]],
                    index=0 if default_backup == "" else [res["Name"] for res in st.session_state.residents_data_by_block[block_choice]].index(default_backup) + 1 if default_backup in [res["Name"] for res in st.session_state.residents_data_by_block[block_choice]] else 0,
                    key=f"prev_backup_{date_str}"
                )
            
            if call and backup:
                new_previous_assignments.append({
                    "Date": date_str,
                    "Call": call,
                    "Backup": backup
                })
            else:
                new_previous_assignments.append({
                    "Date": date_str,
                    "Call": call,
                    "Backup": backup
                })
        
        # Update session state
        st.session_state.previous_assignments_by_block[block_choice] = new_previous_assignments

with tabs[5]:
    st.subheader("Previous Call Counts")
    
    if block_choice == "Block 1":
        st.info("No previous call counts needed for Block 1.")
    else:
        st.info("Please upload the call statistics CSV file(s) from the previous block(s). For Block 2, upload Block 1's statistics. For Block 3, upload both Block 1 and Block 2's statistics.")
        
        # Initialize previous call counts in session state if not exists
        if 'previous_call_counts_by_block' not in st.session_state:
            st.session_state.previous_call_counts_by_block = {}
        
        # File upload section (CSV only)
        uploaded_files = st.file_uploader(
            "Upload previous block call statistics (CSV)",
            type=['csv'],
            accept_multiple_files=True if block_choice == "Block 3" else False
        )
        
        if uploaded_files:
            try:
                all_counts = {}
                # Always treat uploaded_files as a list for uniform processing
                files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
                for file in files:
                    # Handle both file-like and bytes objects
                    if hasattr(file, 'seek'):
                        file.seek(0)
                        df = pd.read_csv(file)
                    else:
                        # file is bytes, decode to string and use StringIO
                        s = file.decode('utf-8')
                        df = pd.read_csv(io.StringIO(s))
                    # Ensure all expected columns are present
                    for col in ['Resident', 'Weekday', 'Fridays', 'Saturday', 'Sunday', 'Total']:
                        if col not in df.columns:
                            df[col] = 0
                    for _, row in df.iterrows():
                        resident = norm_name(row['Resident'])
                        display_name = str(row['Resident']).strip()
                        if resident not in all_counts:
                            all_counts[resident] = {
                                'display_name': display_name,
                                'Weekday': 0,
                                'Fridays': 0,
                                'Saturday': 0,
                                'Sunday': 0,
                                'Total': 0
                            }
                        all_counts[resident]['Weekday'] += safe_int_nan(row.get('Weekday', 0))
                        all_counts[resident]['Fridays'] += safe_int_nan(row.get('Fridays', 0))
                        all_counts[resident]['Saturday'] += safe_int_nan(row.get('Saturday', 0))
                        all_counts[resident]['Sunday'] += safe_int_nan(row.get('Sunday', 0))
                        all_counts[resident]['Total'] += safe_int_nan(row.get('Total', 0))
                st.session_state.previous_call_counts_by_block[block_choice] = all_counts
                st.success("Successfully processed previous call counts!")
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                st.error("Please make sure you're uploading the correct call statistics CSV files.")
        else:
            st.warning("Please upload the call statistics CSV file(s) from the previous block(s).")

    # After the previous call counts upload section, add a new tab for Block 2 and 3 to display the uploaded previous call counts
    if block_choice in ["Block 2", "Block 3"]:
        prev_counts = st.session_state.previous_call_counts_by_block.get(block_choice, {})
        if prev_counts:
            st.markdown("### Uploaded Previous Call Counts")
            # Organize by PGY for display
            prev_by_pgy = {1: [], 2: [], 3: [], 4: []}
            for resident, stats in prev_counts.items():
                pgy = stats.get('PGY', None)
                # If PGY is not in the uploaded file, try to infer from current block's resident list
                if pgy is None:
                    for res in st.session_state.residents_data_by_block[block_choice]:
                        if norm_name(res['Name']) == resident:
                            pgy = int(res['PGY'])
                            break
                if pgy is None:
                    continue
                row = {
                    'Resident': stats.get('display_name', resident),
                    'Weekday': stats.get('Weekday', 0),
                    'Fridays': stats.get('Fridays', 0),
                    'Saturday': stats.get('Saturday', 0),
                    'Sunday': stats.get('Sunday', 0),
                    'Total': stats.get('Total', 0)
                }
                prev_by_pgy[pgy].append(row)
            for pgy in [1, 2, 3, 4]:
                if prev_by_pgy[pgy]:
                    st.markdown(f"#### PGY-{pgy} Previous Call Counts")
                    df = pd.DataFrame(prev_by_pgy[pgy])
                    if pgy == 1:
                        display_cols = ['Weekday', 'Saturday', 'Total']
                    elif pgy == 2:
                        display_cols = ['Weekday', 'Fridays', 'Sunday', 'Total']
                    elif pgy == 3:
                        display_cols = ['Weekday', 'Saturday', 'Total']
                    elif pgy == 4:
                        display_cols = ['Total']
                    st.dataframe(df.set_index('Resident')[display_cols], use_container_width=True)

with tabs[6]:
    st.subheader("Generate & Review")
    # --- Review Checklist ---
    checklist = []
    all_complete = True
    missing_items = []

    # Residents
    resident_count = len(st.session_state.residents_data_by_block[block_choice])
    if resident_count > 0:
        checklist.append(f"✅ Residents: {resident_count} entered")
    else:
        checklist.append(f"❌ Residents: None entered")
        all_complete = False
        missing_items.append("Residents")

    # Holidays
    if not st.session_state.disable_holidays_by_block[block_choice]:
        holiday_count = len(st.session_state.holiday_assignments_by_block[block_choice])
        if holiday_count > 0:
            checklist.append(f"✅ Holidays: {holiday_count} entered")
        else:
            checklist.append(f"❌ Holidays: None entered")
            all_complete = False
            missing_items.append("Holidays")
    else:
        checklist.append("ℹ️ Holidays: Disabled")

    # Hard Constraints (PTO)
    if not st.session_state.disable_pto_by_block[block_choice]:
        pto_count = sum(len(v) for v in st.session_state.pto_requests_by_block[block_choice].values()) if isinstance(st.session_state.pto_requests_by_block[block_choice], dict) else 0
        checklist.append(f"ℹ️ Hard Constraints (PTO): {pto_count} requests (optional)")
    else:
        checklist.append("ℹ️ Hard Constraints (PTO): Disabled")

    # Soft Constraints
    if not st.session_state.disable_soft_constraints_by_block[block_choice]:
        soft_count = sum(len(v) for v in st.session_state.soft_constraints_by_block[block_choice].values()) if isinstance(st.session_state.soft_constraints_by_block[block_choice], dict) else 0
        checklist.append(f"ℹ️ Soft Constraints: {soft_count} requests (optional)")
    else:
        checklist.append("ℹ️ Soft Constraints: Disabled")

    # Previous Block (if required)
    if block_info[block_choice]['requires_previous']:
        prev_assignments = st.session_state.previous_assignments_by_block[block_choice]
        if len(prev_assignments) == 4 and all(a.get('Call') and a.get('Backup') for a in prev_assignments):
            checklist.append("✅ Previous Block: Complete")
        else:
            checklist.append("❌ Previous Block: Incomplete")
            all_complete = False
            missing_items.append("Previous Block Assignments")
    else:
        checklist.append("ℹ️ Previous Block: Not required")

    # Previous Call Counts (if required)
    if block_choice == "Block 2":
        prev_counts = st.session_state.previous_call_counts_by_block.get(block_choice, {})
        uploaded = 1 if prev_counts else 0
        if uploaded == 1:
            checklist.append("✅ Previous Call Counts: 1/1 uploaded")
        else:
            checklist.append("❌ Previous Call Counts: 0/1 uploaded")
            all_complete = False
            missing_items.append("Previous Call Counts")
    elif block_choice == "Block 3":
        prev1 = st.session_state.previous_call_counts_by_block.get("Block 1", {})
        prev2 = st.session_state.previous_call_counts_by_block.get("Block 2", {})
        uploaded = int(bool(prev1)) + int(bool(prev2))
        if uploaded == 2:
            checklist.append("✅ Previous Call Counts: 2/2 uploaded")
        else:
            checklist.append(f"❌ Previous Call Counts: {uploaded}/2 uploaded")
            all_complete = False
            missing_items.append("Previous Call Counts")
    else:
        checklist.append("ℹ️ Previous Call Counts: Not required")

    # Display checklist as a vertical bullet list
    st.markdown("\n".join([f"- {item}" for item in checklist]))

    # Show missing items if not complete
    if not all_complete:
        st.warning(f"Please complete the following before generating the schedule: {', '.join(missing_items)}")

    # --- On Generate Schedule Button Press ---
    if st.button("Generate Schedule", disabled=not all_complete):
        # Clear previous results for this block only
        for key in [
            'last_schedule_df_by_block', 'last_stats_by_block', 'last_excel_file_by_block', 'last_block_name_by_block', 'last_success_by_block',
            'last_call_distribution_by_block', 'last_pgy_stats_by_block', 'last_csv_buffer_by_block', 'last_soft_constraint_stats_by_block', 'show_results_by_block']:
            if key in st.session_state and block_choice in st.session_state[key]:
                del st.session_state[key][block_choice]

        if not st.session_state.residents_data_by_block[block_choice]:
            st.error("Please enter at least one resident.")
        elif block_info[block_choice]['requires_previous'] and len(st.session_state.previous_assignments_by_block[block_choice]) < 4:
            st.error("Please enter all previous block assignments to maintain proper spacing rules.")
        elif block_end_dt <= block_start_dt:
            st.error("End date must be after start date")
        else:
            with st.spinner("Generating schedule..."):
                try:
                    # Transform the form data into the format expected by the engine
                    residents_df = pd.DataFrame([{
                        'Resident': res['Name'],
                        'PGY': res['PGY'],
                        'Transition Date': res['Transition_Date'],
                        'Transition PGY': min(int(res['PGY']) + 1, 4) if res['Transition_Date'] else None
                    } for res in st.session_state.residents_data_by_block[block_choice]])
                    # Validate transition dates are within the block
                    for _, row in residents_df.iterrows():
                        if pd.notna(row['Transition Date']):
                            trans_val = row['Transition Date']
                            if isinstance(trans_val, (dt_type, date_type)):
                                trans_date = trans_val
                            else:
                                trans_date = dt_type.strptime(str(trans_val), "%Y-%m-%d")
                            bsd = block_start_dt
                            bed = block_end_dt
                            if isinstance(bsd, date_type) and not isinstance(bsd, dt_type):
                                bsd = dt_type.combine(bsd, dt_type.min.time())
                            if isinstance(bed, date_type) and not isinstance(bed, dt_type):
                                bed = dt_type.combine(bed, dt_type.min.time())
                            if isinstance(trans_date, date_type) and not isinstance(trans_date, dt_type):
                                trans_date = dt_type.combine(trans_date, dt_type.min.time())
                            if not (bsd <= trans_date <= bed):
                                st.warning(f"Transition date for {row['Resident']} ({row['Transition Date']}) is outside the selected block period.")
                    # Create previous assignments DataFrame if needed
                    prev_df = None
                    if block_info[block_choice]['requires_previous'] and st.session_state.previous_assignments_by_block[block_choice]:
                        prev_df = pd.DataFrame(st.session_state.previous_assignments_by_block[block_choice])
                    # Transform holiday assignments into the format expected by the engine
                    holidays_df = pd.DataFrame([{
                        'Date': holiday['Date'],
                        'Call': holiday['Call'],
                        'Backup': holiday['Backup']
                    } for holiday in st.session_state.holiday_assignments_by_block[block_choice]]) if not st.session_state.disable_holidays_by_block[block_choice] and st.session_state.holiday_assignments_by_block[block_choice] else pd.DataFrame(columns=['Date', 'Call', 'Backup'])
                    # Transform PTO requests into the format expected by the engine
                    pto_df = pd.DataFrame()
                    if not st.session_state.disable_pto_by_block[block_choice] and st.session_state.pto_requests_by_block[block_choice]:
                        pto_data = []
                        for resident, requests in st.session_state.pto_requests_by_block[block_choice].items():
                            for req in requests:
                                pto_data.append({
                                    'Resident': resident,
                                    'Start Date': req['Start_Date'],
                                    'End Date': req['End_Date']
                                })
                        pto_df = pd.DataFrame(pto_data)
                    # Transform soft constraints into the format expected by the engine
                    soft_constraints_df = pd.DataFrame()
                    if not st.session_state.disable_soft_constraints_by_block[block_choice] and st.session_state.soft_constraints_by_block[block_choice]:
                        soft_data = []
                        for resident, requests in st.session_state.soft_constraints_by_block[block_choice].items():
                            for req in requests:
                                soft_data.append({
                                    'Resident': resident,
                                    'Start Date': req['Start_Date'],
                                    'End Date': req['End_Date']
                                })
                        soft_constraints_df = pd.DataFrame(soft_data)
                    # Fix previous_call_counts logic for each block
                    if block_choice == "Block 2":
                        block1_prev = st.session_state.previous_call_counts_by_block.get(block_choice, {})
                        prev_counts_for_engine = {}
                        for res in st.session_state.residents_data_by_block[block_choice]:
                            name = res['Name']
                            norm = norm_name(name)
                            if norm in block1_prev:
                                prev_counts_for_engine[norm] = {
                                    "Weekday": block1_prev[norm].get("Weekday", 0),
                                    "Fridays": block1_prev[norm].get("Fridays", 0),
                                    "Saturday": block1_prev[norm].get("Saturday", 0),
                                    "Sunday": block1_prev[norm].get("Sunday", 0),
                                    "Total": block1_prev[norm].get("Total", 0)
                                }
                            else:
                                prev_counts_for_engine[norm] = {
                                    "Weekday": 0,
                                    "Fridays": 0,
                                    "Saturday": 0,
                                    "Sunday": 0,
                                    "Total": 0
                                }
                    elif block_choice == "Block 3":
                        prev1 = st.session_state.previous_call_counts_by_block.get("Block 1", {})
                        prev2 = st.session_state.previous_call_counts_by_block.get("Block 2", {})
                        prev_counts_for_engine = {}
                        for res in st.session_state.residents_data_by_block[block_choice]:
                            name = res['Name']
                            norm = norm_name(name)
                            prev_counts_for_engine[norm] = {
                                "Weekday": prev1.get(norm, {}).get("Weekday", 0) + prev2.get(norm, {}).get("Weekday", 0),
                                "Fridays": prev1.get(norm, {}).get("Fridays", 0) + prev2.get(norm, {}).get("Fridays", 0),
                                "Saturday": prev1.get(norm, {}).get("Saturday", 0) + prev2.get(norm, {}).get("Saturday", 0),
                                "Sunday": prev1.get(norm, {}).get("Sunday", 0) + prev2.get(norm, {}).get("Sunday", 0),
                                "Total": prev1.get(norm, {}).get("Total", 0) + prev2.get(norm, {}).get("Total", 0)
                            }
                    else:
                        prev_counts_for_engine = None
                    # Run the scheduling engine with selected dates and pgy4_cap
                    schedule_df = run_scheduling_engine(
                        prev_df,
                        residents_df,
                        pto_df,
                        holidays_df,
                        block_start_dt,
                        block_end_dt,
                        pgy4_cap=pgy4_cap,
                        previous_call_counts=prev_counts_for_engine,
                        soft_constraints=soft_constraints_df,
                        fairness_weight=fairness_weight,
                        soft_constraint_weight=soft_constraint_weight
                    )
                    # Get soft constraint statistics
                    soft_constraint_stats = schedule_df.attrs.get('soft_constraint_stats', {})
                    # Calculate call distribution
                    call_distribution = calculate_call_distribution(schedule_df, block_end_dt)
                    # Convert call distribution to pgy_stats format
                    pgy_stats = {1: [], 2: [], 3: [], 4: []}
                    for _, row in call_distribution.iterrows():
                        pgy = int(row['PGY'])
                        stats = {
                            'Resident': row['Resident'],
                            'Weekday': row.get('Weekday', 0),
                            'Fridays': row.get('Fridays', 0),
                            'Saturday': row.get('Saturday', 0),
                            'Sunday': row.get('Sunday', 0),
                            'Total': row['Total']
                        }
                        pgy_stats[pgy].append(stats)
                    # Format the schedule
                    wb = format_schedule(schedule_df, schedule_df, schedule_df)  # Using same schedule for all blocks
                    # Save to BytesIO
                    excel_file = BytesIO()
                    wb.save(excel_file)
                    excel_file.seek(0)
                    # Store in session state
                    st.session_state['last_schedule_df_by_block'][block_choice] = schedule_df
                    st.session_state['last_stats_by_block'][block_choice] = pgy_stats
                    st.session_state['last_excel_file_by_block'][block_choice] = excel_file.getvalue()
                    st.session_state['last_block_name_by_block'][block_choice] = block_choice.lower().replace(' ', '_')
                    st.session_state['last_success_by_block'][block_choice] = True
                    st.session_state['last_call_distribution_by_block'][block_choice] = call_distribution
                    st.session_state['last_pgy_stats_by_block'][block_choice] = pgy_stats
                    st.session_state['last_csv_buffer_by_block'][block_choice] = call_distribution.to_csv(index=False)
                    st.session_state['last_soft_constraint_stats_by_block'][block_choice] = soft_constraint_stats
                    st.session_state['show_results_by_block'][block_choice] = True
                except Exception as e:
                    st.session_state['last_success_by_block'][block_choice] = False
                    st.session_state['show_results_by_block'][block_choice] = False
                    st.error(f"An error occurred: {str(e)}")
                    print("=== FULL TRACEBACK ===")
                    traceback.print_exc()

    # --- Display call breakdown and download buttons if present in session state ---
    if st.session_state['show_results_by_block'].get(block_choice) and \
       block_choice in st.session_state['last_call_distribution_by_block'] and \
       block_choice in st.session_state['last_pgy_stats_by_block'] and \
       block_choice in st.session_state['last_excel_file_by_block'] and \
       block_choice in st.session_state['last_csv_buffer_by_block']:
        call_distribution = st.session_state['last_call_distribution_by_block'][block_choice]
        pgy_stats = st.session_state['last_pgy_stats_by_block'][block_choice]
        excel_file_bytes = st.session_state['last_excel_file_by_block'][block_choice]
        csv_buffer_val = st.session_state['last_csv_buffer_by_block'][block_choice]
        soft_constraint_stats = st.session_state['last_soft_constraint_stats_by_block'].get(block_choice, {})
        # Fix previous_counts for running total tab
        if block_choice in ["Block 2", "Block 3"]:
            previous_counts = st.session_state.previous_call_counts_by_block.get(block_choice, {})
        else:
            previous_counts = {}
        # Tabs: Call Distribution, Running Total (if applicable), Soft Constraint Results, Download
        show_running_total = block_choice in ["Block 2", "Block 3"]
        if show_running_total:
            tab1, tab2, tab3, tab4 = st.tabs(["Call Distribution", "Running Total", "Soft Constraint Results", "Download"])
        else:
            tab1, tab3, tab4 = st.tabs(["Call Distribution", "Soft Constraint Results", "Download"])
            tab2 = None
        with tab1:
            for pgy in [1, 2, 3, 4]:
                st.markdown(f"### PGY-{pgy}")
                pgy_df = call_distribution[call_distribution['PGY'] == pgy]
                if not pgy_df.empty:
                    if pgy == 1:
                        display_cols = ['Weekday', 'Saturday', 'Total']
                    elif pgy == 2:
                        display_cols = ['Weekday', 'Fridays', 'Sunday', 'Total']
                    elif pgy == 3:
                        display_cols = ['Weekday', 'Saturday', 'Total']
                    elif pgy == 4:
                        display_cols = ['Total']
                    st.dataframe(pgy_df.set_index('Resident')[display_cols], use_container_width=True)
                else:
                    st.info(f"No PGY-{pgy} residents")
        if show_running_total and tab2 is not None:
            with tab2:
                st.subheader("Running Total (All Blocks)")
                running_totals = {1: [], 2: [], 3: [], 4: []}
                for pgy in [1, 2, 3, 4]:
                    current_stats = {norm_name(s['Resident']): s for s in pgy_stats[pgy]}
                    prev_stats = {}
                    for k, v in previous_counts.items():
                        prev_pgy = v.get('PGY')
                        if prev_pgy is None:
                            for res in st.session_state.residents_data_by_block[block_choice]:
                                if norm_name(res['Name']) == k:
                                    prev_pgy = int(res['PGY'])
                                    break
                        if prev_pgy == pgy:
                            prev_stats[k] = v
                    all_names = set(current_stats.keys()) | set(prev_stats.keys())
                    for name in all_names:
                        display_name = current_stats.get(name, prev_stats.get(name, {})).get('Resident', prev_stats.get(name, {}).get('display_name', name))
                        weekday = safe_int_nan(current_stats.get(name, {}).get('Weekday', 0)) + safe_int_nan(prev_stats.get(name, {}).get('Weekday', 0))
                        fridays = safe_int_nan(current_stats.get(name, {}).get('Fridays', 0)) + safe_int_nan(prev_stats.get(name, {}).get('Fridays', 0))
                        saturday = safe_int_nan(current_stats.get(name, {}).get('Saturday', 0)) + safe_int_nan(prev_stats.get(name, {}).get('Saturday', 0))
                        sunday = safe_int_nan(current_stats.get(name, {}).get('Sunday', 0)) + safe_int_nan(prev_stats.get(name, {}).get('Sunday', 0))
                        total = safe_int_nan(current_stats.get(name, {}).get('Total', 0)) + safe_int_nan(prev_stats.get(name, {}).get('Total', 0))
                        total_stats = {
                            'Resident': display_name,
                            'Weekday': weekday,
                            'Fridays': fridays,
                            'Saturday': saturday,
                            'Sunday': sunday,
                            'Total': total
                        }
                        running_totals[pgy].append(total_stats)
                for pgy in [1, 2, 3, 4]:
                    st.markdown(f"### PGY-{pgy} Running Total")
                    if running_totals[pgy]:
                        df = pd.DataFrame(running_totals[pgy])
                        if pgy == 1:
                            display_cols = ['Weekday', 'Saturday', 'Total']
                        elif pgy == 2:
                            display_cols = ['Weekday', 'Fridays', 'Sunday', 'Total']
                        elif pgy == 3:
                            display_cols = ['Weekday', 'Saturday', 'Total']
                        elif pgy == 4:
                            display_cols = ['Total']
                        st.dataframe(df.set_index('Resident')[display_cols], use_container_width=True)
                    else:
                        st.info(f"No PGY-{pgy} residents")
        with tab3:
            if soft_constraint_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Soft Constraints", soft_constraint_stats['total_constraints'])
                with col2:
                    st.metric("Fulfilled Constraints", soft_constraint_stats['fulfilled'])
                with col3:
                    st.metric("Violated Constraints", soft_constraint_stats['violations'])
                if soft_constraint_stats['violation_details']:
                    st.subheader("Violation Details")
                    violations_df = pd.DataFrame(soft_constraint_stats['violation_details'])
                    st.dataframe(violations_df, use_container_width=True)
            else:
                st.info("No soft constraints were provided for this schedule.")
        with tab4:
            st.markdown("### Download Schedule and Call Statistics")
            st.download_button(
                label="Download Schedule",
                data=excel_file_bytes,
                file_name=f"call_schedule_{block_choice.lower().replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_schedule_{block_choice.lower().replace(' ', '_')}_downloadtab"
            )
            st.download_button(
                label="Download Call Statistics (CSV)",
                data=csv_buffer_val,
                file_name=f"call_statistics_{block_choice.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"download_csv_{block_choice.lower().replace(' ', '_')}_downloadtab"
            ) 
