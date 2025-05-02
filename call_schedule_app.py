# call_schedule_app.py

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import json
import os
from scheduling_engine import run_scheduling_engine
from run_formatter import format_schedule
from openpyxl import Workbook

# Initialize all session state variables
if 'residents_data' not in st.session_state:
    st.session_state.residents_data = []
if 'holiday_assignments_list' not in st.session_state:
    st.session_state.holiday_assignments_list = []
if 'pto_requests' not in st.session_state:
    st.session_state.pto_requests = []
if 'soft_constraints' not in st.session_state:
    st.session_state.soft_constraints = []
if 'resident_count' not in st.session_state:
    st.session_state.resident_count = 1
if 'holiday_count' not in st.session_state:
    st.session_state.holiday_count = 1
if 'pto_count' not in st.session_state:
    st.session_state.pto_count = 1
if 'soft_constraint_count' not in st.session_state:
    st.session_state.soft_constraint_count = 1
if 'current_academic_year' not in st.session_state:
    st.session_state.current_academic_year = None
if 'loaded_residents' not in st.session_state:
    st.session_state.loaded_residents = []
if 'disable_holidays' not in st.session_state:
    st.session_state.disable_holidays = False
if 'disable_pto' not in st.session_state:
    st.session_state.disable_pto = False
if 'disable_soft_constraints' not in st.session_state:
    st.session_state.disable_soft_constraints = False
if 'removed_residents' not in st.session_state:
    st.session_state.removed_residents = set()
if 'removed_holidays' not in st.session_state:
    st.session_state.removed_holidays = set()
if 'removed_pto' not in st.session_state:
    st.session_state.removed_pto = set()
if 'removed_soft_constraints' not in st.session_state:
    st.session_state.removed_soft_constraints = set()
if 'block_dates' not in st.session_state:
    st.session_state.block_dates = None

def save_data(academic_year):
    """Save only resident information for an academic year"""
    if not os.path.exists('saved_data'):
        os.makedirs('saved_data')
    
    data = {
        'residents': st.session_state.residents_data
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

st.set_page_config(page_title="Kall Scheduler Kuhnel (KSK)", layout="wide")

st.markdown("""
<h2 style='text-align: center;'>Kall Scheduler Kuhnel (KSK)</h2>
""", unsafe_allow_html=True)

# Academic Year Selection
current_year = datetime.now().year
year_options = [f"{year}-{year+1}" for year in range(2025, current_year+3)]
academic_year = st.selectbox("Select Academic Year:", year_options)
start_year = int(academic_year.split('-')[0])

# Initialize or update block dates based on academic year
if st.session_state.block_dates is None or st.session_state.current_academic_year != academic_year:
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
    st.session_state.current_academic_year = academic_year
    # Reset holiday assignments when academic year changes
    st.session_state.holiday_assignments_list = []
    st.session_state.holiday_count = 1
    # Reset PTO requests when academic year changes
    st.session_state.pto_requests = []
    st.session_state.pto_count = 1
    st.session_state.removed_pto = set()
    # Reset soft constraints when academic year changes
    st.session_state.soft_constraints = []
    st.session_state.soft_constraint_count = 1
    st.session_state.removed_soft_constraints = set()

# Load saved data when academic year changes
if st.session_state.current_academic_year != academic_year:
    loaded_residents = load_data(academic_year)
    if loaded_residents:
        st.session_state.loaded_residents = loaded_residents
        st.session_state.resident_count = len(loaded_residents)
    st.session_state.current_academic_year = academic_year

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
        value=st.session_state.get('pgy4_cap', 4),
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
        value=datetime.strptime(st.session_state.block_dates[block_choice]["start"], "%Y-%m-%d").date(),
        min_value=datetime(start_year, 7, 1).date(),
        max_value=datetime(start_year+1, 6, 30).date(),
        key=f"block_start_{block_choice}"
    )
with col2:
    block_end = st.date_input(
        "Block End Date",
        value=datetime.strptime(st.session_state.block_dates[block_choice]["end"], "%Y-%m-%d").date(),
        min_value=datetime(start_year, 7, 1).date(),
        max_value=datetime(start_year+1, 6, 30).date(),
        key=f"block_end_{block_choice}"
    )

# Convert to datetime objects for comparison
block_start_dt = datetime.combine(block_start, datetime.min.time())
block_end_dt = datetime.combine(block_end, datetime.min.time())

# Validate dates
if block_end_dt <= block_start_dt:
    st.error("End date must be after start date")
elif block_choice == "Block 1" and block_start != datetime.strptime(f"{start_year}-07-01", "%Y-%m-%d").date():
    st.error("Block 1 must start on July 1st")
elif block_choice == "Block 3" and block_end != datetime.strptime(f"{start_year+1}-06-30", "%Y-%m-%d").date():
    st.error("Block 3 must end on June 30th")

# Store the selected dates in session state
st.session_state.block_dates[block_choice] = {
    "start": block_start.strftime("%Y-%m-%d"),
    "end": block_end.strftime("%Y-%m-%d")
}

# Create tabs for different data entry sections
tabs = st.tabs(["Residents", "Holiday Assignments", "PTO Requests", "Soft Constraints", "Previous Block"])

with tabs[0]:
    st.subheader("Resident Information")
    
    def add_resident():
        st.session_state.resident_count += 1

    def remove_resident(index):
        st.session_state.removed_residents.add(index)
        st.rerun()

    # Initialize residents_data if not already initialized
    if 'residents_data' not in st.session_state:
        st.session_state.residents_data = []
    
    # Temporary list to collect new resident data
    new_residents = []
    
    for i in range(st.session_state.resident_count):
        if i in st.session_state.removed_residents:
            continue
            
        col1, col2, col3, col4 = st.columns([2, 1, 1, 0.5])
        with col1:
            default_name = st.session_state.loaded_residents[i]['Name'] if hasattr(st.session_state, 'loaded_residents') and i < len(st.session_state.loaded_residents) else ""
            name = st.text_input(f"Resident Name #{i+1}", value=default_name, key=f"res_name_{i}")
        with col2:
            default_pgy = st.session_state.loaded_residents[i]['PGY'] if hasattr(st.session_state, 'loaded_residents') and i < len(st.session_state.loaded_residents) else 1
            pgy = st.selectbox(f"PGY Level #{i+1}", [1, 2, 3, 4], index=[1, 2, 3, 4].index(default_pgy), key=f"res_pgy_{i}")
        with col3:
            default_trans = bool(st.session_state.loaded_residents[i].get('Transition_Date')) if hasattr(st.session_state, 'loaded_residents') and i < len(st.session_state.loaded_residents) else False
            is_transitioning = st.checkbox("Transitioning?", value=default_trans, key=f"res_trans_{i}")
        with col4:
            st.button("❌", key=f"remove_res_{i}", on_click=remove_resident, args=(i,))
        
        if is_transitioning:
            default_date = datetime.strptime(st.session_state.loaded_residents[i]['Transition_Date'], "%Y-%m-%d") if hasattr(st.session_state, 'loaded_residents') and i < len(st.session_state.loaded_residents) and st.session_state.loaded_residents[i].get('Transition_Date') else datetime(start_year, 7, 1)
            trans_date = st.date_input(
                "Transition Date",
                value=default_date,
                min_value=datetime(start_year, 7, 1),
                max_value=datetime(start_year+1, 6, 30),
                key=f"res_trans_date_{i}"
            )
        
        if name:  # Only add if name is provided
            new_residents.append({
                "Name": name,
                "PGY": pgy,
                "Transition_Date": trans_date.strftime("%Y-%m-%d") if is_transitioning else None
            })
    
    # Update residents_data with new data
    st.session_state.residents_data = new_residents

    st.button("Add Another Resident", on_click=add_resident)

    # Save Residents button
    if st.button("Save Residents"):
        if st.session_state.residents_data:
            filename = save_data(academic_year)
            st.success(f"Saved resident data for {academic_year}")
        else:
            st.warning("No resident data to save")

with tabs[1]:
    st.subheader("Holiday Call Assignments")
    
    def add_holiday():
        st.session_state.holiday_count += 1

    def remove_holiday(index):
        st.session_state.removed_holidays.add(index)
        st.rerun()

    # Add disable holidays checkbox
    st.session_state.disable_holidays = st.checkbox("Disable Holiday Assignments", value=st.session_state.disable_holidays)

    if not st.session_state.disable_holidays:
        # Initialize holiday_assignments_list if not already initialized
        if 'holiday_assignments_list' not in st.session_state:
            st.session_state.holiday_assignments_list = []
        
        # Filter PGY-2 residents for holiday assignments
        pgy2_residents = [res["Name"] for res in st.session_state.residents_data if res["PGY"] == 2]
        
        # Temporary list to collect new holiday assignments
        new_holiday_assignments = []
        
        for i in range(st.session_state.holiday_count):
            if i in st.session_state.removed_holidays:
                continue
                
            st.markdown(f"### Holiday #{i+1}")
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 0.5])
            
            with col1:
                # Get existing holiday name if available
                default_name = st.session_state.holiday_assignments_list[i].get('Name', "") if i < len(st.session_state.holiday_assignments_list) else ""
                holiday_name = st.text_input("Holiday Name", value=default_name, key=f"holiday_name_{i}")
            with col2:
                # Get existing holiday date if available, otherwise use July 4th of the current academic year
                if i < len(st.session_state.holiday_assignments_list):
                    default_date = datetime.strptime(st.session_state.holiday_assignments_list[i]['Date'], "%Y-%m-%d")
                else:
                    default_date = datetime(start_year, 7, 4)
                holiday_date = st.date_input(
                    "Date",
                    value=default_date,
                    min_value=datetime(start_year, 7, 1),
                    max_value=datetime(start_year+1, 6, 30),
                    key=f"holiday_date_{i}"
                )
            with col3:
                # Get existing call assignment if available
                default_call = st.session_state.holiday_assignments_list[i].get('Call', "") if i < len(st.session_state.holiday_assignments_list) else ""
                call = st.selectbox(
                    "Call Assignment",
                    [""] + pgy2_residents if pgy2_residents else [""],
                    index=0 if default_call == "" else pgy2_residents.index(default_call) + 1 if default_call in pgy2_residents else 0,
                    key=f"holiday_call_{i}"
                )
            with col4:
                # Get existing backup assignment if available
                default_backup = st.session_state.holiday_assignments_list[i].get('Backup', "") if i < len(st.session_state.holiday_assignments_list) else ""
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
        st.session_state.holiday_assignments_list = new_holiday_assignments

        col1, col2 = st.columns([6, 1])
        with col1:
            st.button("Add Another Holiday", on_click=add_holiday)
        with col2:
            if st.button("Clear Holidays"):
                st.session_state.holiday_count = 1
                st.session_state.holiday_assignments_list = []
                st.session_state.removed_holidays = set()
                st.rerun()

with tabs[2]:
    st.subheader("PTO Requests")
    
    def add_pto():
        st.session_state.pto_count += 1

    def remove_pto(index):
        st.session_state.removed_pto.add(index)
        st.rerun()

    # Add disable PTO checkbox
    st.session_state.disable_pto = st.checkbox("Disable PTO Requests", value=st.session_state.disable_pto)

    if not st.session_state.disable_pto:
        # Initialize pto_requests if not already initialized
        if 'pto_requests' not in st.session_state:
            st.session_state.pto_requests = []
        
        # Temporary list to collect new PTO requests
        new_pto_requests = []
        
        for i in range(st.session_state.pto_count):
            if i in st.session_state.removed_pto:
                continue
            
            st.markdown(f"### PTO Request #{i+1}")
            col1, col2, col3, col4 = st.columns([2, 1, 1, 0.5])
            with col1:
                default_res = st.session_state.pto_requests[i]['Resident'] if i < len(st.session_state.pto_requests) else (st.session_state.residents_data[0]['Name'] if st.session_state.residents_data else "")
                resident = st.selectbox(
                    "Resident",
                    [res["Name"] for res in st.session_state.residents_data],
                    index=[res["Name"] for res in st.session_state.residents_data].index(default_res) if default_res in [res["Name"] for res in st.session_state.residents_data] else 0,
                    key=f"pto_res_{i}"
                )
            with col2:
                # Get existing PTO start date if available, otherwise use start of academic year
                if i < len(st.session_state.pto_requests):
                    default_start = datetime.strptime(st.session_state.pto_requests[i]['Start_Date'], "%Y-%m-%d")
                else:
                    default_start = datetime(start_year, 7, 1)
                start_date = st.date_input(
                    "Start Date",
                    value=default_start,
                    min_value=datetime(start_year, 7, 1).date(),
                    max_value=datetime(start_year+1, 6, 30).date(),
                    key=f"pto_start_{i}"
                )
            with col3:
                # Get existing PTO end date if available, otherwise use start date
                if i < len(st.session_state.pto_requests):
                    default_end = datetime.strptime(st.session_state.pto_requests[i]['End_Date'], "%Y-%m-%d")
                else:
                    default_end = default_start
                end_date = st.date_input(
                    "End Date",
                    value=default_end,
                    min_value=datetime(start_year, 7, 1).date(),
                    max_value=datetime(start_year+1, 6, 30).date(),
                    key=f"pto_end_{i}"
                )
            with col4:
                st.button("❌", key=f"remove_pto_{i}", on_click=remove_pto, args=(i,))
            
            if resident:  # Only add if resident is selected
                new_pto_requests.append({
                    "Resident": resident,
                    "Start_Date": start_date.strftime("%Y-%m-%d"),
                    "End_Date": end_date.strftime("%Y-%m-%d")
                })
        
        # Update pto_requests with new data
        st.session_state.pto_requests = new_pto_requests

        st.button("Add Another PTO Request", on_click=add_pto)

with tabs[3]:
    st.subheader("Soft Constraints")
    st.info("These are low-priority preferences that will be satisfied only if they don't impact schedule fairness or core rules.")
    
    def add_soft_constraint():
        st.session_state.soft_constraint_count += 1

    def remove_soft_constraint(index):
        st.session_state.removed_soft_constraints.add(index)
        st.rerun()

    # Add disable soft constraints checkbox
    st.session_state.disable_soft_constraints = st.checkbox("Disable Soft Constraints", value=st.session_state.disable_soft_constraints)

    if not st.session_state.disable_soft_constraints:
        # Initialize soft_constraints if not already initialized
        if 'soft_constraints' not in st.session_state:
            st.session_state.soft_constraints = []
        
        # Temporary list to collect new soft constraints
        new_soft_constraints = []
        
        for i in range(st.session_state.soft_constraint_count):
            if i in st.session_state.removed_soft_constraints:
                continue
            
            st.markdown(f"### Constraint #{i+1}")
            col1, col2, col3, col4 = st.columns([2, 1, 1, 0.5])
            with col1:
                default_res = st.session_state.soft_constraints[i]['Resident'] if i < len(st.session_state.soft_constraints) else (st.session_state.residents_data[0]['Name'] if st.session_state.residents_data else "")
                resident = st.selectbox(
                    "Resident",
                    [res["Name"] for res in st.session_state.residents_data],
                    index=[res["Name"] for res in st.session_state.residents_data].index(default_res) if default_res in [res["Name"] for res in st.session_state.residents_data] else 0,
                    key=f"soft_res_{i}"
                )
            with col2:
                if i < len(st.session_state.soft_constraints):
                    default_start = datetime.strptime(st.session_state.soft_constraints[i]['Start_Date'], "%Y-%m-%d")
                else:
                    default_start = datetime(start_year, 7, 1)
                start_date = st.date_input(
                    "Start Date",
                    value=default_start,
                    min_value=datetime(start_year, 7, 1).date(),
                    max_value=datetime(start_year+1, 6, 30).date(),
                    key=f"soft_start_{i}"
                )
            with col3:
                if i < len(st.session_state.soft_constraints):
                    default_end = datetime.strptime(st.session_state.soft_constraints[i]['End_Date'], "%Y-%m-%d")
                else:
                    default_end = default_start
                end_date = st.date_input(
                    "End Date",
                    value=default_end,
                    min_value=datetime(start_year, 7, 1).date(),
                    max_value=datetime(start_year+1, 6, 30).date(),
                    key=f"soft_end_{i}"
                )
            with col4:
                st.button("❌", key=f"remove_soft_{i}", on_click=remove_soft_constraint, args=(i,))
            
            if resident:  # Only add if resident is selected
                new_soft_constraints.append({
                    "Resident": resident,
                    "Start_Date": start_date.strftime("%Y-%m-%d"),
                    "End_Date": end_date.strftime("%Y-%m-%d")
                })
        
        # Update soft_constraints with new data
        st.session_state.soft_constraints = new_soft_constraints

        st.button("Add Another Constraint", on_click=add_soft_constraint)

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
        if 'previous_assignments' not in st.session_state:
            st.session_state.previous_assignments = []
        
        # Create a list to store new assignments
        new_previous_assignments = []
        
        # Create input fields for each date
        for idx, date in enumerate(dates):
            date_str = date.strftime("%Y-%m-%d")
            st.markdown(f"**{date_str}**")
            col1, col2 = st.columns(2)
            with col1:
                default_call = st.session_state.previous_assignments[idx]['Call'] if idx < len(st.session_state.previous_assignments) else ""
                call = st.selectbox(
                    "Call Resident",
                    [""] + [res["Name"] for res in st.session_state.residents_data],
                    index=0 if default_call == "" else [res["Name"] for res in st.session_state.residents_data].index(default_call) + 1 if default_call in [res["Name"] for res in st.session_state.residents_data] else 0,
                    key=f"prev_call_{date_str}"
                )
            with col2:
                default_backup = st.session_state.previous_assignments[idx]['Backup'] if idx < len(st.session_state.previous_assignments) else ""
                backup = st.selectbox(
                    "Backup Resident",
                    [""] + [res["Name"] for res in st.session_state.residents_data],
                    index=0 if default_backup == "" else [res["Name"] for res in st.session_state.residents_data].index(default_backup) + 1 if default_backup in [res["Name"] for res in st.session_state.residents_data] else 0,
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
        st.session_state.previous_assignments = new_previous_assignments

# Generate button
if st.button("Generate Schedule"):
    # Clear previous results
    st.session_state.pop('last_schedule_df', None)
    st.session_state.pop('last_stats', None)
    st.session_state.pop('last_excel_file', None)
    st.session_state.pop('last_block_choice', None)
    st.session_state.pop('last_block_name', None)
    st.session_state.pop('last_success', None)

    if not st.session_state.residents_data:
        st.error("Please enter at least one resident.")
    elif block_info[block_choice]['requires_previous'] and len(st.session_state.previous_assignments) < 4:
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
                    'Transition PGY': res['PGY'] + 1 if res['Transition_Date'] else None
                } for res in st.session_state.residents_data])
                
                # Validate transition dates are within the block
                for _, row in residents_df.iterrows():
                    if pd.notna(row['Transition Date']):
                        trans_date = datetime.strptime(row['Transition Date'], "%Y-%m-%d")
                        if not (block_start_dt <= trans_date <= block_end_dt):
                            st.warning(f"Transition date for {row['Resident']} ({row['Transition Date']}) is outside the selected block period.")
                
                # Create previous assignments DataFrame if needed
                prev_df = None
                if block_info[block_choice]['requires_previous'] and st.session_state.previous_assignments:
                    prev_df = pd.DataFrame(st.session_state.previous_assignments)
                
                # Transform holiday assignments into the format expected by the engine
                holidays_df = pd.DataFrame([{
                    'Date': holiday['Date'],
                    'Call': holiday['Call'],
                    'Backup': holiday['Backup']
                } for holiday in st.session_state.holiday_assignments_list]) if not st.session_state.disable_holidays and st.session_state.holiday_assignments_list else pd.DataFrame(columns=['Date', 'Call', 'Backup'])
                
                # Transform PTO requests into the format expected by the engine
                pto_df = pd.DataFrame([{
                    'Resident': pto['Resident'],
                    'Start Date': pto['Start_Date'],
                    'End Date': pto['End_Date']
                } for pto in st.session_state.pto_requests]) if not st.session_state.disable_pto and st.session_state.pto_requests else pd.DataFrame(columns=['Resident', 'Start Date', 'End Date'])
                
                # Run the scheduling engine with selected dates and pgy4_cap
                schedule_df = run_scheduling_engine(
                    prev_df,
                    residents_df,
                    pto_df,
                    holidays_df,
                    block_start_dt,
                    block_end_dt,
                    pgy4_cap=pgy4_cap
                )
                
                # Calculate call statistics
                # st.subheader("Call Distribution Statistics")
                
                # Create DataFrames for each PGY level
                pgy_stats = {1: [], 2: [], 3: [], 4: []}
                
                for _, resident_row in residents_df.iterrows():
                    resident = resident_row['Resident']
                    pgy = resident_row['PGY']
                    # Use Transition PGY if available for display purposes
                    display_pgy = resident_row['Transition PGY'] if pd.notna(resident_row['Transition PGY']) else pgy
                    
                    # For PGY-1, count intern assignments
                    if display_pgy == 1:
                        intern_calls = schedule_df[schedule_df['Intern'] == resident]
                        weekday_calls = len(intern_calls[intern_calls['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() not in [5, 6])])
                        saturday_calls = len(intern_calls[intern_calls['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() == 5)])
                        total_calls = len(intern_calls)
                        
                        stats = {
                            'Resident': resident,
                            'Weekday': weekday_calls,
                            'Saturday': saturday_calls,
                            'Total': total_calls
                        }
                        pgy_stats[display_pgy].append(stats)
                    else:
                        # For other PGY levels, count primary call assignments
                        resident_calls = schedule_df[schedule_df['Call'] == resident]
                        
                        # Calculate day type breakdowns
                        weekday_calls = len(resident_calls[resident_calls['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() in [0,1,2,3])])
                        friday_calls = len(resident_calls[resident_calls['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() == 4)])
                        saturday_calls = len(resident_calls[resident_calls['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() == 5)])
                        sunday_calls = len(resident_calls[resident_calls['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() == 6)])
                        total_calls = len(resident_calls)
                        
                        stats = {
                            'Resident': resident
                        }
                        
                        # Add appropriate columns based on display PGY level
                        if display_pgy == 2:
                            stats.update({
                                'Weekday': weekday_calls,
                                'Fridays': friday_calls,
                                'Sunday': sunday_calls
                            })
                        elif display_pgy == 3:
                            stats.update({
                                'Weekday': weekday_calls,
                                'Saturday': saturday_calls
                            })
                        elif display_pgy == 4:
                            stats.update({
                                'Total': total_calls
                            })
                        
                        pgy_stats[display_pgy].append(stats)
                
                # Display PGY-1 stats
                # st.markdown("### PGY-1")
                # if pgy_stats[1]:
                #     pgy1_df = pd.DataFrame(pgy_stats[1])
                #     st.dataframe(pgy1_df.set_index('Resident'), use_container_width=True)
                # else:
                #     st.info("No PGY-1 residents")
                
                # Display PGY-2 stats
                # st.markdown("### PGY-2")
                # if pgy_stats[2]:
                #     pgy2_df = pd.DataFrame(pgy_stats[2])
                #     st.dataframe(pgy2_df.set_index('Resident'), use_container_width=True)
                # else:
                #     st.info("No PGY-2 residents")
                
                # Display PGY-3 stats
                # st.markdown("### PGY-3")
                # if pgy_stats[3]:
                #     pgy3_df = pd.DataFrame(pgy_stats[3])
                #     st.dataframe(pgy3_df.set_index('Resident'), use_container_width=True)
                # else:
                #     st.info("No PGY-3 residents")
                
                # Display PGY-4 stats
                # st.markdown("### PGY-4")
                # if pgy_stats[4]:
                #     pgy4_df = pd.DataFrame(pgy_stats[4])
                #     st.dataframe(pgy4_df.set_index('Resident'), use_container_width=True)
                # else:
                #     st.info("No PGY-4 residents")
                
                # Format the schedule
                wb = format_schedule(schedule_df, schedule_df, schedule_df)  # Using same schedule for all blocks
                
                # Save to BytesIO
                excel_file = BytesIO()
                wb.save(excel_file)
                excel_file.seek(0)
                
                # Store in session state
                st.session_state['last_schedule_df'] = schedule_df
                st.session_state['last_stats'] = pgy_stats
                st.session_state['last_excel_file'] = excel_file.getvalue()
                st.session_state['last_block_choice'] = block_choice
                st.session_state['last_block_name'] = block_choice.lower().replace(' ', '_')
                st.session_state['last_success'] = True
            except Exception as e:
                st.session_state['last_success'] = False
                st.error(f"An error occurred: {str(e)}")

# --- Always display if present in session state ---
if st.session_state.get('last_success'):
    schedule_df = st.session_state.get('last_schedule_df')
    pgy_stats = st.session_state.get('last_stats')
    excel_file_bytes = st.session_state.get('last_excel_file')
    block_choice = st.session_state.get('last_block_choice', block_choice)
    block_name = st.session_state.get('last_block_name', block_choice.lower().replace(' ', '_'))

    st.subheader("Call Distribution Statistics")
    # Display PGY-1 stats
    st.markdown("### PGY-1")
    if pgy_stats[1]:
        pgy1_df = pd.DataFrame(pgy_stats[1])
        if 'Total' in pgy1_df.columns:
            pgy1_df = pgy1_df.drop(columns=['Total'])
        st.dataframe(pgy1_df.set_index('Resident'), use_container_width=True)
    else:
        st.info("No PGY-1 residents")
    # Display PGY-2 stats
    st.markdown("### PGY-2")
    if pgy_stats[2]:
        pgy2_df = pd.DataFrame(pgy_stats[2])
        st.dataframe(pgy2_df.set_index('Resident'), use_container_width=True)
    else:
        st.info("No PGY-2 residents")
    # Display PGY-3 stats
    st.markdown("### PGY-3")
    if pgy_stats[3]:
        pgy3_df = pd.DataFrame(pgy_stats[3])
        st.dataframe(pgy3_df.set_index('Resident'), use_container_width=True)
    else:
        st.info("No PGY-3 residents")
    # Display PGY-4 stats
    st.markdown("### PGY-4")
    if pgy_stats[4]:
        pgy4_df = pd.DataFrame(pgy_stats[4])
        st.dataframe(pgy4_df.set_index('Resident'), use_container_width=True)
    else:
        st.info("No PGY-4 residents")
    # Download button
    st.download_button(
        label="Download Schedule",
        data=excel_file_bytes,
        file_name=f"call_schedule_{block_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.success("Schedule generated successfully!") 