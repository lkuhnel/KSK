# scheduling_engine.py

import pandas as pd
import random
from datetime import datetime, timedelta
import re
import argparse

# --- CallScheduler CLASS ---

class CallScheduler:
    def __init__(self, residents_info, fixed_assignments, holidays, pto_requests=None, transitions=None, pgy4_cap=None):
        self.residents_info = residents_info
        self.fixed_assignments = fixed_assignments
        self.holidays = holidays
        self.pto_requests = pto_requests if pto_requests else {}
        self.transitions = transitions if transitions else {}
        self.pgy4_cap = pgy4_cap
        
        self.call_log = {}
        self.backup_log = {}
        self.intern_log = {}  # New: Track intern assignments
        self.assignments = []

        # Update call counts to include intern-specific tracking
        self.call_counts = {r: {
            "weekday": 0, 
            "friday": 0, 
            "sunday": 0, 
            "saturday": 0, 
            "total": 0,
            "intern_weekday": 0,  # New: Track intern weekday calls
            "intern_saturday": 0  # New: Track intern Saturday calls
        } for r in self.get_all_residents()}

        for date_str, (call, backup) in self.fixed_assignments.items():
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            self.call_log.setdefault(call, []).append(date_obj)
            self.backup_log.setdefault(backup, []).append(date_obj)

    def get_all_residents(self):
        return sum(self.residents_info.values(), [])

    def get_resident_pgy(self, resident, date):
        if resident in self.transitions:
            transition_date, new_pgy = self.transitions[resident]
            if date > transition_date:  # Only return new PGY if date is strictly after transition date
                return new_pgy
        for pgy, residents in self.residents_info.items():
            if resident in residents:
                return pgy
        return None

    def is_pgy_match(self, resident, date, role="call"):
        date_str = date.strftime("%Y-%m-%d")
        if date_str in self.fixed_assignments:
            # For holidays, only the assigned residents are eligible
            call_fixed, backup_fixed = self.fixed_assignments[date_str]
            if role == "call":
                return resident == call_fixed
            else:  # backup
                return resident == backup_fixed
        
        pgy = self.get_resident_pgy(resident, date)
        if pgy is None:
            return False

        # For backup role, allow any PGY-2 or higher
        if role == "backup":
            return pgy >= 2

        # For call role, use specific day requirements
        dow = date.weekday()
        if pgy == 1:  # PGY-1 interns
            # Interns are not eligible for primary call
            return False
        elif pgy == 2:
            if dow in [1, 2, 4, 6]:  # Tuesday, Wednesday, Friday, Sunday
                return True
        elif pgy == 3:
            if dow in [0, 2, 3, 5]:  # Monday, Wednesday, Thursday, Saturday
                return True
        elif pgy == 4:
            if dow == 3:  # Thursday
                return True
        return False

    def spacing_okay(self, resident, date, role):
        for assigned_date in self.call_log.get(resident, []):
            # Only check spacing within the same block
            if abs((date - assigned_date).days) < 4 and (date.month == assigned_date.month or abs((date - assigned_date).days) < 4):
                return False
        for assigned_date in self.backup_log.get(resident, []):
            # Only check spacing within the same block
            if role == "call" and abs((date - assigned_date).days) < 4 and (date.month == assigned_date.month or abs((date - assigned_date).days) < 4):
                return False
            if role == "backup" and abs((date - assigned_date).days) < 3 and (date.month == assigned_date.month or abs((date - assigned_date).days) < 3):
                return False
        return True

    def pto_okay(self, resident, date):
        return date.strftime("%Y-%m-%d") not in self.pto_requests.get(resident, [])

    def fairness_score(self, resident, dow):
        counts = self.call_counts[resident]
        score = counts["total"]
        if dow in [0,1,2,3]:
            score += counts["weekday"] * 2
        elif dow == 4:
            score += counts["friday"] * 3
        elif dow == 5:
            score += counts["saturday"] * 2
        elif dow == 6:
            score += counts["sunday"] * 3
        return score

    def eligible_residents(self, date, role):
        candidates = []
        date_str = date.strftime("%Y-%m-%d")
        print(f"\nChecking eligibility for {date_str} ({role}):")
        for r in self.get_all_residents():
            if not self.is_pgy_match(r, date, role):
                print(f"  {r}: Failed PGY match")
                continue
            if not self.spacing_okay(r, date, role):
                print(f"  {r}: Failed spacing")
                continue
            if not self.pto_okay(r, date):
                print(f"  {r}: On PTO")
                continue
            # Enforce PGY-4 cap for call role
            if role == "call" and self.pgy4_cap is not None:
                pgy = self.get_resident_pgy(r, date)
                if pgy == 4 and self.call_counts[r]["total"] >= self.pgy4_cap:
                    print(f"  {r}: Reached PGY-4 cap ({self.pgy4_cap})")
                    continue
            candidates.append(r)
            print(f"  {r}: ELIGIBLE")
        return candidates

    def is_intern_eligible(self, intern, date, call_resident):
        """Check if an intern can be assigned for a given date and call resident"""
        date_str = date.strftime("%Y-%m-%d")
        
        # Interns can't be assigned on holidays
        if date_str in self.fixed_assignments:
            return False
            
        # Interns can only be assigned when PGY-3 or PGY-4 is on call
        call_pgy = self.get_resident_pgy(call_resident, date)
        if call_pgy not in [3, 4]:
            return False
            
        # Check PTO
        if not self.pto_okay(intern, date):
            return False
            
        # No spacing requirements for interns
        return True

    def assign_day(self, date):
        date_str = date.strftime("%Y-%m-%d")
        dow = date.weekday()

        if date_str in self.fixed_assignments:
            call_fixed, backup_fixed = self.fixed_assignments[date_str]
            self.assignments.append((date_str, call_fixed, backup_fixed, None))  # Add None for intern
            self.update_counters(call_fixed, backup_fixed, dow)
            return

        # First assign the main call resident
        call_candidates = self.eligible_residents(date, "call")
        if not call_candidates:
            raise Exception(f"No eligible CALL resident for {date_str}")

        call_resident = min(call_candidates, key=lambda r: self.fairness_score(r, dow))
        call_pgy = self.get_resident_pgy(call_resident, date)
        print(f"\nAssigned {call_resident} (PGY-{call_pgy}) as call resident for {date_str}")

        # Get backup candidates of the same PGY level
        backup_candidates = []
        for r in self.get_all_residents():
            if r == call_resident:
                continue
            if not self.spacing_okay(r, date, "backup"):
                continue
            if not self.pto_okay(r, date):
                continue
            backup_pgy = self.get_resident_pgy(r, date)
            if backup_pgy == call_pgy:  # Must be same PGY level
                backup_candidates.append(r)

        if not backup_candidates:
            raise Exception(f"No eligible BACKUP resident of PGY-{call_pgy} for {date_str} after selecting {call_resident} for call")

        backup_resident = min(backup_candidates, key=lambda r: self.fairness_score(r, dow))
        print(f"Assigned {backup_resident} as backup resident")

        # If PGY-3 or PGY-4 is on call, try to assign an intern
        intern_assigned = None
        if call_pgy in [3, 4]:
            print(f"\nChecking intern assignment for {date_str} (PGY-{call_pgy} on call)")
            # Get all PGY-1 residents
            intern_candidates = self.residents_info.get(1, [])
            print(f"Available interns: {intern_candidates}")
            if intern_candidates:
                # Filter eligible interns
                eligible_interns = [r for r in intern_candidates if self.is_intern_eligible(r, date, call_resident)]
                print(f"Eligible interns: {eligible_interns}")
                if eligible_interns:
                    # Choose intern based on the type of day
                    if dow == 5:  # Saturday
                        # Sort by Saturday calls first, then use weekday calls as tiebreaker
                        intern_assigned = min(eligible_interns, 
                            key=lambda r: (
                                self.call_counts[r]["intern_saturday"],
                                self.call_counts[r]["intern_weekday"]
                            )
                        )
                        print(f"Saturday assignment - Intern call counts:")
                        for intern in eligible_interns:
                            print(f"  {intern}: {self.call_counts[intern]['intern_saturday']} Saturdays, {self.call_counts[intern]['intern_weekday']} Weekdays")
                    else:  # Weekday
                        # Sort by weekday calls first, then use Saturday calls as tiebreaker
                        intern_assigned = min(eligible_interns, 
                            key=lambda r: (
                                self.call_counts[r]["intern_weekday"],
                                self.call_counts[r]["intern_saturday"]
                            )
                        )
                        print(f"Weekday assignment - Intern call counts:")
                        for intern in eligible_interns:
                            print(f"  {intern}: {self.call_counts[intern]['intern_weekday']} Weekdays, {self.call_counts[intern]['intern_saturday']} Saturdays")
                    
                    print(f"Assigned {intern_assigned} as intern")
                    self.intern_log.setdefault(intern_assigned, []).append(date)
                    # Update intern call counts
                    if dow == 5:  # Saturday
                        self.call_counts[intern_assigned]["intern_saturday"] += 1
                    else:
                        self.call_counts[intern_assigned]["intern_weekday"] += 1
                else:
                    print("No eligible interns found")
            else:
                print("No PGY-1 residents available")

        self.assignments.append((date_str, call_resident, backup_resident, intern_assigned))
        self.call_log.setdefault(call_resident, []).append(date)
        self.backup_log.setdefault(backup_resident, []).append(date)
        self.update_counters(call_resident, backup_resident, dow)

    def update_counters(self, call, backup, dow):
        self.call_counts[call]["total"] += 1
        if dow == 4:
            self.call_counts[call]["friday"] += 1
        elif dow == 5:
            self.call_counts[call]["saturday"] += 1
        elif dow == 6:
            self.call_counts[call]["sunday"] += 1
        else:
            self.call_counts[call]["weekday"] += 1

    def schedule_range(self, start_date, end_date):
        current = start_date
        while current <= end_date:
            self.assign_day(current)
            current += timedelta(days=1)

    def export_schedule(self):
        # Convert assignments to DataFrame with intern column
        df = pd.DataFrame(self.assignments, columns=["Date", "Call", "Backup", "Intern"])
        return df

# --- Wrapper Function to Connect to App ---

def run_scheduling_engine(prev_df, res_df, pto_df, hol_df, start_date=None, end_date=None, pgy4_cap=None):
    residents_info = {1: [], 2: [], 3: [], 4: []}  # Added PGY-1
    transitions = {}

    # Default dates for Block 1
    if start_date is None:
        start_date = datetime(2025, 7, 1)
    if end_date is None:
        end_date = datetime(2025, 10, 31)

    print("\nProcessing residents:")
    for _, row in res_df.iterrows():
        name = row["Resident"]
        pgy = int(row["PGY"])
        residents_info[pgy].append(name)
        print(f"Added {name} as PGY-{pgy}")
        
        # Handle transitions
        if pd.notna(row["Transition Date"]):
            trans_date = datetime.strptime(row["Transition Date"], "%Y-%m-%d")
            transitions[name] = (trans_date, int(row["Transition PGY"]))
            print(f"  Transition: {name} will become PGY-{row['Transition PGY']} on {row['Transition Date']}")

    print("\nResidents by PGY level:")
    for pgy, residents in residents_info.items():
        print(f"PGY-{pgy}: {residents}")

    # Process PTO requests
    pto_requests = {}
    if not pto_df.empty:
        for _, row in pto_df.iterrows():
            resident = row["Resident"]
            start = datetime.strptime(row["Start Date"], "%Y-%m-%d")
            end = datetime.strptime(row["End Date"], "%Y-%m-%d")
            current = start
            while current <= end:
                pto_requests.setdefault(resident, []).append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

    # Process holiday assignments
    fixed_assignments = {}
    if not hol_df.empty:
        print("\nProcessing holiday assignments:")
        for _, row in hol_df.iterrows():
            date_str = row["Date"]
            call = row["Call"]
            backup = row["Backup"]
            fixed_assignments[date_str] = (call, backup)
            print(f"Added holiday assignment for {date_str}: Call={call}, Backup={backup}")

    # Process previous block assignments
    if prev_df is not None and not prev_df.empty:
        for _, row in prev_df.iterrows():
            date_str = row["Date"]
            call = row["Call"]
            backup = row["Backup"]
            fixed_assignments[date_str] = (call, backup)

    # Create scheduler instance
    scheduler = CallScheduler(residents_info, fixed_assignments, hol_df, pto_requests, transitions, pgy4_cap=pgy4_cap)
    
    # Generate schedule
    scheduler.schedule_range(start_date, end_date)
    
    # Export schedule and add supervisor assignment
    df = scheduler.export_schedule()
    df["Supervisor"] = None

    # Build a lookup for call assignments by date
    call_by_date = {row["Date"]: row["Call"] for _, row in df.iterrows()}
    pgy_by_name = {}
    for _, row in res_df.iterrows():
        name = row["Resident"]
        pgy = int(row["PGY"])
        pgy_by_name[name] = pgy
        if pd.notna(row["Transition Date"]):
            trans_date = datetime.strptime(row["Transition Date"], "%Y-%m-%d")
            pgy_by_name[(name, trans_date.strftime("%Y-%m-%d"))] = int(row["Transition PGY"])

    # Helper to get PGY for a resident on a given date
    def get_pgy(resident, date):
        # Check for transition
        for _, row in res_df.iterrows():
            if row["Resident"] == resident and pd.notna(row["Transition Date"]):
                trans_date = datetime.strptime(row["Transition Date"], "%Y-%m-%d")
                if date > trans_date:
                    return int(row["Transition PGY"])
        return pgy_by_name.get(resident, None)

    # Supervisor assignment tracking
    supervisor_counts = {r: 0 for r in scheduler.get_all_residents() if get_pgy(r, start_date) in [3, 4]}
    last_call_by_resident = {}

    for idx, row in df.iterrows():
        date = datetime.strptime(row["Date"], "%Y-%m-%d")
        call_resident = row["Call"]
        # Skip holidays (already assigned in fixed_assignments)
        if row["Date"] in fixed_assignments:
            continue
        # Only assign supervisor if call resident is PGY-2 on this date
        if get_pgy(call_resident, date) != 2:
            continue
        # Build eligible supervisor list
        eligible_supervisors = []
        for r in supervisor_counts:
            # Not on call the previous day
            prev_date = (date - timedelta(days=1)).strftime("%Y-%m-%d")
            if call_by_date.get(prev_date) == r:
                continue
            # Must be PGY-3 or PGY-4 on this date
            if get_pgy(r, date) not in [3, 4]:
                continue
            # Check PTO
            if date.strftime("%Y-%m-%d") in pto_requests.get(r, []):
                continue
            eligible_supervisors.append(r)
        # Friday rule: if Friday, try to assign Saturday call resident as supervisor
        if date.weekday() == 4:  # Friday
            sat_date = (date + timedelta(days=1)).strftime("%Y-%m-%d")
            sat_call = call_by_date.get(sat_date)
            if sat_call in eligible_supervisors:
                df.at[idx, "Supervisor"] = sat_call
                supervisor_counts[sat_call] += 1
                continue
        # Otherwise, pick eligible supervisor with fewest assignments
        if eligible_supervisors:
            chosen = min(eligible_supervisors, key=lambda r: supervisor_counts[r])
            df.at[idx, "Supervisor"] = chosen
            supervisor_counts[chosen] += 1
        else:
            # If no one is eligible, leave blank (or could relax rule/log warning)
            df.at[idx, "Supervisor"] = None
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate call schedule')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--previous_schedule', type=str, help='Path to previous block schedule CSV')
    parser.add_argument('--output_file', type=str, help='Path to save the generated schedule')
    args = parser.parse_args()

    # Read input files
    res_df = pd.read_csv('resident_list_structured.csv')
    pto_df = pd.read_csv('pto_requests.csv')
    hol_df = pd.read_csv('holiday_schedule.csv')
    
    # Read previous schedule if provided
    prev_df = None
    if args.previous_schedule:
        prev_df = pd.read_csv(args.previous_schedule)

    # Convert dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Generate schedule
    schedule_df = run_scheduling_engine(prev_df, res_df, pto_df, hol_df, start_date, end_date)
    
    # Save the schedule
    output_file = args.output_file if args.output_file else 'generated_schedule.csv'
    schedule_df.to_csv(output_file, index=False)
    print(f"Schedule generated from {args.start_date} to {args.end_date}")
    if args.previous_schedule:
        print(f"Using previous schedule from: {args.previous_schedule}")
    print(f"Schedule saved to: {output_file}")
