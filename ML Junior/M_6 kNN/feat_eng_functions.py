def income_per_dependant(row):
    return row.personal_income / (row.dependants + 1)

def loan_closed_ratio(row):
    return row.loan_num_closed / row.loan_num_total

def financial_burden(row):
    return row.loan_num_total / (row.personal_income + 1)

def total_responsibility(row):
    return row.child_total + row.dependants

def work_status(row):
    return row.socstatus_work_fl + 2 * row.socstatus_pens_fl

def age_gender_group(row):
    return row.gender * 10 + (row.age // 10)

def has_dependants_or_children(row):
    return (row.child_total > 0) and (row.dependants > 0)