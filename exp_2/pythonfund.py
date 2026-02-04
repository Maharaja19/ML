# Tuple to store student IDs
student_ids = ('S101', 'S102', 'S103', 'S104')

# Dictionary to store student academic details
students = {
    'S101': {'name': 'Asha', 'assignment': 78, 'test': 80, 'attendance': 92, 'hours': 8},
    'S102': {'name': 'Ravi', 'assignment': 65, 'test': 68, 'attendance': 85, 'hours': 5},
    'S103': {'name': 'Meena', 'assignment': 88, 'test': 90, 'attendance': 96, 'hours': 10},
    'S104': {'name': 'Kiran', 'assignment': 55, 'test': 58, 'attendance': 78, 'hours': 4}
}

# Function to calculate average score
def calculate_average(assignment, test):
    return (assignment + test) / 2

# Function to determine academic risk level
def determine_risk(avg_score, attendance, hours):
    if avg_score < 60 or attendance < 80 or hours < 5:
        return "High Risk"
    elif avg_score < 75:
        return "Moderate Risk"
    else:
        return "Low Risk"

# Create a list to store processed student records
report = []

for sid in student_ids:
    data = students[sid]
    avg = calculate_average(data['assignment'], data['test'])
    risk = determine_risk(avg, data['attendance'], data['hours'])
    
    report.append({
        'id': sid,
        'name': data['name'],
        'average': avg,
        'attendance': data['attendance'],
        'hours': data['hours'],
        'risk': risk
    })

# Sort the report by average score in decreasing order
report.sort(key=lambda x: x['average'], reverse=True)

# Display structured performance report
print("STUDENT PERFORMANCE REPORT (Sorted by Average Score)")
print("-" * 55)

for student in report:
    print(f"Student ID      : {student['id']}")
    print(f"Name            : {student['name']}")
    print(f"Average Score   : {student['average']:.2f}")
    print(f"Attendance (%)  : {student['attendance']}")
    print(f"Study Hours     : {student['hours']} hrs/week")
    print(f"Risk Level      : {student['risk']}")
    print("-" * 55)
