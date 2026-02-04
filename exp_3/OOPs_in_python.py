import re
from collections import defaultdict

VALID_ACTIVITIES = {"LOGIN", "LOGOUT", "SUBMIT_ASSIGNMENT"}

# Student class
class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.activities = []

    def add_activity(self, activity, date, time):
        self.activities.append((activity, date, time))

    def activity_summary(self):
        logins = sum(1 for a in self.activities if a[0] == "LOGIN")
        submissions = sum(1 for a in self.activities if a[0] == "SUBMIT_ASSIGNMENT")
        return logins, submissions


# Generator to read valid log entries
def read_log_file(filename):
    with open(filename, "r") as file:
        for line in file:
            try:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) != 5:
                    raise ValueError("Invalid format")

                sid, name, activity, date, time = parts

                # Validate student ID
                if not re.match(r"S\d+", sid):
                    raise ValueError("Invalid Student ID")

                # Validate activity
                if activity not in VALID_ACTIVITIES:
                    raise ValueError("Invalid Activity")

                yield sid, name, activity, date, time

            except Exception:
                continue  # Ignore invalid entries


# Main processing
students = {}
daily_stats = defaultdict(int)
login_tracker = defaultdict(int)

for sid, name, activity, date, time in read_log_file("./exp 3/activity_log.txt"):
    if sid not in students:
        students[sid] = Student(sid, name)

    students[sid].add_activity(activity, date, time)
    daily_stats[date] += 1

    if activity == "LOGIN":
        login_tracker[sid] += 1
    elif activity == "LOGOUT":
        login_tracker[sid] -= 1


# Generate report
with open("./exp 3/activity_report.txt", "w") as report:
    print("STUDENT ACTIVITY REPORT")
    print("-" * 40)
    report.write("STUDENT ACTIVITY REPORT\n")

    for student in students.values():
        logins, submissions = student.activity_summary()
        line = f"{student.student_id} | Logins: {logins} | Submissions: {submissions}"
        print(line)
        report.write(line + "\n")

    print("\nABNORMAL BEHAVIOR (Multiple logins without logout):")
    report.write("\nABNORMAL BEHAVIOR:\n")
    for sid, count in login_tracker.items():
        if count > 0:
            msg = f"{sid} has {count} unclosed login(s)"
            print(msg)
            report.write(msg + "\n")

    print("\nDAILY ACTIVITY STATISTICS:")
    report.write("\nDAILY ACTIVITY STATISTICS:\n")
    for date, count in daily_stats.items():
        msg = f"{date} : {count} activities"
        print(msg)
        report.write(msg + "\n")
