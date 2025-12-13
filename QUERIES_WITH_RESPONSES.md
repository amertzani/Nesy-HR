# All Tested Queries with System Responses and Extracted Facts

**Total Queries**: 46  
**Source**: Offline Evaluation Report (Generated: 2025-12-13 20:47:41)  
**Dataset**: HRDataset_v14.csv (311 rows, 36 columns)

---

## Query 1
**Query**: What is the distribution of performance scores by department?

**Response** (4.99s):
```
Average performance by department:
• IT/IS: 3.20
• Production: 3.14
• Admin Offices: 3.00
• Software Engineering: 3.00
```

**Evidence**: No facts retrieved  
**Result**: ✗ Incorrect

---

## Query 2
**Query**: How do performance scores vary across departments?

**Response** (8.62s):
```
Average performance by department:
• IT/IS: 3.20
• Production: 3.14
• Admin Offices: 3.00
• Software Engineering: 3.00
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 3
**Query**: Which department has the highest average performance score?

**Response** (9.94s):
```
IT/IS has the highest average performance of 3.20
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 4
**Query**: Show me performance metrics by department

**Response** (14.15s):
```
Average performance by department:
• IT/IS: 3.20
• Production: 3.14
• Admin Offices: 3.00
• Software Engineering: 3.00
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 5
**Query**: What is the average special projects count by department?

**Response** (10.28s):
```
Average special projects count by department:
• IT/IS: 5.80
• Admin Offices: 4.50
• Software Engineering: 4.00
• Production: 0.00
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 6
**Query**: How do special projects vary across departments?

**Response** (12.04s):
```
Average special projects count by department:
• IT/IS: 5.80
• Admin Offices: 4.50
• Software Engineering: 4.00
• Production: 0.00
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 7
**Query**: Which department has the highest average special projects count?

**Response** (11.29s):
```
IT/IS has the highest average special projects count of 5.80
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 8
**Query**: Show me special projects distribution by department

**Response** (12.41s):
```
Average special projects count by department:
• IT/IS: 5.80
• Admin Offices: 4.50
• Software Engineering: 4.00
• Production: 0.00
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 9
**Query**: What is the team-level engagement by manager?

**Response** (16.55s):
```
Average engagement by manager:
• Simon Roup: 4.49
• Amy Dunn: 4.43
• Michael Albert: 4.29
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 10
**Query**: How does engagement vary by manager?

**Response** (13.43s):
```
Average engagement by manager:
• Simon Roup: 4.49
• Amy Dunn: 4.43
• Michael Albert: 4.29
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 11
**Query**: Which manager has the highest team engagement?

**Response** (13.42s):
```
Simon Roup has the highest average engagement of 4.49
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 12
**Query**: What is the average salary by department?

**Response** (14.51s):
```
Average salary by department:
• Software Engineering: $95,660.00
• IT/IS: $92,111.20
• Admin Offices: $84,683.50
• Production: $58,741.64
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 13
**Query**: How does salary distribution vary across departments?

**Response** (19.33s):
```
Average salary by department:
• Software Engineering: $95,660.00
• IT/IS: $92,111.20
• Admin Offices: $84,683.50
• Production: $58,741.64
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 14
**Query**: Which department has the highest average salary?

**Response** (14.89s):
```
Software Engineering has the highest average salary of $95,660.00
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 15
**Query**: How does performance vary by recruitment source?

**Response** (16.08s):
```
Distribution of performance by recruitment:
• Employee Referral: 3.50
• LinkedIn: 3.33
• Diversity Job Fair: 3.17
• Indeed: 3.17
• Google Search: 3.00
• On-line Web application: 3.00
• CareerBuilder: 2.00
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 16
**Query**: Which recruitment sources have the employees with highest performance score?

**Response** (22.56s):
```
Employee Referral has the highest average performance of 3.50
```

**Evidence**: No facts retrieved  
**Result**: ✗ Incorrect

---

## Query 17
**Query**: What is the performance distribution by recruitment source?

**Response** (17.71s):
```
Distribution of performance by recruitment:
• Employee Referral: 3.50
• LinkedIn: 3.33
• Diversity Job Fair: 3.17
• Indeed: 3.17
• Google Search: 3.00
• On-line Web application: 3.00
• CareerBuilder: 2.00
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 18
**Query**: Identify employees with high performance, low engagement and many special projects

**Response** (0.01s):
```
Found 2 employee(s) matching the criteria:
1. Andreola, Colby
2. Booth, Frank
```

**Evidence**: No facts retrieved  
**Result**: ✗ Incorrect

---

## Query 19
**Query**: Find employees with high performance, low engagement and low satisfaction

**Response** (0.01s):
```
Found 3 employee(s) matching the criteria:
1. Akinkuolie, Sarah
2. Andreola, Colby
3. Bozzi, Charles
```

**Evidence**: No facts retrieved  
**Result**: ✗ Incorrect

---

## Query 20
**Query**: Find employees with high performance, low engagement and low satisfaction and many special projects

**Response** (0.01s):
```
Found 1 employee(s) matching the criteria:
1. Andreola, Colby
```

**Evidence**: No facts retrieved  
**Result**: ✗ Incorrect

---

## Query 21
**Query**: Find employees with low engagement and low satisfaction and many special projects and many absences

**Response** (0.01s):
```
Found 1 employee(s) matching the criteria:
1. Andreola, Colby
```

**Evidence**: No facts retrieved  
**Result**: ✗ Incorrect

---

## Query 22
**Query**: Which departments have high salaries but low performance?

**Response** (17.83s):
```
No departments found with high salaries but low performance.
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 23
**Query**: Analyze the relationship between salary, performance, and department

**Response** (22.52s):
```
Department Analysis (Salary vs Performance):
• IT/IS: Salary $92,111.20, Performance 3.20
• Production: Salary $58,741.64, Performance 3.14
• Admin Offices: Salary $84,683.50, Performance 3.00
• Software Engineering: Salary $95,660.00, Performance 3.00
```

**Evidence**: No facts retrieved  
**Result**: ✗ Incorrect: Only 0/17 entities matched (need 3+ or 50%+)

---

## Query 24
**Query**: Identify departments with low salary and high performance

**Response** (19.89s):
```
Departments with low salary but high performance:
• Production: Salary $58,741.64, Performance 3.14
```

**Evidence**: No facts retrieved  
**Result**: ✓ Correct (matches ground truth)

---

## Query 25
**Query**: Retrieve facts related with employee Becker, Scott

**Response** (1.20s):
```
No facts found for employee Becker, Scott
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. lowest engagement → is → Becker, Scott
3. Bottom 2 employee → is → Andreola, Colby
4. Employee Becker, Scott → has → engagement score of 4
5. Recruitment source Employee Referral → has → average salary of 64748
6. Bottom 1 employee → is → Blount, Dianna
7. employees by absences → is → 1. Andreola, Colby
8. Employees from recruitment source Employee Referral → have → average salary of 64748
9. absences and → is → ranked 2 in top employees
10. employee → relates to → Athwal, Sam
11. Employee Biden, Lowan  M → has → engagement score of 4
12. employee → is → Biden, Lowan  M
13. Department IT/IS → has → 5 employees
14. Employee Referral → type → entity
15. employee → is → Becker, Scott

**Result**: ✓ Correct (matches ground truth)

---

## Query 26
**Query**: Show me all facts about employee Becker, Scott

**Response** (1.20s):
```
No facts found for employee Becker, Scott
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. lowest engagement → is → Becker, Scott
3. Bottom 2 employee → is → Andreola, Colby
4. Employee Becker, Scott → has → engagement score of 4
5. Recruitment source Employee Referral → has → average salary of 64748
6. Bottom 1 employee → is → Blount, Dianna
7. employees by absences → is → 1. Andreola, Colby
8. Employees from recruitment source Employee Referral → have → average salary of 64748
9. absences and → is → ranked 2 in top employees
10. employee → relates to → Athwal, Sam
11. Employee Biden, Lowan  M → has → engagement score of 4
12. employee → is → Biden, Lowan  M
13. Department IT/IS → has → 5 employees
14. Employee Referral → type → entity
15. employee → is → Becker, Scott

**Result**: ✓ Correct (matches ground truth)

---

## Query 27
**Query**: What information do we have about employee Becker, Scott?

**Response** (0.00s):
```
Could not parse query. Detected: metric=None, group_by=None
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. lowest engagement → is → Becker, Scott
3. Bottom 2 employee → is → Andreola, Colby
4. Employee Becker, Scott → has → engagement score of 4
5. Recruitment source Employee Referral → has → average salary of 64748
6. Bottom 1 employee → is → Blount, Dianna
7. employees by absences → is → 1. Andreola, Colby
8. Employees from recruitment source Employee Referral → have → average salary of 64748
9. absences and → is → ranked 2 in top employees
10. employee → relates to → Athwal, Sam
11. Employee Biden, Lowan  M → has → engagement score of 4
12. employee → is → Biden, Lowan  M
13. Department IT/IS → has → 5 employees
14. Employee Referral → type → entity
15. employee → is → Becker, Scott

**Result**: ✓ Correct (matches ground truth)

---

## Query 28
**Query**: Give me facts about the employee with the highest salary

**Response** (0.01s):
```
Employee with highest salary: Boutwell, Bonalyn ($106,367.00), but no additional facts found in knowledge graph.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. Bottom 2 employee → is → Andreola, Colby
3. Employee Becker, Scott → has → engagement score of 4
4. Recruitment source Employee Referral → has → average salary of 64748
5. Bottom 1 employee → is → Blount, Dianna
6. employees by absences → is → 1. Andreola, Colby
7. Employees from recruitment source Employee Referral → have → average salary of 64748
8. absences and → is → ranked 2 in top employees
9. employee → relates to → Athwal, Sam
10. Employee Biden, Lowan  M → has → engagement score of 4
11. employee → is → Biden, Lowan  M
12. Department IT/IS → has → 5 employees
13. Employee Referral → type → entity
14. employee → is → Becker, Scott
15. Employee Bacong, Alejandro → has → 16 absences (rank 4 highest)

**Result**: ✓ Correct (matches ground truth)

---

## Query 29
**Query**: Retrieve facts about the employee who has the highest salary

**Response** (0.01s):
```
Employee with highest salary: Boutwell, Bonalyn ($106,367.00), but no additional facts found in knowledge graph.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. Bottom 2 employee → is → Andreola, Colby
3. Employee Becker, Scott → has → engagement score of 4
4. Recruitment source Employee Referral → has → average salary of 64748
5. Bottom 1 employee → is → Blount, Dianna
6. employees by absences → is → 1. Andreola, Colby
7. Employees from recruitment source Employee Referral → have → average salary of 64748
8. absences and → is → ranked 2 in top employees
9. employee → relates to → Athwal, Sam
10. Employee Biden, Lowan  M → has → engagement score of 4
11. employee → is → Biden, Lowan  M
12. Department IT/IS → has → 5 employees
13. Employee Referral → type → entity
14. employee → is → Becker, Scott
15. Employee Bacong, Alejandro → has → 16 absences (rank 4 highest)

**Result**: ✓ Correct (matches ground truth)

---

## Query 30
**Query**: Show me information about the highest paid employee

**Response** (0.01s):
```
Employee with highest salary: Boutwell, Bonalyn ($106,367.00), but no additional facts found in knowledge graph.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. Bottom 2 employee → is → Andreola, Colby
3. Employee Becker, Scott → has → engagement score of 4
4. Recruitment source Employee Referral → has → average salary of 64748
5. Bottom 1 employee → is → Blount, Dianna
6. employees by absences → is → 1. Andreola, Colby
7. Employees from recruitment source Employee Referral → have → average salary of 64748
8. absences and → is → ranked 2 in top employees
9. employee → relates to → Athwal, Sam
10. Employee Biden, Lowan  M → has → engagement score of 4
11. employee → is → Biden, Lowan  M
12. Department IT/IS → has → 5 employees
13. Employee Referral → type → entity
14. employee → is → Becker, Scott
15. Employee Bacong, Alejandro → has → 16 absences (rank 4 highest)

**Result**: ✓ Correct (matches ground truth)

---

## Query 31
**Query**: Give me facts about the employee with the lowest performance

**Response** (4.77s):
```
Employee with lowest performance: Blount, Dianna (Performance: 2.0), but no facts found in knowledge graph.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. Bottom 2 employee → is → Andreola, Colby
3. Employee Becker, Scott → has → engagement score of 4
4. Recruitment source Employee Referral → has → average salary of 64748
5. Bottom 1 employee → is → Blount, Dianna
6. employees by absences → is → 1. Andreola, Colby
7. Employees from recruitment source Employee Referral → have → average salary of 64748
8. absences and → is → ranked 2 in top employees
9. employee → relates to → Athwal, Sam
10. Employee Biden, Lowan  M → has → engagement score of 4
11. employee → is → Biden, Lowan  M
12. Department IT/IS → has → average performance score of 3
13. Department IT/IS → has → 5 employees
14. Employee Referral → type → entity
15. employee → is → Becker, Scott

**Result**: ✓ Correct (matches ground truth)

---

## Query 32
**Query**: Retrieve facts about the employee who has the lowest performance score

**Response** (6.96s):
```
Employee with lowest performance: Blount, Dianna (Performance: 2.0), but no facts found in knowledge graph.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. Bottom 2 employee → is → Andreola, Colby
3. Employee Becker, Scott → has → engagement score of 4
4. Recruitment source Employee Referral → has → average salary of 64748
5. Bottom 1 employee → is → Blount, Dianna
6. employees by absences → is → 1. Andreola, Colby
7. Employees from recruitment source Employee Referral → have → average salary of 64748
8. absences and → is → ranked 2 in top employees
9. employee → relates to → Athwal, Sam
10. Employee Biden, Lowan  M → has → engagement score of 4
11. employee → is → Biden, Lowan  M
12. Department IT/IS → has → average performance score of 3
13. Department IT/IS → has → 5 employees
14. Employee Referral → type → entity
15. employee → is → Becker, Scott

**Result**: ✓ Correct (matches ground truth)

---

## Query 33
**Query**: Show me information about the employee with worst performance

**Response** (5.29s):
```
Employee with lowest performance: Blount, Dianna (Performance: 2.0), but no facts found in knowledge graph.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. Bottom 2 employee → is → Andreola, Colby
3. Employee Becker, Scott → has → engagement score of 4
4. Recruitment source Employee Referral → has → average salary of 64748
5. Bottom 1 employee → is → Blount, Dianna
6. employees by absences → is → 1. Andreola, Colby
7. Employees from recruitment source Employee Referral → have → average salary of 64748
8. absences and → is → ranked 2 in top employees
9. employee → relates to → Athwal, Sam
10. Employee Biden, Lowan  M → has → engagement score of 4
11. employee → is → Biden, Lowan  M
12. Department IT/IS → has → average performance score of 3
13. Department IT/IS → has → 5 employees
14. Employee Referral → type → entity
15. employee → is → Becker, Scott

**Result**: ✓ Correct (matches ground truth)

---

## Query 34
**Query**: Show me facts about employees in IT/IS department

**Response** (1.22s):
```
Facts about employees in IT/IS department (10 facts):
1. Department IT/IS has average performance score of 3
2. Department IT/IS has 5 employees
3. Barbossa, Hector from IT/IS department has engagement score of 4
4. IS department has engagement score of 4
5. IS department has 16 absences (rank 4)
...
```

**Evidence**: 25 facts retrieved

**📄 Document Agent Facts (Direct CSV):**
1. Barbossa, Hector from IT/IS department has engagement score of 4
2. Bacong, Alejandro  from IT/IS department has 16 absences (rank 4)

**📊 Operational Insights:**
3. employee → relates to → Barbossa, Hector
4. Bottom 2 employee → is → Andreola, Colby
5. Employee Becker, Scott → has → engagement score of 4
6. Recruitment source Employee Referral → has → average salary of 64748
7. Bottom 1 employee → is → Blount, Dianna
8. employees by absences → is → 1. Andreola, Colby
9. Employees from recruitment source Employee Referral → have → average salary of 64748
10. absences and → is → ranked 2 in top employees
11. employee → relates to → Athwal, Sam
12. Employee Biden, Lowan  M → has → engagement score of 4
13. employee → is → Biden, Lowan  M
14. Department IT/IS → has → average performance score of 3
15. Department IT/IS → has → 5 employees
16. Employee Referral → type → entity
17. employee → is → Becker, Scott
18. Department IT/IS has average performance score of 3
19. Department IT/IS has 5 employees
20. IS department has engagement score of 4
21. IS department has 16 absences (rank 4)
22. Department IT/IS has average absences of 10
23. average salary in department IT/IS is 92111
24. Department IT/IS has average salary of 92111
25. Average salary for department IT/IS is 92111

**Result**: ✓ Correct (matches ground truth)

---

## Query 35
**Query**: What facts are available about salary information?

**Response** (0.00s):
```
Could not parse fact-based query. Please specify an employee name or ask for highest/lowest employee.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. Recruitment source Employee Referral → has → average salary of 64748
2. Employees from recruitment source Employee Referral → have → average salary of 64748
3. Department Production → has → average salary of 58741
4. Employees from recruitment source On-line Web application → have → average salary of 52505
5. average salary in department Admin Offices → relates to → 84683
6. Employees from recruitment source CareerBuilder → has → average salary of 66441
7. Department Software Engineering → has → average salary of 95660
8. average salary in department Software Engineering → relates to → 95660
9. Recruitment source CareerBuilder → has → average salary of 66441
10. average salary in department Admin Offices → is → 84683
11. Employees from recruitment source LinkedIn → have → average salary of 72925
12. Recruitment source LinkedIn → has → average salary of 72925
13. Employees from recruitment source CareerBuilder → have → average salary of 66441
14. Recruitment source On-line Web application → has → average salary of 52505
15. Employees from recruitment source Employee Referral → has → average salary of 64748

**Result**: ✓ Correct (matches ground truth)

---

## Query 36
**Query**: Retrieve facts related to performance scores

**Response** (0.00s):
```
Could not parse fact-based query. Please specify an employee name or ask for highest/lowest employee.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. Department IT/IS → has → average performance score of 3
2. line Web application → has → average performance score of 3
3. Recruitment source CareerBuilder → has → average performance score of 2
4. Department Software Engineering → has → average performance score of 3
5. Recruitment source LinkedIn → has → average performance score of 3
6. Manager Amy Dunn → has → average team performance score of 3
7. Recruitment source Diversity Job Fair → has → average performance score of 3
8. Manager Michael Albert → has → average team performance score of 3
9. Recruitment source Indeed → has → average performance score of 3
10. Department Production → has → average performance score of 3
11. Recruitment source Employee Referral → has → average performance score of 3
12. Department Admin Offices → has → average performance score of 3
13. Recruitment source On-line Web application → has → average performance score of 3
14. the average performance score for recruitment source on → is → line Web application is 3.00
15. Manager Simon Roup → has → average team performance score of 3

**Result**: ✓ Correct (matches ground truth)

---

## Query 37
**Query**: Find facts about engagement by manager

**Response** (0.00s):
```
Could not parse fact-based query. Please specify an employee name or ask for highest/lowest employee.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. lowest engagement → is → Becker, Scott
2. Employee Becker, Scott → has → engagement score of 4
3. Manager Simon Roup → type → entity
4. Blount, Dianna → has → engagement score of 2
5. Employee Biden, Lowan  M → has → engagement score of 4
6. Manager Michael Albert → has → average engagement survey value of 4
7. Manager Michael Albert → has → average team engagement score of 4
8. Biden, Lowan  M (managed → has → engagement score of 4
9. lowest engagement → relates to → Andreola, Colby
10. Manager Amy Dunn → has → average engagement survey score of 4
11. Manager Simon Roup → has → average engagement survey value of 4
12. Barbossa, Hector from IT/IS department → has → engagement score of 4
13. Andreola, Colby (managed → has → engagement score of 3
14. Blount, Dianna (managed → has → engagement score of 2
15. Manager Michael Albert → has → average engagement survey score of 4

**Result**: ✓ Correct (matches ground truth)

---

## Query 38
**Query**: Search for facts containing 'department' and 'salary'

**Response** (21.77s):
```
Average salary by department:
• Software Engineering: $95,660.00
• IT/IS: $92,111.20
• Admin Offices: $84,683.50
• Production: $58,741.64
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. Recruitment source Employee Referral → has → average salary of 64748
2. Employees from recruitment source Employee Referral → have → average salary of 64748
3. Department IT/IS → has → average performance score of 3
4. Department IT/IS → has → 5 employees
5. Department Production → has → average salary of 58741
6. Department Software Engineering → has → average absences of 19
7. Employees from recruitment source On-line Web application → have → average salary of 52505
8. Barbossa, Hector from IT/IS department → has → engagement score of 4
9. IS department → has → engagement score of 4
10. department → is → Production is 9.86 days
11. average salary in department Admin Offices → relates to → 84683
12. Employees from recruitment source CareerBuilder → has → average salary of 66441
13. Department Software Engineering → has → average salary of 95660
14. average salary in department Software Engineering → relates to → 95660
15. Recruitment source CareerBuilder → has → average salary of 66441

**Result**: ✓ Correct (matches ground truth)

---

## Query 39
**Query**: Find facts about 'performance' and 'manager'

**Response** (0.00s):
```
Could not parse fact-based query. Please specify an employee name or ask for highest/lowest employee.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. Manager Simon Roup → type → entity
2. Manager Michael Albert → has → average engagement survey value of 4
3. Department IT/IS → has → average performance score of 3
4. Manager Michael Albert → has → average team engagement score of 4
5. Manager Amy Dunn → has → average engagement survey score of 4
6. Manager Simon Roup → has → average engagement survey value of 4
7. Manager Michael Albert → has → average engagement survey score of 4
8. Manager Amy Dunn → has → average team satisfaction score of 3
9. line Web application → has → average performance score of 3
10. Recruitment source CareerBuilder → has → average performance score of 2
11. Manager Amy Dunn's team → has → average engagement survey value of 4
12. Manager Simon Roup's team → has → average engagement survey value of 4
13. Department Software Engineering → has → average performance score of 3
14. Recruitment source LinkedIn → has → average performance score of 3
15. Manager Amy Dunn → has → average team performance score of 3

**Result**: ✓ Correct (matches ground truth)

---

## Query 40
**Query**: Retrieve facts with keywords 'engagement' and 'team'

**Response** (0.00s):
```
Could not parse fact-based query. Please specify an employee name or ask for highest/lowest employee.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. lowest engagement → is → Becker, Scott
2. Employee Becker, Scott → has → engagement score of 4
3. Blount, Dianna → has → engagement score of 2
4. Employee Biden, Lowan  M → has → engagement score of 4
5. Manager Michael Albert → has → average engagement survey value of 4
6. Manager Michael Albert → has → average team engagement score of 4
7. Biden, Lowan  M (managed → has → engagement score of 4
8. lowest engagement → relates to → Andreola, Colby
9. Manager Amy Dunn → has → average engagement survey score of 4
10. Manager Simon Roup → has → average engagement survey value of 4
11. Barbossa, Hector from IT/IS department → has → engagement score of 4
12. Andreola, Colby (managed → has → engagement score of 3
13. Blount, Dianna (managed → has → engagement score of 2
14. Manager Michael Albert → has → average engagement survey score of 4
15. IS department → has → engagement score of 4

**Result**: ✓ Correct (matches ground truth)

---

## Query 41
**Query**: What facts are stored about IT/IS department?

**Response** (1.24s):
```
Facts about IT/IS department (10 facts):
1. Department IT/IS has average performance score of 3
2. Department IT/IS has 5 employees
3. Barbossa, Hector from IT/IS department has engagement score of 4
4. IS department has engagement score of 4
5. IS department has 16 absences (rank 4)
...
```

**Evidence**: 25 facts retrieved

**📄 Document Agent Facts (Direct CSV):**
1. Barbossa, Hector from IT/IS department has engagement score of 4
2. Bacong, Alejandro  from IT/IS department has 16 absences (rank 4)

**📊 Operational Insights:**
3. Department IT/IS → has → average performance score of 3
4. Department IT/IS → has → 5 employees
5. Department Production → has → average salary of 58741
6. Department Software Engineering → has → average absences of 19
7. Barbossa, Hector from IT/IS department → has → engagement score of 4
8. IS department → has → engagement score of 4
9. department → is → Production is 9.86 days
10. average salary in department Admin Offices → relates to → 84683
11. Department Software Engineering → has → average salary of 95660
12. average salary in department Software Engineering → relates to → 95660
13. average salary in department Admin Offices → is → 84683
14. IS department → has → 16 absences (rank 4)
15. Department Software Engineering → has → average performance score of 3
16. Department Admin Offices → has → 2 employees
17. Billis, Helen from Production        department → has → 19 absences (rank 3)
18. Department IT/IS has average performance score of 3
19. Department IT/IS has 5 employees
20. IS department has engagement score of 4
21. IS department has 16 absences (rank 4)
22. Department IT/IS has average absences of 10
23. average salary in department IT/IS is 92111
24. Department IT/IS has average salary of 92111
25. Average salary for department IT/IS is 92111

**Result**: ✓ Correct (matches ground truth)

---

## Query 42
**Query**: Show me all facts related to Production department

**Response** (1.34s):
```
Facts about Production department (8 facts):
1. Department Production has average salary of 58741
2. Average salary for department Production relates to 58741
3. Department Production has average performance score of 3
4. average salary in department Production relates to 58741
5. Average ...
```

**Evidence**: 23 facts retrieved

**📊 Operational Insights:**
1. Department IT/IS → has → average performance score of 3
2. Department IT/IS → has → 5 employees
3. Department Production → has → average salary of 58741
4. Department Software Engineering → has → average absences of 19
5. Barbossa, Hector from IT/IS department → has → engagement score of 4
6. IS department → has → engagement score of 4
7. department → is → Production is 9.86 days
8. average salary in department Admin Offices → relates to → 84683
9. Department Software Engineering → has → average salary of 95660
10. average salary in department Software Engineering → relates to → 95660
11. average salary in department Admin Offices → is → 84683
12. IS department → has → 16 absences (rank 4)
13. Department Software Engineering → has → average performance score of 3
14. Department Admin Offices → has → 2 employees
15. Billis, Helen from Production        department → has → 19 absences (rank 3)
16. Department Production has average salary of 58741
17. Average salary for department Production relates to 58741
18. Department Production has average performance score of 3
19. average salary in department Production relates to 58741
20. Average salary for department Production is 58741
21. Department Production has 22 employees
22. Department Production has average absences of 9
23. average salary in department Production is 58741

**Result**: ✓ Correct (matches ground truth)

---

## Query 43
**Query**: Retrieve facts about Sales department

**Response** (1.99s):
```
No facts found specifically about Sales department.
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. Department IT/IS → has → average performance score of 3
2. Department IT/IS → has → 5 employees
3. Department Production → has → average salary of 58741
4. Department Software Engineering → has → average absences of 19
5. Barbossa, Hector from IT/IS department → has → engagement score of 4
6. IS department → has → engagement score of 4
7. department → is → Production is 9.86 days
8. average salary in department Admin Offices → relates to → 84683
9. Department Software Engineering → has → average salary of 95660
10. average salary in department Software Engineering → relates to → 95660
11. average salary in department Admin Offices → is → 84683
12. IS department → has → 16 absences (rank 4)
13. Department Software Engineering → has → average performance score of 3
14. Department Admin Offices → has → 2 employees
15. Billis, Helen from Production        department → has → 19 absences (rank 3)

**Result**: ✓ Correct (matches ground truth)

---

## Query 44
**Query**: Show me all facts about employee Barbossa, Hector

**Response** (1.78s):
```
No facts found for employee Barbossa, Hector
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. Bottom 2 employee → is → Andreola, Colby
3. Employee Becker, Scott → has → engagement score of 4
4. Recruitment source Employee Referral → has → average salary of 64748
5. Bottom 1 employee → is → Blount, Dianna
6. employees by absences → is → 1. Andreola, Colby
7. Employees from recruitment source Employee Referral → have → average salary of 64748
8. absences and → is → ranked 2 in top employees
9. employee → relates to → Athwal, Sam
10. Employee Biden, Lowan  M → has → engagement score of 4
11. employee → is → Biden, Lowan  M
12. Department IT/IS → has → 5 employees
13. Employee Referral → type → entity
14. employee → is → Becker, Scott
15. Employee Bacong, Alejandro → has → 16 absences (rank 4 highest)

**Result**: ✓ Correct (matches ground truth)

---

## Query 45
**Query**: What facts are stored about employee Becker, Scott?

**Response** (1.92s):
```
No facts found for employee Becker, Scott
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. lowest engagement → is → Becker, Scott
3. Bottom 2 employee → is → Andreola, Colby
4. Employee Becker, Scott → has → engagement score of 4
5. Recruitment source Employee Referral → has → average salary of 64748
6. Bottom 1 employee → is → Blount, Dianna
7. employees by absences → is → 1. Andreola, Colby
8. Employees from recruitment source Employee Referral → have → average salary of 64748
9. absences and → is → ranked 2 in top employees
10. employee → relates to → Athwal, Sam
11. Employee Biden, Lowan  M → has → engagement score of 4
12. employee → is → Biden, Lowan  M
13. Department IT/IS → has → 5 employees
14. Employee Referral → type → entity
15. employee → is → Becker, Scott

**Result**: ✓ Correct (matches ground truth)

---

## Query 46
**Query**: Retrieve all information about employee Bacong, Alejandro

**Response** (2.29s):
```
No facts found for employee Bacong, Alejandro
```

**Evidence**: 15 facts retrieved

**📊 Operational Insights:**
1. employee → relates to → Barbossa, Hector
2. Bottom 2 employee → is → Andreola, Colby
3. Employee Becker, Scott → has → engagement score of 4
4. Recruitment source Employee Referral → has → average salary of 64748
5. Bottom 1 employee → is → Blount, Dianna
6. employees by absences → is → 1. Andreola, Colby
7. Employees from recruitment source Employee Referral → have → average salary of 64748
8. absences and → is → ranked 2 in top employees
9. employee → relates to → Athwal, Sam
10. Employee Biden, Lowan  M → has → engagement score of 4
11. employee → is → Biden, Lowan  M
12. Department IT/IS → has → 5 employees
13. Employee Referral → type → entity
14. employee → is → Becker, Scott
15. Employee Bacong, Alejandro → has → 16 absences (rank 4 highest)

**Result**: ✓ Correct (matches ground truth)

---

## Summary Statistics

- **Total Queries**: 46
- **Correct Answers**: 39/46 (84.8%)
- **Incorrect Answers**: 7/46 (15.2%)
- **Average Response Time**: 7.51s
- **Total Evidence Facts**: 358
- **Average Evidence per Query**: 7.8
- **Evidence Retrieval Queries**: 22
- **Queries with Evidence**: 22/22 (100.0%)

## Incorrect Queries

The following 7 queries were marked as incorrect:

1. **Query 1**: What is the distribution of performance scores by department?
2. **Query 16**: Which recruitment sources have the employees with highest performance score?
3. **Query 18**: Identify employees with high performance, low engagement and many special projects
4. **Query 19**: Find employees with high performance, low engagement and low satisfaction
5. **Query 20**: Find employees with high performance, low engagement and low satisfaction and many special projects
6. **Query 21**: Find employees with low engagement and low satisfaction and many special projects and many absences
7. **Query 23**: Analyze the relationship between salary, performance, and department

## Response Patterns

1. **Operational Queries (1-17, 22-24)**: Most provide structured lists with bullet points showing department/manager/recruitment source metrics
2. **Strategic Queries (18-21, 23)**: Return employee names or department analysis, but some fail evaluation criteria
3. **Evidence Queries (25-46)**: Many return "No facts found" or parsing errors, but still retrieve evidence from knowledge graph
4. **Parsing Errors**: Several queries (27, 35-37, 39-40) show "Could not parse fact-based query" but still retrieve evidence

## Evidence Fact Categories

- **Document Agent Facts**: Direct facts extracted from CSV (highest priority, most reliable)
- **Operational Insights**: Aggregated/computed facts from knowledge graph
- **Statistics**: Statistical analysis facts (correlations, distributions)
