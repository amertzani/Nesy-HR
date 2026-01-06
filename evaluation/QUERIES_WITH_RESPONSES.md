# All Tested Queries with System Responses and Extracted Facts

**Total Queries**: 30
**Source**: Offline Evaluation Report (Generated: 2025-12-14 19:26:01)
**Dataset**: HRDataset_v14.csv (311 rows, 36 columns)

---

## Query 1
**Query**: What is the distribution of performance scores by department?

**Response** (0.15s):
```
Average performance by department:
  • Software Engineering: 3.09
  • IT/IS: 3.06
  • Executive Office: 3.00
  • Admin Offices: 3.00
  • Production       : 2.98
  • Sales: 2.77
```

**Evidence**: 6 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Department Admin Offices → has → average performance score of 3
2. Department Executive Office → has → average performance score of 3
3. Department IT/IS → has → average performance score of 3
4. Department Production → has → average performance score of 2
5. Department Sales → has → average performance score of 2
6. Department Software Engineering → has → average performance score of 3

**Result**: ✗ Incorrect: None

---

## Query 2
**Query**: How do performance scores vary across departments?

**Response** (0.15s):
```
Average performance by department:
  • Software Engineering: 3.09
  • IT/IS: 3.06
  • Executive Office: 3.00
  • Admin Offices: 3.00
  • Production       : 2.98
  • Sales: 2.77
```

**Evidence**: 6 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Department Admin Offices → has → average performance score of 3
2. Department Executive Office → has → average performance score of 3
3. Department IT/IS → has → average performance score of 3
4. Department Production → has → average performance score of 2
5. Department Sales → has → average performance score of 2
6. Department Software Engineering → has → average performance score of 3

**Result**: ✓ Correct (matches ground truth)

---

## Query 3
**Query**: Which department has the highest average performance score?

**Response** (0.16s):
```
Software Engineering has the highest average performance of 3.09
```

**Evidence**: 6 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Department Admin Offices → has → average performance score of 3
2. Department Executive Office → has → average performance score of 3
3. Department IT/IS → has → average performance score of 3
4. Department Production → has → average performance score of 2
5. Department Sales → has → average performance score of 2
6. Department Software Engineering → has → average performance score of 3

**Result**: ✗ Incorrect: None

---

## Query 4
**Query**: Show me performance metrics by department

**Response** (0.17s):
```
Average performance by department:
  • Software Engineering: 3.09
  • IT/IS: 3.06
  • Executive Office: 3.00
  • Admin Offices: 3.00
  • Production       : 2.98
  • Sales: 2.77
```

**Evidence**: 6 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Department Admin Offices → has → average performance score of 3
2. Department Executive Office → has → average performance score of 3
3. Department IT/IS → has → average performance score of 3
4. Department Production → has → average performance score of 2
5. Department Sales → has → average performance score of 2
6. Department Software Engineering → has → average performance score of 3

**Result**: ✓ Correct (matches ground truth)

---

## Query 5
**Query**: What is the average special projects count by department?

**Response** (0.15s):
```
No relevant facts found in knowledge graph.
```

**Evidence**: No facts retrieved

**Result**: ✗ Incorrect: Only 0/6 entities matched (need 2+ or 50%+)

---

## Query 6
**Query**: How do special projects vary across departments?

**Response** (0.16s):
```
No relevant facts found in knowledge graph.
```

**Evidence**: No facts retrieved

**Result**: ✗ Incorrect: Only 0/6 entities matched (need 2+ or 50%+)

---

## Query 7
**Query**: Which department has the highest average special projects count?

**Response** (0.16s):
```
No relevant facts found in knowledge graph.
```

**Evidence**: No facts retrieved

**Result**: ✗ Incorrect: None

---

## Query 8
**Query**: Show me special projects distribution by department

**Response** (0.15s):
```
No relevant facts found in knowledge graph.
```

**Evidence**: No facts retrieved

**Result**: ✗ Incorrect: Only 0/6 entities matched (need 2+ or 50%+)

---

## Query 9
**Query**: What is the team-level engagement by manager?

**Response** (0.08s):
```
Average engagement by manager:
  • Eric Dougall: 4.58
  • Kelley Spirea: 4.48
  • Brandon R. LeBlanc: 4.35
  • Webster Butler: 4.33
  • Simon Roup: 4.23
  • Janet King: 4.18
  • David Stanley: 4.15
  • Alex Sweetwater: 4.08
  • Elijiah Gray: 4.07
  • Michael Albert: 4.07
  • Brian Champaigne: 4.06
 ...
```

**Evidence**: 50 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. LeBlanc's team → has → average engagement survey value of 4
2. Manager Alex Sweetwater → has average engagement survey value → 4.08
3. Manager Alex Sweetwater → has → average engagement survey value of 4
4. Manager Amy Dunn → has average engagement survey value → 3.92
5. Manager Amy Dunn → has → average team engagement score of 3
6. Manager Board of Directors → has average engagement survey value → 4.92
7. Manager Board of Directors → has → average engagement survey score of 4
8. Manager Board of Directors → has → average team engagement score of 4
9. Manager Brandon R. LeBlanc → has average engagement survey value → 4.35
10. Manager Brannon Miller → has → average engagement survey score of 4
11. Manager Brannon Miller's team → has → average engagement survey value of 4
12. Manager Brian Champaigne → has → average engagement survey score of 4
13. Manager Brian Champaigne → has → average team engagement score of 4
14. Manager David Stanley → has average engagement survey value → 4.15
15. Manager David Stanley → has → average engagement survey value of 4
16. Manager David Stanley's team → has → average engagement survey value of 4
17. Manager Debra Houlihan → has average engagement survey value → 3.84
18. Manager Debra Houlihan → has → average team engagement score of 3
19. Manager Debra Houlihan's team → has → average engagement survey value of 3
20. Manager Elijiah Gray → has → average engagement survey score of 4
21. Manager Elijiah Gray → has → average team engagement score of 4
22. Manager Eric Dougall → has → average engagement survey score of 4
23. Manager Eric Dougall → has → average team engagement score of 4
24. Manager Janet King's team → has → average engagement survey value of 4
25. Manager Jennifer Zamora → has → average engagement survey score of 3
26. Manager John Smith → has → average engagement survey value of 3
27. Manager Kelley Spirea → has → average engagement survey score of 4
28. Manager Kelley Spirea → has → average team engagement score of 4
29. Manager Kelley Spirea's team → has → average engagement survey value of 4
30. Manager Ketsia Liebig → has average engagement survey value → 4.05
31. Manager Ketsia Liebig → has → average engagement survey score of 4
32. Manager Ketsia Liebig → has → average engagement survey value of 4
33. Manager Ketsia Liebig → has → average team engagement score of 4
34. Manager Kissy Sullivan → has → average engagement survey value of 4
35. Manager Kissy Sullivan → has → average team engagement score of 4
36. Manager Kissy Sullivan's team → has → average engagement survey value of 4
37. Manager Lynn Daneault → has average engagement survey value → 3.80
38. Manager Lynn Daneault → has → average engagement survey score of 3
39. Manager Lynn Daneault → has → average engagement survey value of 3
40. Manager Michael Albert → has → average engagement survey score of 4
41. Manager Michael Albert → has → average team engagement score of 4
42. Manager Peter Monroe → has average engagement survey value → 4.03
43. Manager Peter Monroe → has → average engagement survey score of 4
44. Manager Peter Monroe → has → average team engagement score of 4
45. Manager Simon Roup → has average engagement survey value → 4.23
46. Manager Simon Roup → has → average engagement survey score of 4
47. Manager Simon Roup's team → has → average engagement survey value of 4
48. Manager Webster Butler → has → average engagement survey value of 4
49. Manager Webster Butler's team → has → average engagement survey value of 4
50. s team → has → average engagement survey value of 4

**Result**: ✓ Correct (matches ground truth)

---

## Query 10
**Query**: How does engagement vary by manager?

**Response** (0.07s):
```
Average engagement by manager:
  • Eric Dougall: 4.58
  • Kelley Spirea: 4.48
  • Brandon R. LeBlanc: 4.35
  • Webster Butler: 4.33
  • Simon Roup: 4.23
  • Janet King: 4.18
  • David Stanley: 4.15
  • Alex Sweetwater: 4.08
  • Elijiah Gray: 4.07
  • Michael Albert: 4.07
  • Brian Champaigne: 4.06
 ...
```

**Evidence**: 50 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. LeBlanc's team → has → average engagement survey value of 4
2. Manager Alex Sweetwater → has average engagement survey value → 4.08
3. Manager Alex Sweetwater → has → average engagement survey value of 4
4. Manager Amy Dunn → has average engagement survey value → 3.92
5. Manager Amy Dunn → has → average team engagement score of 3
6. Manager Board of Directors → has average engagement survey value → 4.92
7. Manager Board of Directors → has → average engagement survey score of 4
8. Manager Board of Directors → has → average team engagement score of 4
9. Manager Brandon R. LeBlanc → has average engagement survey value → 4.35
10. Manager Brannon Miller → has → average engagement survey score of 4
11. Manager Brannon Miller's team → has → average engagement survey value of 4
12. Manager Brian Champaigne → has → average engagement survey score of 4
13. Manager Brian Champaigne → has → average team engagement score of 4
14. Manager David Stanley → has average engagement survey value → 4.15
15. Manager David Stanley → has → average engagement survey value of 4
16. Manager David Stanley's team → has → average engagement survey value of 4
17. Manager Debra Houlihan → has average engagement survey value → 3.84
18. Manager Debra Houlihan → has → average team engagement score of 3
19. Manager Debra Houlihan's team → has → average engagement survey value of 3
20. Manager Elijiah Gray → has → average engagement survey score of 4
21. Manager Elijiah Gray → has → average team engagement score of 4
22. Manager Eric Dougall → has → average engagement survey score of 4
23. Manager Eric Dougall → has → average team engagement score of 4
24. Manager Janet King's team → has → average engagement survey value of 4
25. Manager Jennifer Zamora → has → average engagement survey score of 3
26. Manager John Smith → has → average engagement survey value of 3
27. Manager Kelley Spirea → has → average engagement survey score of 4
28. Manager Kelley Spirea → has → average team engagement score of 4
29. Manager Kelley Spirea's team → has → average engagement survey value of 4
30. Manager Ketsia Liebig → has average engagement survey value → 4.05
31. Manager Ketsia Liebig → has → average engagement survey score of 4
32. Manager Ketsia Liebig → has → average engagement survey value of 4
33. Manager Ketsia Liebig → has → average team engagement score of 4
34. Manager Kissy Sullivan → has → average engagement survey value of 4
35. Manager Kissy Sullivan → has → average team engagement score of 4
36. Manager Kissy Sullivan's team → has → average engagement survey value of 4
37. Manager Lynn Daneault → has average engagement survey value → 3.80
38. Manager Lynn Daneault → has → average engagement survey score of 3
39. Manager Lynn Daneault → has → average engagement survey value of 3
40. Manager Michael Albert → has → average engagement survey score of 4
41. Manager Michael Albert → has → average team engagement score of 4
42. Manager Peter Monroe → has average engagement survey value → 4.03
43. Manager Peter Monroe → has → average engagement survey score of 4
44. Manager Peter Monroe → has → average team engagement score of 4
45. Manager Simon Roup → has average engagement survey value → 4.23
46. Manager Simon Roup → has → average engagement survey score of 4
47. Manager Simon Roup's team → has → average engagement survey value of 4
48. Manager Webster Butler → has → average engagement survey value of 4
49. Manager Webster Butler's team → has → average engagement survey value of 4
50. s team → has → average engagement survey value of 4

**Result**: ✓ Correct (matches ground truth)

---

## Query 11
**Query**: Which manager has the highest team engagement?

**Response** (0.07s):
```
Eric Dougall has the highest average engagement of 4.58
```

**Evidence**: 50 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. LeBlanc's team → has → average engagement survey value of 4
2. Manager Alex Sweetwater → has average engagement survey value → 4.08
3. Manager Alex Sweetwater → has → average engagement survey value of 4
4. Manager Amy Dunn → has average engagement survey value → 3.92
5. Manager Amy Dunn → has → average team engagement score of 3
6. Manager Board of Directors → has average engagement survey value → 4.92
7. Manager Board of Directors → has → average engagement survey score of 4
8. Manager Board of Directors → has → average team engagement score of 4
9. Manager Brandon R. LeBlanc → has average engagement survey value → 4.35
10. Manager Brannon Miller → has → average engagement survey score of 4
11. Manager Brannon Miller's team → has → average engagement survey value of 4
12. Manager Brian Champaigne → has → average engagement survey score of 4
13. Manager Brian Champaigne → has → average team engagement score of 4
14. Manager David Stanley → has average engagement survey value → 4.15
15. Manager David Stanley → has → average engagement survey value of 4
16. Manager David Stanley's team → has → average engagement survey value of 4
17. Manager Debra Houlihan → has average engagement survey value → 3.84
18. Manager Debra Houlihan → has → average team engagement score of 3
19. Manager Debra Houlihan's team → has → average engagement survey value of 3
20. Manager Elijiah Gray → has → average engagement survey score of 4
21. Manager Elijiah Gray → has → average team engagement score of 4
22. Manager Eric Dougall → has → average engagement survey score of 4
23. Manager Eric Dougall → has → average team engagement score of 4
24. Manager Janet King's team → has → average engagement survey value of 4
25. Manager Jennifer Zamora → has → average engagement survey score of 3
26. Manager John Smith → has → average engagement survey value of 3
27. Manager Kelley Spirea → has → average engagement survey score of 4
28. Manager Kelley Spirea → has → average team engagement score of 4
29. Manager Kelley Spirea's team → has → average engagement survey value of 4
30. Manager Ketsia Liebig → has average engagement survey value → 4.05
31. Manager Ketsia Liebig → has → average engagement survey score of 4
32. Manager Ketsia Liebig → has → average engagement survey value of 4
33. Manager Ketsia Liebig → has → average team engagement score of 4
34. Manager Kissy Sullivan → has → average engagement survey value of 4
35. Manager Kissy Sullivan → has → average team engagement score of 4
36. Manager Kissy Sullivan's team → has → average engagement survey value of 4
37. Manager Lynn Daneault → has average engagement survey value → 3.80
38. Manager Lynn Daneault → has → average engagement survey score of 3
39. Manager Lynn Daneault → has → average engagement survey value of 3
40. Manager Michael Albert → has → average engagement survey score of 4
41. Manager Michael Albert → has → average team engagement score of 4
42. Manager Peter Monroe → has average engagement survey value → 4.03
43. Manager Peter Monroe → has → average engagement survey score of 4
44. Manager Peter Monroe → has → average team engagement score of 4
45. Manager Simon Roup → has average engagement survey value → 4.23
46. Manager Simon Roup → has → average engagement survey score of 4
47. Manager Simon Roup's team → has → average engagement survey value of 4
48. Manager Webster Butler → has → average engagement survey value of 4
49. Manager Webster Butler's team → has → average engagement survey value of 4
50. s team → has → average engagement survey value of 4

**Result**: ✓ Correct (matches ground truth)

---

## Query 12
**Query**: What is the average salary by department?

**Response** (0.16s):
```
Average salary by department:
  • Executive Office: $250,000.00
  • IT/IS: $97,064.64
  • Software Engineering: $94,989.45
  • Admin Offices: $71,791.89
  • Sales: $69,061.26
  • Production       : $59,953.55
```

**Evidence**: 28 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Average salary for department Admin Offices → is → 71791
2. Average salary for department Admin Offices → relates to → 71791
3. Average salary for department Executive Office → is → 250000
4. Average salary for department Executive Office → relates to → 250000
5. Average salary for department IT/IS → is → 97064
6. Average salary for department Production → is → 59953
7. Average salary for department Production → relates to → 59953
8. Average salary for department Sales → is → 69061
9. Average salary for department Sales → relates to → 69061
10. Average salary for department Software Engineering → is → 94989
11. Average salary for department Software Engineering → relates to → 94989
12. Department Admin Offices → has → average salary of 71791
13. Department Executive Office → has → average salary of 250000
14. Department IT/IS → has → average salary of 97064
15. Department Production → has → average salary of 59953
16. Department Sales → has → average salary of 69061
17. Department Software Engineering → has → average salary of 94989
18. average salary in department Admin Offices → is → 71791
19. average salary in department Admin Offices → relates to → 71791
20. average salary in department Executive Office → is → 250000
21. average salary in department Executive Office → relates to → 250000
22. average salary in department IT/IS → is → 97064
23. average salary in department Production → is → 59953
24. average salary in department Production → relates to → 59953
25. average salary in department Sales → is → 69061
26. average salary in department Sales → relates to → 69061
27. average salary in department Software Engineering → is → 94989
28. average salary in department Software Engineering → relates to → 94989

**Result**: ✓ Correct (matches ground truth)

---

## Query 13
**Query**: How does salary distribution vary across departments?

**Response** (0.14s):
```
Average salary by department:
  • Executive Office: $250,000.00
  • IT/IS: $97,064.64
  • Software Engineering: $94,989.45
  • Admin Offices: $71,791.89
  • Sales: $69,061.26
  • Production       : $59,953.55
```

**Evidence**: 28 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Average salary for department Admin Offices → is → 71791
2. Average salary for department Admin Offices → relates to → 71791
3. Average salary for department Executive Office → is → 250000
4. Average salary for department Executive Office → relates to → 250000
5. Average salary for department IT/IS → is → 97064
6. Average salary for department Production → is → 59953
7. Average salary for department Production → relates to → 59953
8. Average salary for department Sales → is → 69061
9. Average salary for department Sales → relates to → 69061
10. Average salary for department Software Engineering → is → 94989
11. Average salary for department Software Engineering → relates to → 94989
12. Department Admin Offices → has → average salary of 71791
13. Department Executive Office → has → average salary of 250000
14. Department IT/IS → has → average salary of 97064
15. Department Production → has → average salary of 59953
16. Department Sales → has → average salary of 69061
17. Department Software Engineering → has → average salary of 94989
18. average salary in department Admin Offices → is → 71791
19. average salary in department Admin Offices → relates to → 71791
20. average salary in department Executive Office → is → 250000
21. average salary in department Executive Office → relates to → 250000
22. average salary in department IT/IS → is → 97064
23. average salary in department Production → is → 59953
24. average salary in department Production → relates to → 59953
25. average salary in department Sales → is → 69061
26. average salary in department Sales → relates to → 69061
27. average salary in department Software Engineering → is → 94989
28. average salary in department Software Engineering → relates to → 94989

**Result**: ✓ Correct (matches ground truth)

---

## Query 14
**Query**: Which department has the highest average salary?

**Response** (0.16s):
```
Executive Office has the highest average salary of $250,000.00
```

**Evidence**: 28 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Average salary for department Admin Offices → is → 71791
2. Average salary for department Admin Offices → relates to → 71791
3. Average salary for department Executive Office → is → 250000
4. Average salary for department Executive Office → relates to → 250000
5. Average salary for department IT/IS → is → 97064
6. Average salary for department Production → is → 59953
7. Average salary for department Production → relates to → 59953
8. Average salary for department Sales → is → 69061
9. Average salary for department Sales → relates to → 69061
10. Average salary for department Software Engineering → is → 94989
11. Average salary for department Software Engineering → relates to → 94989
12. Department Admin Offices → has → average salary of 71791
13. Department Executive Office → has → average salary of 250000
14. Department IT/IS → has → average salary of 97064
15. Department Production → has → average salary of 59953
16. Department Sales → has → average salary of 69061
17. Department Software Engineering → has → average salary of 94989
18. average salary in department Admin Offices → is → 71791
19. average salary in department Admin Offices → relates to → 71791
20. average salary in department Executive Office → is → 250000
21. average salary in department Executive Office → relates to → 250000
22. average salary in department IT/IS → is → 97064
23. average salary in department Production → is → 59953
24. average salary in department Production → relates to → 59953
25. average salary in department Sales → is → 69061
26. average salary in department Sales → relates to → 69061
27. average salary in department Software Engineering → is → 94989
28. average salary in department Software Engineering → relates to → 94989

**Result**: ✓ Correct (matches ground truth)

---

## Query 15
**Query**: How does performance vary by recruitment source?

**Response** (0.16s):
```
Distribution of performance by recruitment:
  • Employee Referral: 3.10
  • On-line Web application: 3.00
  • Diversity Job Fair: 3.00
  • Other: 3.00
  • LinkedIn: 3.00
  • Indeed: 2.98
  • Google Search: 2.94
  • CareerBuilder: 2.91
  • Website: 2.77
```

**Evidence**: 10 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Recruitment source CareerBuilder → has → average performance score of 2
2. Recruitment source Diversity Job Fair → has → average performance score of 3
3. Recruitment source Employee Referral → has → average performance score of 3
4. Recruitment source Google Search → has → average performance score of 2
5. Recruitment source Indeed → has → average performance score of 2
6. Recruitment source LinkedIn → has → average performance score of 3
7. Recruitment source On-line Web application → has → average performance score of 3
8. Recruitment source Other → has → average performance score of 3
9. Recruitment source Website → has → average performance score of 2
10. the average performance score for recruitment source on → is → line Web application is 3.00

**Result**: ✓ Correct (matches ground truth)

---

## Query 16
**Query**: Which recruitment sources have the employees with highest performance score?

**Response** (0.19s):
```
Employee Referral has the highest average performance of 3.10
```

**Evidence**: 10 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Recruitment source CareerBuilder → has → average performance score of 2
2. Recruitment source Diversity Job Fair → has → average performance score of 3
3. Recruitment source Employee Referral → has → average performance score of 3
4. Recruitment source Google Search → has → average performance score of 2
5. Recruitment source Indeed → has → average performance score of 2
6. Recruitment source LinkedIn → has → average performance score of 3
7. Recruitment source On-line Web application → has → average performance score of 3
8. Recruitment source Other → has → average performance score of 3
9. Recruitment source Website → has → average performance score of 2
10. the average performance score for recruitment source on → is → line Web application is 3.00

**Result**: ✗ Incorrect: None

---

## Query 17
**Query**: What is the performance distribution by recruitment source?

**Response** (0.22s):
```
Distribution of performance by recruitment:
  • Employee Referral: 3.10
  • On-line Web application: 3.00
  • Diversity Job Fair: 3.00
  • Other: 3.00
  • LinkedIn: 3.00
  • Indeed: 2.98
  • Google Search: 2.94
  • CareerBuilder: 2.91
  • Website: 2.77
```

**Evidence**: 10 facts retrieved

**Extracted Facts:**

**❓ Other Facts:**
1. Recruitment source CareerBuilder → has → average performance score of 2
2. Recruitment source Diversity Job Fair → has → average performance score of 3
3. Recruitment source Employee Referral → has → average performance score of 3
4. Recruitment source Google Search → has → average performance score of 2
5. Recruitment source Indeed → has → average performance score of 2
6. Recruitment source LinkedIn → has → average performance score of 3
7. Recruitment source On-line Web application → has → average performance score of 3
8. Recruitment source Other → has → average performance score of 3
9. Recruitment source Website → has → average performance score of 2
10. the average performance score for recruitment source on → is → line Web application is 3.00

**Result**: ✗ Incorrect: None

---

## Query 18
**Query**: Identify employees with high performance, low engagement and many special projects

**Response** (2.89s):
```
Found 20 employee(s) matching the criteria:
  1. Andreola, Colby
      - Employee Andreola, Colby has RaceDesc White
      - Employee Andreola, Colby has EngagementSurvey 3.04
      - Employee Andreola, Colby has EmploymentStatus Active
      - Employee Andreola, Colby has DeptID 4
      - HRDataset v14.csv Row 6 has Employee Name Andreola, Colby
  2. Booth, Frank
      - Employee Booth, Frank has EmpSatisfaction 5
      - Employee Booth, Frank has FromDiversityJobFairID 0
      - Employee Booth, Frank has Sex M
      - Employee Booth, Frank has DateofTermination 2/19/2016
      - Employee Booth, Frank has MarriedID 0
  3. Cady, Max
      - Employee Cady, Max has Zip 2184
      - Employee Cady, Max has HispanicLatino No
      - Employee Cady, Max has State MA
      - Employee Cady, Max has SpecialProjectsCount 5
      - Employee Cady, Max has RaceDesc White
  ... and 17 more employees
```

**Evidence**: 95 facts retrieved

**Extracted Facts:**
1. Employee Andreola, Colby has RaceDesc White
2. Employee Andreola, Colby has EngagementSurvey 3.04
3. Employee Andreola, Colby has EmploymentStatus Active
4. Employee Andreola, Colby has DeptID 4
5. HRDataset v14.csv Row 6 has Employee Name Andreola, Colby
6. Employee Booth, Frank has EmpSatisfaction 5
7. Employee Booth, Frank has FromDiversityJobFairID 0
8. Employee Booth, Frank has Sex M
9. Employee Booth, Frank has DateofTermination 2/19/2016
10. Employee Booth, Frank has MarriedID 0
... and 85 more facts

**Result**: ✓ Correct (matches ground truth)

---

## Query 19
**Query**: Find employees with high performance, low engagement and low satisfaction

**Response** (2.99s):
```
Found 33 employee(s) matching the criteria:
  1. Akinkuolie, Sarah
      - Employee Akinkuolie, Sarah has MaritalDesc Married
      - Employee Akinkuolie, Sarah has EngagementSurvey 3.02
      - Employee Akinkuolie, Sarah has DOB 09/19/88
      - Employee Akinkuolie, Sarah has EmpID 10196
      - Employee Akinkuolie, Sarah has State MA
  2. Andreola, Colby
      - Employee Andreola, Colby has RaceDesc White
      - Employee Andreola, Colby has EngagementSurvey 3.04
      - Employee Andreola, Colby has EmploymentStatus Active
      - Employee Andreola, Colby has DeptID 4
      - HRDataset v14.csv Row 6 has Employee Name Andreola, Colby
  ... and 31 more employees
```

**Evidence**: 100 facts retrieved

**Extracted Facts:**
1. Employee Akinkuolie, Sarah has MaritalDesc Married
2. Employee Akinkuolie, Sarah has EngagementSurvey 3.02
3. Employee Akinkuolie, Sarah has DOB 09/19/88
4. Employee Akinkuolie, Sarah has EmpID 10196
5. Employee Akinkuolie, Sarah has State MA
6. Employee Andreola, Colby has RaceDesc White
7. Employee Andreola, Colby has EngagementSurvey 3.04
8. Employee Andreola, Colby has EmploymentStatus Active
9. Employee Andreola, Colby has DeptID 4
10. HRDataset v14.csv Row 6 has Employee Name Andreola, Colby
... and 90 more facts

**Result**: ✓ Correct (matches ground truth)

---

## Query 20
**Query**: Find employees with high performance, low engagement and low satisfaction and many special projects

**Response** (1.26s):
```
Found 10 employee(s) matching the criteria:
  1. Andreola, Colby
      - Employee Andreola, Colby has RaceDesc White
      - Employee Andreola, Colby has EngagementSurvey 3.04
      - Employee Andreola, Colby has EmploymentStatus Active
      - Employee Andreola, Colby has DeptID 4
      - HRDataset v14.csv Row 6 has Employee Name Andreola, Colby
  2. Cady, Max
      - Employee Cady, Max has Zip 2184
      - Employee Cady, Max has HispanicLatino No
      - Employee Cady, Max has State MA
      - Employee Cady, Max has SpecialProjectsCount 5
      - Employee Cady, Max has RaceDesc White
  ... and 8 more employees
```

**Evidence**: 50 facts retrieved

**Extracted Facts:**
1. Employee Andreola, Colby has RaceDesc White
2. Employee Andreola, Colby has EngagementSurvey 3.04
3. Employee Andreola, Colby has EmploymentStatus Active
4. Employee Andreola, Colby has DeptID 4
5. HRDataset v14.csv Row 6 has Employee Name Andreola, Colby
6. Employee Cady, Max has Zip 2184
7. Employee Cady, Max has HispanicLatino No
8. Employee Cady, Max has State MA
9. Employee Cady, Max has SpecialProjectsCount 5
10. Employee Cady, Max has RaceDesc White
... and 40 more facts

**Result**: ✓ Correct (matches ground truth)

---

## Query 21
**Query**: Find employees with low engagement and low satisfaction and many special projects and many absences

**Response** (1.02s):
```
Found 8 employee(s) matching the criteria:
  1. Andreola, Colby
      - Employee Andreola, Colby has RaceDesc White
      - Employee Andreola, Colby has EngagementSurvey 3.04
      - Employee Andreola, Colby has EmploymentStatus Active
      - Employee Andreola, Colby has DeptID 4
      - HRDataset v14.csv Row 6 has Employee Name Andreola, Colby
  2. Howard, Estelle
      - Employee Howard, Estelle has EmpStatusID 1
      - Employee Howard, Estelle has EmpID 10182
      - Employee Howard, Estelle has Absences 6
      - Employee Howard, Estelle has DateofHire 2/16/2015
      - Employee Howard, Estelle has PositionID 2
  ... and 6 more employees
```

**Evidence**: 40 facts retrieved

**Extracted Facts:**
1. Employee Andreola, Colby has RaceDesc White
2. Employee Andreola, Colby has EngagementSurvey 3.04
3. Employee Andreola, Colby has EmploymentStatus Active
4. Employee Andreola, Colby has DeptID 4
5. HRDataset v14.csv Row 6 has Employee Name Andreola, Colby
6. Employee Howard, Estelle has EmpStatusID 1
7. Employee Howard, Estelle has EmpID 10182
8. Employee Howard, Estelle has Absences 6
9. Employee Howard, Estelle has DateofHire 2/16/2015
10. Employee Howard, Estelle has PositionID 2
... and 30 more facts

**Result**: ✓ Correct (matches ground truth)

---

## Query 22
**Query**: Which departments have high salaries but low performance?

**Response** (0.01s):
```
No departments found with high salaries but low performance.
```

**Evidence**: No facts retrieved

**Result**: ✗ Incorrect: Only 0/17 entities matched (need 3+ or 50%+)

---

## Query 23
**Query**: Retrieve facts related with employee Becker, Scott

**Response** (0.30s):
```
Related facts (34):
  1. Employee Becker, Scott has EmpID 10277
  2. Employee Becker, Scott has Zip 2452
  3. Employee Becker, Scott has MarriedID 0
  4. Employee Becker, Scott has DateofHire 7/8/2013
  5. Employee Becker, Scott has MaritalDesc Single
  6. Employee Becker, Scott has Department Prod...
```

**Evidence**: 50 facts retrieved

**Extracted Facts:**

**📄 Document Agent Facts (Direct CSV):**
1. Employee O'hare, Lynn → has Salary → 52674
2. Employee Guilianno, Mike → has ManagerName → John Smith
3. Employee Veera, Abdellah → has Sex → M
4. Employee Becker, Scott → has EmpID → 10277
5. Employee Gross, Paula → has PerformanceScore → Fully Meets
6. Employee Keatts, Kramer → has EmpID → 10192
7. Employee Stoica, Rick → has MaritalDesc → Married
8. Employee Gonzalez, Juan → has DeptID → 5
9. HRDataset v14.csv Row 35 → has Employee Name → Cady, Max
10. Employee Wallace, Courtney  E → has MarriedID → 1
11. Employee Dietrich, Jenna → has EmploymentStatus → Active
12. Employee Buccheri, Joseph → has MarriedID → 0
13. Employee Rhoads, Thomas → has Termd → 1
14. Employee Trzeciak, Cybil → has Sex → F
15. Employee Bacong, Alejandro → has DeptID → 3
16. Employee Gonzalez, Maria → has PerformanceScore → Fully Meets
17. Employee Szabo, Andrew → has GenderID → 1
18. Employee Sloan, Constance → has State → MA
19. Employee Billis, Helen → has Termd → 0
20. HRDataset v14.csv Row 10 → has Employee Name → Baczenski, Rachael
21. Employee Clukey, Elijian → has MaritalDesc → Married
22. Employee Szabo, Andrew → has EmpID → 10024
23. Employee Kinsella, Kathleen → has Department → Production
24. Employee Smith, Sade → has State → MA
25. Employee Dolan, Linda → has Position → IT Support
26. Employee Soto, Julia → has GenderID → 0
27. Employee Foreman, Tanya → has State → MA
28. Employee Sutwell, Barbara → has ManagerID → 16.0
29. Employee Soto, Julia → has RecruitmentSource → LinkedIn
30. Employee Leruth, Giovanni → has Salary → 70468
31. Employee Becker, Scott has EmpID 10277
32. Employee Becker, Scott has Zip 2452
33. Employee Becker, Scott has MarriedID 0
34. Employee Becker, Scott has DateofHire 7/8/2013
35. Employee Becker, Scott has MaritalDesc Single
36. Employee Becker, Scott has Department Production
37. Employee Becker, Scott has Position Production Technician I
38. Employee Becker, Scott has EngagementSurvey 4.2
39. Employee Becker, Scott has CitizenDesc US Citizen
40. Employee Becker, Scott has EmpStatusID 3
41. Employee Becker, Scott has RecruitmentSource LinkedIn
42. Employee Becker, Scott has GenderID 1
43. Employee Becker, Scott has DaysLateLast30 0
44. Employee Becker, Scott has TermReason N/A-StillEmployed
45. Employee Becker, Scott has FromDiversityJobFairID 0
46. Employee Becker, Scott has SpecialProjectsCount 0
47. Employee Becker, Scott has PerfScoreID 3
48. Employee Becker, Scott has Sex M
49. Employee Becker, Scott has LastPerformanceReview Date 1/11/2019
50. Employee Becker, Scott has Termd 0

**Result**: ✓ Correct (matches ground truth)

---

## Query 24
**Query**: What information do we have about employee Becker, Scott?

**Response** (0.31s):
```
Related facts (34):
  1. Employee Becker, Scott has EmpID 10277
  2. Employee Becker, Scott has Zip 2452
  3. Employee Becker, Scott has MarriedID 0
  4. Employee Becker, Scott has DateofHire 7/8/2013
  5. Employee Becker, Scott has MaritalDesc Single
  6. Employee Becker, Scott has Department Prod...
```

**Evidence**: 50 facts retrieved

**Extracted Facts:**

**📄 Document Agent Facts (Direct CSV):**
1. Employee O'hare, Lynn → has Salary → 52674
2. Employee Guilianno, Mike → has ManagerName → John Smith
3. Employee Veera, Abdellah → has Sex → M
4. Employee Becker, Scott → has EmpID → 10277
5. Employee Gross, Paula → has PerformanceScore → Fully Meets
6. Employee Keatts, Kramer → has EmpID → 10192
7. Employee Stoica, Rick → has MaritalDesc → Married
8. Employee Gonzalez, Juan → has DeptID → 5
9. HRDataset v14.csv Row 35 → has Employee Name → Cady, Max
10. Employee Wallace, Courtney  E → has MarriedID → 1
11. Employee Dietrich, Jenna → has EmploymentStatus → Active
12. Employee Buccheri, Joseph → has MarriedID → 0
13. Employee Rhoads, Thomas → has Termd → 1
14. Employee Trzeciak, Cybil → has Sex → F
15. Employee Bacong, Alejandro → has DeptID → 3
16. Employee Gonzalez, Maria → has PerformanceScore → Fully Meets
17. Employee Szabo, Andrew → has GenderID → 1
18. Employee Sloan, Constance → has State → MA
19. Employee Billis, Helen → has Termd → 0
20. HRDataset v14.csv Row 10 → has Employee Name → Baczenski, Rachael
21. Employee Clukey, Elijian → has MaritalDesc → Married
22. Employee Szabo, Andrew → has EmpID → 10024
23. Employee Kinsella, Kathleen → has Department → Production
24. Employee Smith, Sade → has State → MA
25. Employee Dolan, Linda → has Position → IT Support
26. Employee Soto, Julia → has GenderID → 0
27. Employee Foreman, Tanya → has State → MA
28. Employee Sutwell, Barbara → has ManagerID → 16.0
29. Employee Soto, Julia → has RecruitmentSource → LinkedIn
30. Employee Leruth, Giovanni → has Salary → 70468
31. Employee Becker, Scott has EmpID 10277
32. Employee Becker, Scott has Zip 2452
33. Employee Becker, Scott has MarriedID 0
34. Employee Becker, Scott has DateofHire 7/8/2013
35. Employee Becker, Scott has MaritalDesc Single
36. Employee Becker, Scott has Department Production
37. Employee Becker, Scott has Position Production Technician I
38. Employee Becker, Scott has EngagementSurvey 4.2
39. Employee Becker, Scott has CitizenDesc US Citizen
40. Employee Becker, Scott has EmpStatusID 3
41. Employee Becker, Scott has RecruitmentSource LinkedIn
42. Employee Becker, Scott has GenderID 1
43. Employee Becker, Scott has DaysLateLast30 0
44. Employee Becker, Scott has TermReason N/A-StillEmployed
45. Employee Becker, Scott has FromDiversityJobFairID 0
46. Employee Becker, Scott has SpecialProjectsCount 0
47. Employee Becker, Scott has PerfScoreID 3
48. Employee Becker, Scott has Sex M
49. Employee Becker, Scott has LastPerformanceReview Date 1/11/2019
50. Employee Becker, Scott has Termd 0

**Result**: ✓ Correct (matches ground truth)

---

## Query 23
**Query**: Give me facts about the employee with the highest salary

**Response** (0.00s):
```
Could not determine employee with highest salary.
```

**Evidence**: 34 facts retrieved

**Extracted Facts:**

**📄 Document Agent Facts (Direct CSV):**
1. Employee O'hare, Lynn → has Salary → 52674
2. Employee Guilianno, Mike → has ManagerName → John Smith
3. Employee Veera, Abdellah → has Sex → M
4. Employee Becker, Scott → has EmpID → 10277
5. Employee Gross, Paula → has PerformanceScore → Fully Meets
6. Employee Keatts, Kramer → has EmpID → 10192
7. Employee Stoica, Rick → has MaritalDesc → Married
8. Employee Gonzalez, Juan → has DeptID → 5
9. HRDataset v14.csv Row 35 → has Employee Name → Cady, Max
10. Employee Wallace, Courtney  E → has MarriedID → 1
11. Employee Dietrich, Jenna → has EmploymentStatus → Active
12. Employee Buccheri, Joseph → has MarriedID → 0
13. Employee Rhoads, Thomas → has Termd → 1
14. Employee Trzeciak, Cybil → has Sex → F
15. Employee Bacong, Alejandro → has DeptID → 3
16. Employee Gonzalez, Maria → has PerformanceScore → Fully Meets
17. Employee Szabo, Andrew → has GenderID → 1
18. Employee Sloan, Constance → has State → MA
19. Employee Billis, Helen → has Termd → 0
20. HRDataset v14.csv Row 10 → has Employee Name → Baczenski, Rachael
21. Employee Clukey, Elijian → has MaritalDesc → Married
22. Employee Szabo, Andrew → has EmpID → 10024
23. Employee Kinsella, Kathleen → has Department → Production
24. Employee Smith, Sade → has State → MA
25. Employee Dolan, Linda → has Position → IT Support
26. Employee Soto, Julia → has GenderID → 0
27. Employee Foreman, Tanya → has State → MA
28. Employee Sutwell, Barbara → has ManagerID → 16.0
29. Employee Soto, Julia → has RecruitmentSource → LinkedIn
30. Employee Leruth, Giovanni → has Salary → 70468

**❓ Other Facts:**
1. HRDataset v14.csv Row 309 → has Salary → 89292
2. HRDataset v14.csv Row 115 → has Salary → 48285
3. HRDataset v14.csv Row 34 → has Salary → 62162
4. HRDataset v14.csv Row 13 → has Salary → 58709

**Result**: ✓ Correct (matches ground truth)

---

## Query 24
**Query**: Retrieve facts about the employee who has the highest salary

**Response** (0.00s):
```
Could not determine employee with highest salary.
```

**Evidence**: 34 facts retrieved

**Extracted Facts:**

**📄 Document Agent Facts (Direct CSV):**
1. Employee O'hare, Lynn → has Salary → 52674
2. Employee Guilianno, Mike → has ManagerName → John Smith
3. Employee Veera, Abdellah → has Sex → M
4. Employee Becker, Scott → has EmpID → 10277
5. Employee Gross, Paula → has PerformanceScore → Fully Meets
6. Employee Keatts, Kramer → has EmpID → 10192
7. Employee Stoica, Rick → has MaritalDesc → Married
8. Employee Gonzalez, Juan → has DeptID → 5
9. HRDataset v14.csv Row 35 → has Employee Name → Cady, Max
10. Employee Wallace, Courtney  E → has MarriedID → 1
11. Employee Dietrich, Jenna → has EmploymentStatus → Active
12. Employee Buccheri, Joseph → has MarriedID → 0
13. Employee Rhoads, Thomas → has Termd → 1
14. Employee Trzeciak, Cybil → has Sex → F
15. Employee Bacong, Alejandro → has DeptID → 3
16. Employee Gonzalez, Maria → has PerformanceScore → Fully Meets
17. Employee Szabo, Andrew → has GenderID → 1
18. Employee Sloan, Constance → has State → MA
19. Employee Billis, Helen → has Termd → 0
20. HRDataset v14.csv Row 10 → has Employee Name → Baczenski, Rachael
21. Employee Clukey, Elijian → has MaritalDesc → Married
22. Employee Szabo, Andrew → has EmpID → 10024
23. Employee Kinsella, Kathleen → has Department → Production
24. Employee Smith, Sade → has State → MA
25. Employee Dolan, Linda → has Position → IT Support
26. Employee Soto, Julia → has GenderID → 0
27. Employee Foreman, Tanya → has State → MA
28. Employee Sutwell, Barbara → has ManagerID → 16.0
29. Employee Soto, Julia → has RecruitmentSource → LinkedIn
30. Employee Leruth, Giovanni → has Salary → 70468

**❓ Other Facts:**
1. HRDataset v14.csv Row 309 → has Salary → 89292
2. HRDataset v14.csv Row 115 → has Salary → 48285
3. HRDataset v14.csv Row 34 → has Salary → 62162
4. HRDataset v14.csv Row 13 → has Salary → 58709

**Result**: ✓ Correct (matches ground truth)

---

## Query 23
**Query**: Show me information about the highest paid employee

**Response** (0.00s):
```
Could not determine employee with highest salary.
```

**Evidence**: 30 facts retrieved

**Extracted Facts:**

**📄 Document Agent Facts (Direct CSV):**
1. Employee O'hare, Lynn → has Salary → 52674
2. Employee Guilianno, Mike → has ManagerName → John Smith
3. Employee Veera, Abdellah → has Sex → M
4. Employee Becker, Scott → has EmpID → 10277
5. Employee Gross, Paula → has PerformanceScore → Fully Meets
6. Employee Keatts, Kramer → has EmpID → 10192
7. Employee Stoica, Rick → has MaritalDesc → Married
8. Employee Gonzalez, Juan → has DeptID → 5
9. HRDataset v14.csv Row 35 → has Employee Name → Cady, Max
10. Employee Wallace, Courtney  E → has MarriedID → 1
11. Employee Dietrich, Jenna → has EmploymentStatus → Active
12. Employee Buccheri, Joseph → has MarriedID → 0
13. Employee Rhoads, Thomas → has Termd → 1
14. Employee Trzeciak, Cybil → has Sex → F
15. Employee Bacong, Alejandro → has DeptID → 3
16. Employee Gonzalez, Maria → has PerformanceScore → Fully Meets
17. Employee Szabo, Andrew → has GenderID → 1
18. Employee Sloan, Constance → has State → MA
19. Employee Billis, Helen → has Termd → 0
20. HRDataset v14.csv Row 10 → has Employee Name → Baczenski, Rachael
21. Employee Clukey, Elijian → has MaritalDesc → Married
22. Employee Szabo, Andrew → has EmpID → 10024
23. Employee Kinsella, Kathleen → has Department → Production
24. Employee Smith, Sade → has State → MA
25. Employee Dolan, Linda → has Position → IT Support
26. Employee Soto, Julia → has GenderID → 0
27. Employee Foreman, Tanya → has State → MA
28. Employee Sutwell, Barbara → has ManagerID → 16.0
29. Employee Soto, Julia → has RecruitmentSource → LinkedIn
30. Employee Leruth, Giovanni → has Salary → 70468

**Result**: ✓ Correct (matches ground truth)

---

## Query 24
**Query**: Give me facts about the employee with the lowest performance

**Response** (0.00s):
```
Could not determine employee with lowest performance.
```

**Evidence**: 35 facts retrieved

**Extracted Facts:**

**📄 Document Agent Facts (Direct CSV):**
1. Employee O'hare, Lynn → has Salary → 52674
2. Employee Guilianno, Mike → has ManagerName → John Smith
3. Employee Veera, Abdellah → has Sex → M
4. Employee Becker, Scott → has EmpID → 10277
5. Employee Gross, Paula → has PerformanceScore → Fully Meets
6. Employee Keatts, Kramer → has EmpID → 10192
7. Employee Stoica, Rick → has MaritalDesc → Married
8. Employee Gonzalez, Juan → has DeptID → 5
9. HRDataset v14.csv Row 35 → has Employee Name → Cady, Max
10. Employee Wallace, Courtney  E → has MarriedID → 1
11. Employee Dietrich, Jenna → has EmploymentStatus → Active
12. Employee Buccheri, Joseph → has MarriedID → 0
13. Employee Rhoads, Thomas → has Termd → 1
14. Employee Trzeciak, Cybil → has Sex → F
15. Employee Bacong, Alejandro → has DeptID → 3
16. Employee Gonzalez, Maria → has PerformanceScore → Fully Meets
17. Employee Szabo, Andrew → has GenderID → 1
18. Employee Sloan, Constance → has State → MA
19. Employee Billis, Helen → has Termd → 0
20. HRDataset v14.csv Row 10 → has Employee Name → Baczenski, Rachael
21. Employee Clukey, Elijian → has MaritalDesc → Married
22. Employee Szabo, Andrew → has EmpID → 10024
23. Employee Kinsella, Kathleen → has Department → Production
24. Employee Smith, Sade → has State → MA
25. Employee Dolan, Linda → has Position → IT Support
26. Employee Soto, Julia → has GenderID → 0
27. Employee Foreman, Tanya → has State → MA
28. Employee Sutwell, Barbara → has ManagerID → 16.0
29. Employee Soto, Julia → has RecruitmentSource → LinkedIn
30. Employee Leruth, Giovanni → has Salary → 70468

**❓ Other Facts:**
1. HRDataset v14.csv Row 113 → has LastPerformanceReview Date → 1/28/2019
2. HRDataset v14.csv Row 212 → has LastPerformanceReview Date → 8/16/2015
3. HRDataset v14.csv Row 20 → has PerformanceScore → Fully Meets
4. HRDataset v14.csv Row 284 → has LastPerformanceReview Date → 4/15/2015
5. HRDataset v14.csv Row 7 → has PerformanceScore → Fully Meets

**Result**: ✓ Correct (matches ground truth)

---

## Query 23
**Query**: Retrieve facts about the employee who has the lowest performance score

**Response** (0.00s):
```
Could not determine employee with lowest performance.
```

**Evidence**: 35 facts retrieved

**Extracted Facts:**

**📄 Document Agent Facts (Direct CSV):**
1. Employee O'hare, Lynn → has Salary → 52674
2. Employee Guilianno, Mike → has ManagerName → John Smith
3. Employee Veera, Abdellah → has Sex → M
4. Employee Becker, Scott → has EmpID → 10277
5. Employee Gross, Paula → has PerformanceScore → Fully Meets
6. Employee Keatts, Kramer → has EmpID → 10192
7. Employee Stoica, Rick → has MaritalDesc → Married
8. Employee Gonzalez, Juan → has DeptID → 5
9. HRDataset v14.csv Row 35 → has Employee Name → Cady, Max
10. Employee Wallace, Courtney  E → has MarriedID → 1
11. Employee Dietrich, Jenna → has EmploymentStatus → Active
12. Employee Buccheri, Joseph → has MarriedID → 0
13. Employee Rhoads, Thomas → has Termd → 1
14. Employee Trzeciak, Cybil → has Sex → F
15. Employee Bacong, Alejandro → has DeptID → 3
16. Employee Gonzalez, Maria → has PerformanceScore → Fully Meets
17. Employee Szabo, Andrew → has GenderID → 1
18. Employee Sloan, Constance → has State → MA
19. Employee Billis, Helen → has Termd → 0
20. HRDataset v14.csv Row 10 → has Employee Name → Baczenski, Rachael
21. Employee Clukey, Elijian → has MaritalDesc → Married
22. Employee Szabo, Andrew → has EmpID → 10024
23. Employee Kinsella, Kathleen → has Department → Production
24. Employee Smith, Sade → has State → MA
25. Employee Dolan, Linda → has Position → IT Support
26. Employee Soto, Julia → has GenderID → 0
27. Employee Foreman, Tanya → has State → MA
28. Employee Sutwell, Barbara → has ManagerID → 16.0
29. Employee Soto, Julia → has RecruitmentSource → LinkedIn
30. Employee Leruth, Giovanni → has Salary → 70468

**❓ Other Facts:**
1. HRDataset v14.csv Row 113 → has LastPerformanceReview Date → 1/28/2019
2. HRDataset v14.csv Row 212 → has LastPerformanceReview Date → 8/16/2015
3. HRDataset v14.csv Row 20 → has PerformanceScore → Fully Meets
4. HRDataset v14.csv Row 284 → has LastPerformanceReview Date → 4/15/2015
5. HRDataset v14.csv Row 7 → has PerformanceScore → Fully Meets

**Result**: ✓ Correct (matches ground truth)

---

## Query 24
**Query**: Show me information about the employee with worst performance

**Response** (0.00s):
```
Could not determine employee with lowest performance.
```

**Evidence**: 35 facts retrieved

**Extracted Facts:**

**📄 Document Agent Facts (Direct CSV):**
1. Employee O'hare, Lynn → has Salary → 52674
2. Employee Guilianno, Mike → has ManagerName → John Smith
3. Employee Veera, Abdellah → has Sex → M
4. Employee Becker, Scott → has EmpID → 10277
5. Employee Gross, Paula → has PerformanceScore → Fully Meets
6. Employee Keatts, Kramer → has EmpID → 10192
7. Employee Stoica, Rick → has MaritalDesc → Married
8. Employee Gonzalez, Juan → has DeptID → 5
9. HRDataset v14.csv Row 35 → has Employee Name → Cady, Max
10. Employee Wallace, Courtney  E → has MarriedID → 1
11. Employee Dietrich, Jenna → has EmploymentStatus → Active
12. Employee Buccheri, Joseph → has MarriedID → 0
13. Employee Rhoads, Thomas → has Termd → 1
14. Employee Trzeciak, Cybil → has Sex → F
15. Employee Bacong, Alejandro → has DeptID → 3
16. Employee Gonzalez, Maria → has PerformanceScore → Fully Meets
17. Employee Szabo, Andrew → has GenderID → 1
18. Employee Sloan, Constance → has State → MA
19. Employee Billis, Helen → has Termd → 0
20. HRDataset v14.csv Row 10 → has Employee Name → Baczenski, Rachael
21. Employee Clukey, Elijian → has MaritalDesc → Married
22. Employee Szabo, Andrew → has EmpID → 10024
23. Employee Kinsella, Kathleen → has Department → Production
24. Employee Smith, Sade → has State → MA
25. Employee Dolan, Linda → has Position → IT Support
26. Employee Soto, Julia → has GenderID → 0
27. Employee Foreman, Tanya → has State → MA
28. Employee Sutwell, Barbara → has ManagerID → 16.0
29. Employee Soto, Julia → has RecruitmentSource → LinkedIn
30. Employee Leruth, Giovanni → has Salary → 70468

**❓ Other Facts:**
1. HRDataset v14.csv Row 113 → has LastPerformanceReview Date → 1/28/2019
2. HRDataset v14.csv Row 212 → has LastPerformanceReview Date → 8/16/2015
3. HRDataset v14.csv Row 20 → has PerformanceScore → Fully Meets
4. HRDataset v14.csv Row 284 → has LastPerformanceReview Date → 4/15/2015
5. HRDataset v14.csv Row 7 → has PerformanceScore → Fully Meets

**Result**: ✓ Correct (matches ground truth)

---
