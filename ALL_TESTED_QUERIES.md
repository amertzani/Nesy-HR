# All Tested Queries - Complete List

**Total Queries**: 46  
**Source**: Offline Evaluation Report (Generated: 2025-12-13 20:47:41)  
**Dataset**: HRDataset_v14.csv (311 rows, 36 columns)

---

## Operational Queries (Performance by Department)

1. What is the distribution of performance scores by department?
2. How do performance scores vary across departments?
3. Which department has the highest average performance score?
4. Show me performance metrics by department

---

## Operational Queries (Special Projects by Department)

5. What is the average special projects count by department?
6. How do special projects vary across departments?
7. Which department has the highest average special projects count?
8. Show me special projects distribution by department

---

## Operational Queries (Engagement by Manager)

9. What is the team-level engagement by manager?
10. How does engagement vary by manager?
11. Which manager has the highest team engagement?

---

## Operational Queries (Salary by Department)

12. What is the average salary by department?
13. How does salary distribution vary across departments?
14. Which department has the highest average salary?

---

## Operational Queries (Performance by Recruitment Source)

15. How does performance vary by recruitment source?
16. Which recruitment sources have the employees with highest performance score?
17. What is the performance distribution by recruitment source?

---

## Strategic Queries (Multi-Criteria Employee Search)

18. Identify employees with high performance, low engagement and many special projects
19. Find employees with high performance, low engagement and low satisfaction
20. Find employees with high performance, low engagement and low satisfaction and many special projects
21. Find employees with low engagement and low satisfaction and many special projects and many absences

---

## Strategic Queries (Department Analysis)

22. Which departments have high salaries but low performance?
23. Analyze the relationship between salary, performance, and department
24. Identify departments with low salary and high performance

---

## Evidence Retrieval Queries (Employee-Specific)

25. Retrieve facts related with employee Becker, Scott
26. Show me all facts about employee Becker, Scott
27. What information do we have about employee Becker, Scott?

---

## Evidence Retrieval Queries (Highest/Lowest Employee)

28. Give me facts about the employee with the highest salary
29. Retrieve facts about the employee who has the highest salary
30. Show me information about the highest paid employee
31. Give me facts about the employee with the lowest performance
32. Retrieve facts about the employee who has the lowest performance score
33. Show me information about the employee with worst performance

---

## Evidence Retrieval Queries (Department Facts)

34. Show me facts about employees in IT/IS department
41. What facts are stored about IT/IS department?
42. Show me all facts related to Production department
43. Retrieve facts about Sales department

---

## Evidence Retrieval Queries (Keyword-Based)

35. What facts are available about salary information?
36. Retrieve facts related to performance scores
37. Find facts about engagement by manager
38. Search for facts containing 'department' and 'salary'
39. Find facts about 'performance' and 'manager'
40. Retrieve facts with keywords 'engagement' and 'team'

---

## Evidence Retrieval Queries (Employee-Specific - Additional)

44. Show me all facts about employee Barbossa, Hector
45. What facts are stored about employee Becker, Scott?
46. Retrieve all information about employee Bacong, Alejandro

---

## Query Categories Summary

| Category | Count | Query IDs |
|----------|-------|-----------|
| **Operational - Performance by Department** | 4 | 1-4 |
| **Operational - Special Projects by Department** | 4 | 5-8 |
| **Operational - Engagement by Manager** | 3 | 9-11 |
| **Operational - Salary by Department** | 3 | 12-14 |
| **Operational - Performance by Recruitment** | 3 | 15-17 |
| **Strategic - Multi-Criteria Search** | 4 | 18-21 |
| **Strategic - Department Analysis** | 3 | 22-24 |
| **Evidence - Employee-Specific** | 3 | 25-27 |
| **Evidence - Highest/Lowest Employee** | 6 | 28-33 |
| **Evidence - Department Facts** | 4 | 34, 41-43 |
| **Evidence - Keyword-Based** | 6 | 35-40 |
| **Evidence - Employee-Specific (Additional)** | 3 | 44-46 |
| **TOTAL** | **46** | **1-46** |

---

## Results Summary

- **Correct Answers**: 39/46 (84.8%)
- **Average Response Time**: 7.51s
- **Total Evidence Facts**: 358
- **Average Evidence per Query**: 7.8
- **Evidence Retrieval Queries**: 22
- **Queries with Evidence**: 22/22 (100.0%)

---

## Notes

- Queries 1-24 are primarily **operational** and **strategic** queries that compute aggregations
- Queries 25-46 are **evidence retrieval** queries that search the knowledge graph
- Many operational queries (1-24) don't retrieve evidence because they compute directly from the dataset
- Evidence queries (25-46) successfully retrieve facts from the knowledge graph

