# All Tested Queries - Complete List

**Total queries: 30  
**Source**: test_scenarios.json  
**Dataset**: HRDataset_v14.csv (311 rows, 36 columns)

---

## Operational Queries (17 queries)

### Performance Score by Department (Queries 1-4)

1. What is the distribution of performance scores by department?
2. How do performance scores vary across departments?
3. Which department has the highest average performance score?
4. Show me performance metrics by department

---

### Special Projects Count by Department (Queries 5-8)

5. What is the average special projects count by department?
6. How do special projects vary across departments?
7. Which department has the highest average special projects count?
8. Show me special projects distribution by department

---

### Engagement by Manager (Queries 9-11)

9. What is the team-level engagement by manager?
10. How does engagement vary by manager?
11. Which manager has the highest team engagement?

---

### Salary by Department (Queries 12-14)

12. What is the average salary by department?
13. How does salary distribution vary across departments?
14. Which department has the highest average salary?

---

### Performance by Recruitment Source (Queries 15-17)

15. How does performance vary by recruitment source?
16. Which recruitment sources have the employees with highest performance score?
17. What is the performance distribution by recruitment source?

---

## Strategic Queries (7 queries)

### Multi-Criteria Employee Search (Queries 18-21)

18. Identify employees with high performance, low engagement and many special projects
19. Find employees with high performance, low engagement and low satisfaction
20. Find employees with high performance, low engagement and low satisfaction and many special projects
21. Find employees with low engagement and low satisfaction and many special projects and many absences

---

### Department-Salary-Performance Analysis (Queries 22-24)

22. Which departments have high salaries but low performance?
23. Analyze the relationship between salary, performance, and department
24. Identify departments with low salary and high performance

---

## Evidence Retrieval Queries (8 queries)

### Employee-Specific Fact Retrieval - Becker, Scott (Queries 25-26)

25. Retrieve facts related with employee Becker, Scott
26. What information do we have about employee Becker, Scott?

---

### Highest Salary Employee Facts (Queries 27-29)

27. Give me facts about the employee with the highest salary
28. Retrieve facts about the employee who has the highest salary
29. Show me information about the highest paid employee

---

### Lowest Performance Employee Facts (Queries 30-32)

30. Give me facts about the employee with the lowest performance
31. Retrieve facts about the employee who has the lowest performance score
32. Show me information about the employee with worst performance

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
| **Evidence - Employee-Specific (Becker, Scott)** | 2 | 25-26 |
| **Evidence - Highest Salary Employee** | 3 | 27-29 |
| **Evidence - Lowest Performance Employee** | 3 | 30-32 |
| **TOTAL** | **32** | **1-32** |

---

## Notes

- Queries 1-24 are **operational** and **strategic** queries that compute aggregations from the dataset
- Queries 25-32 are **evidence retrieval** queries that search the knowledge graph
- Query 24 (duplicate of Q23) was removed from the test scenarios
- All queries are defined in `test_scenarios.json` with ground truth data for evaluation
