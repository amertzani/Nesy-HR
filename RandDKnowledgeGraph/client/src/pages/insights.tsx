import { useQuery } from "@tanstack/react-query";
import { hfApi } from "@/lib/api-client";
import { Card } from "@/components/ui/card";
import { Loader2, Building2, Users, TrendingUp, AlertCircle, Briefcase } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface OperationalInsights {
  by_department?: Array<{
    department: string;
    employee_count: number;
    avg_performance_score?: number;
    avg_absences?: number;
    avg_salary?: number;
    avg_engagement?: number;
    avg_satisfaction?: number;
  }>;
  by_manager?: Array<{
    manager: string;
    employee_count: number;
    avg_performance_score?: number;
    avg_satisfaction?: number;
    avg_engagement?: number;
    avg_absences?: number;
    avg_salary?: number;
    total_salary?: number;
  }>;
  top_performance?: Array<{
    employee_name: string;
    performance_score: number;
    rank: number;
    department?: string;
    manager?: string;
    salary?: number;
    absences?: number;
  }>;
  top_absences?: Array<{
    employee_name: string;
    absences: number;
    rank: number;
    department?: string;
    position?: string;
    manager?: string;
    salary?: number;
    performance_score?: number;
    engagement_score?: number;
    satisfaction_score?: number;
    days_late_last30?: number;
  }>;
  bottom_engagement?: Array<{
    employee_name: string;
    engagement_score: number;
    rank: number;
    department?: string;
    position?: string;
    manager?: string;
    salary?: number;
    absences?: number;
    performance_score?: number;
    satisfaction_score?: number;
  }>;
  top_special_projects?: Array<{
    employee_name: string;
    special_projects_count: number;
    rank: number;
    department?: string;
    manager?: string;
    salary?: number;
    performance_score?: number;
    engagement_score?: number;
    satisfaction_score?: number;
  }>;
  top_salary?: Array<{
    employee_name: string;
    salary: number;
    rank: number;
    department?: string;
    manager?: string;
    performance_score?: number;
    engagement_score?: number;
    satisfaction_score?: number;
  }>;
  by_recruitment_source?: Array<{
    recruitment_source: string;
    employee_count: number;
    avg_performance_score?: number;
    avg_salary?: number;
    avg_absences?: number;
    active_employees?: number;
    active_percentage?: number;
  }>;
  additional?: {
    by_position?: Array<{
      position: string;
      employee_count: number;
      avg_performance_score?: number;
    }>;
    by_employment_status?: Array<{
      employment_status: string;
      employee_count: number;
      avg_performance_score?: number;
    }>;
  };
}

// Helper function to get max/min value for a column
const getColumnExtreme = (data: any[], key: string, isMin: boolean = false): number | null => {
  const values = data
    .map(item => {
      const val = item[key];
      return typeof val === 'number' ? val : null;
    })
    .filter((v): v is number => v !== null);
  
  if (values.length === 0) return null;
  return isMin ? Math.min(...values) : Math.max(...values);
};

// Helper function to check if value should be bold
const shouldBold = (value: any, extreme: number | null, isMin: boolean = false): boolean => {
  if (extreme === null || typeof value !== 'number') return false;
  return isMin ? value === extreme : value === extreme;
};

export default function InsightsPage() {
  const { toast } = useToast();

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["operational-insights"],
    queryFn: async () => {
      try {
        const response = await hfApi.getOperationalInsights();
        console.log("Operational insights response:", response);
        
        if (!response.success) {
          throw new Error(response.error || "Failed to load operational insights");
        }
        
        if (!response.data) {
          throw new Error("No data received from server");
        }
        
        // Handle both response.data.insights and response.data.data.insights (for backward compatibility)
        const insights = response.data?.insights || response.data?.data?.insights || response.data;
        
        console.log("Operational insights response data:", response.data);
        console.log("Insights data received:", insights);
        console.log("Manager data:", insights?.by_manager);
        console.log("Recruitment source data:", insights?.by_recruitment_source);
        
        // Check if insights are still processing
        const processingStatus = response.data?.processing_status || response.data?.data?.processing_status;
        if (processingStatus === 'processing' || processingStatus === 'pending') {
          console.log("Operational insights are still being processed");
          return { _processing: true } as any;
        }
        
        // Return empty object if no insights (frontend will show "No insights available" message)
        if (!insights || (typeof insights === 'object' && Object.keys(insights).length === 0)) {
          console.warn("No operational insights found in response");
          return {} as OperationalInsights;
        }
        
        return insights as OperationalInsights;
      } catch (err) {
        console.error("Error loading operational insights:", err);
        throw err;
      }
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-muted-foreground">Loading operational insights...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-semibold mb-2">Operational Insights</h1>
          <p className="text-muted-foreground">
            Pre-computed aggregations and statistics from your data
          </p>
        </div>
        <Card className="p-6">
          <div className="text-center text-destructive">
            <AlertCircle className="h-8 w-8 mx-auto mb-2" />
            <p>Error loading insights: {error instanceof Error ? error.message : "Unknown error"}</p>
            <button
              onClick={() => refetch()}
              className="mt-4 text-sm text-primary hover:underline"
            >
              Try again
            </button>
          </div>
        </Card>
      </div>
    );
  }

  // Check if data is empty or still processing
  const isProcessing = (data as any)?._processing === true;
  const isEmpty = !data || Object.keys(data).filter(k => k !== '_processing').length === 0;
  
  if (isEmpty || isProcessing) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-semibold mb-2">Operational Insights</h1>
          <p className="text-muted-foreground">
            Pre-computed aggregations and statistics from your data
          </p>
        </div>
        <Card className="p-6">
          <div className="text-center text-muted-foreground">
            {isProcessing ? (
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <p>Operational insights are being computed in the background. Please check again in a few moments.</p>
                <button
                  onClick={() => refetch()}
                  className="mt-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
                >
                  Refresh
                </button>
              </div>
            ) : (
            <p>No operational insights available. Please upload a CSV file to generate insights.</p>
            )}
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
      <div>
          <h1 className="text-3xl font-semibold mb-2">Operational Insights</h1>
        <p className="text-muted-foreground">
            Pre-computed aggregations and statistics from your data
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="text-sm text-primary hover:underline"
        >
          Refresh
        </button>
      </div>

      {/* Manager-based Insights Section */}
      <div>
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <Users className="h-6 w-6 text-primary" />
          Manager-Based Insights
        </h2>

        {/* Show message if no manager data */}
        {(!data.by_manager || data.by_manager.length === 0) && (
          <Card className="p-6">
            <div className="text-center text-muted-foreground">
              <p>No manager data available. Please ensure your CSV file contains a manager column (ManagerName, ManagerID, or Manager).</p>
            </div>
          </Card>
        )}

        {/* Enhanced Manager Analysis - Single Consolidated Table */}
        {data.by_manager && data.by_manager.length > 0 && (() => {
          const maxTeamSize = getColumnExtreme(data.by_manager, 'employee_count');
          const maxPerformance = getColumnExtreme(data.by_manager, 'avg_performance_score');
          const maxSatisfaction = getColumnExtreme(data.by_manager, 'avg_satisfaction');
          const maxEngagement = getColumnExtreme(data.by_manager, 'avg_engagement');
          const minAbsences = getColumnExtreme(data.by_manager, 'avg_absences', true);
          const maxAvgSalary = getColumnExtreme(data.by_manager, 'avg_salary');
          const maxTotalSalary = getColumnExtreme(data.by_manager, 'total_salary');
          
          return (
            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Users className="h-5 w-5 text-primary" />
                <h2 className="text-xl font-semibold">Manager Analysis</h2>
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Manager</TableHead>
                      <TableHead>Team Size</TableHead>
                      <TableHead>Avg Performance</TableHead>
                      <TableHead>Avg Satisfaction</TableHead>
                      <TableHead>Avg Engagement</TableHead>
                      <TableHead>Avg Absences</TableHead>
                      <TableHead>Avg Salary</TableHead>
                      <TableHead>Total Salary</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.by_manager.map((mgr, idx) => (
                      <TableRow key={idx}>
                        <TableCell className="font-medium">{mgr.manager}</TableCell>
                        <TableCell className={shouldBold(mgr.employee_count, maxTeamSize) ? "font-bold" : ""}>
                          {mgr.employee_count}
                        </TableCell>
                        <TableCell className={shouldBold(mgr.avg_performance_score, maxPerformance) ? "font-bold" : ""}>
                          {mgr.avg_performance_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(mgr.avg_satisfaction, maxSatisfaction) ? "font-bold" : ""}>
                          {mgr.avg_satisfaction?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(mgr.avg_engagement, maxEngagement) ? "font-bold" : ""}>
                          {mgr.avg_engagement?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(mgr.avg_absences, minAbsences, true) ? "font-bold" : ""}>
                          {mgr.avg_absences?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(mgr.avg_salary, maxAvgSalary) ? "font-bold" : ""}>
                          {mgr.avg_salary ? `$${mgr.avg_salary.toFixed(2)}` : "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(mgr.total_salary, maxTotalSalary) ? "font-bold" : ""}>
                          {mgr.total_salary ? `$${mgr.total_salary.toFixed(2)}` : "N/A"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>
          );
        })()}
      </div>

      {/* Department Analysis Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <Building2 className="h-6 w-6 text-primary" />
          Department Analysis
        </h2>
        {data.by_department && data.by_department.length > 0 && (() => {
          const maxEmployees = getColumnExtreme(data.by_department, 'employee_count');
          const maxPerformance = getColumnExtreme(data.by_department, 'avg_performance_score');
          const minAbsences = getColumnExtreme(data.by_department, 'avg_absences', true);
          const maxSalary = getColumnExtreme(data.by_department, 'avg_salary');
          const maxEngagement = getColumnExtreme(data.by_department, 'avg_engagement');
          const maxSatisfaction = getColumnExtreme(data.by_department, 'avg_satisfaction');
          
          return (
            <Card className="p-6">
              <div className="mb-4">
                <p className="text-sm text-muted-foreground">
                  Showing all {data.by_department.length} departments
                </p>
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Department</TableHead>
                      <TableHead>Employees</TableHead>
                      <TableHead>Avg Performance</TableHead>
                      <TableHead>Avg Absences</TableHead>
                      <TableHead>Avg Salary</TableHead>
                      <TableHead>Avg Engagement</TableHead>
                      <TableHead>Avg Satisfaction</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.by_department.map((dept, idx) => (
                      <TableRow key={idx}>
                        <TableCell className="font-medium">{dept.department}</TableCell>
                        <TableCell className={shouldBold(dept.employee_count, maxEmployees) ? "font-bold" : ""}>
                          {dept.employee_count}
                        </TableCell>
                        <TableCell className={shouldBold(dept.avg_performance_score, maxPerformance) ? "font-bold" : ""}>
                          {dept.avg_performance_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(dept.avg_absences, minAbsences, true) ? "font-bold" : ""}>
                          {dept.avg_absences?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(dept.avg_salary, maxSalary) ? "font-bold" : ""}>
                          {dept.avg_salary ? `$${dept.avg_salary.toFixed(2)}` : "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(dept.avg_engagement, maxEngagement) ? "font-bold" : ""}>
                          {dept.avg_engagement?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(dept.avg_satisfaction, maxSatisfaction) ? "font-bold" : ""}>
                          {dept.avg_satisfaction?.toFixed(2) ?? "N/A"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>
          );
        })()}
      </div>

      {/* Recruitment Source Analysis Section - Independent */}
      {data.by_recruitment_source && data.by_recruitment_source.length > 0 && (() => {
        const maxHires = getColumnExtreme(data.by_recruitment_source, 'employee_count');
        const maxActiveEmployees = getColumnExtreme(data.by_recruitment_source, 'active_employees');
        const maxActivePercentage = getColumnExtreme(data.by_recruitment_source, 'active_percentage');
        const maxPerformance = getColumnExtreme(data.by_recruitment_source, 'avg_performance_score');
        const maxSalary = getColumnExtreme(data.by_recruitment_source, 'avg_salary');
        const minAbsences = getColumnExtreme(data.by_recruitment_source, 'avg_absences', true);
        
        return (
          <div className="mt-8">
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <Briefcase className="h-6 w-6 text-primary" />
              Recruitment Source Analysis
            </h2>
            <Card className="p-6">
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Recruitment Source</TableHead>
                      <TableHead>Hires</TableHead>
                      <TableHead>Active Employees</TableHead>
                      <TableHead>Active %</TableHead>
                      <TableHead>Avg Performance</TableHead>
                      <TableHead>Avg Salary</TableHead>
                      <TableHead>Avg Absences</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.by_recruitment_source.map((source, idx) => (
                      <TableRow key={idx}>
                        <TableCell className="font-medium">{source.recruitment_source}</TableCell>
                        <TableCell className={shouldBold(source.employee_count, maxHires) ? "font-bold" : ""}>
                          {source.employee_count}
                        </TableCell>
                        <TableCell className={shouldBold(source.active_employees, maxActiveEmployees) ? "font-bold" : ""}>
                          {source.active_employees ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(source.active_percentage, maxActivePercentage) ? "font-bold" : ""}>
                          {source.active_percentage !== undefined && source.active_percentage !== null
                            ? `${source.active_percentage.toFixed(2)}%`
                            : "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(source.avg_performance_score, maxPerformance) ? "font-bold" : ""}>
                          {source.avg_performance_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(source.avg_salary, maxSalary) ? "font-bold" : ""}>
                          {source.avg_salary ? `$${source.avg_salary.toFixed(2)}` : "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(source.avg_absences, minAbsences, true) ? "font-bold" : ""}>
                          {source.avg_absences?.toFixed(2) ?? "N/A"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>
          </div>
        );
      })()}

      {/* Employee Analysis Section (Active Employees) */}
      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="h-6 w-6 text-primary" />
          Employee Analysis (Active Employees)
        </h2>

        {/* Top Performance */}
        {data.top_performance && data.top_performance.length > 0 && (() => {
          const maxPerformance = getColumnExtreme(data.top_performance, 'performance_score');
          const maxSalary = getColumnExtreme(data.top_performance, 'salary');
          const minAbsences = getColumnExtreme(data.top_performance, 'absences', true);
          
          return (
            <Card className="p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="h-5 w-5 text-primary" />
                <h3 className="text-xl font-semibold">Top 5 Employees by Performance</h3>
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Rank</TableHead>
                      <TableHead>Employee Name</TableHead>
                      <TableHead>Performance Score</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Manager</TableHead>
                      <TableHead>Salary</TableHead>
                      <TableHead>Absences</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.top_performance.map((emp) => (
                      <TableRow key={emp.rank}>
                        <TableCell>{emp.rank}</TableCell>
                        <TableCell className="font-medium">{emp.employee_name}</TableCell>
                        <TableCell className={shouldBold(emp.performance_score, maxPerformance) ? "font-bold" : ""}>
                          {emp.performance_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell>{emp.department ?? "N/A"}</TableCell>
                        <TableCell>{emp.manager ?? "N/A"}</TableCell>
                        <TableCell className={shouldBold(emp.salary, maxSalary) ? "font-bold" : ""}>
                          {emp.salary ? `$${emp.salary.toFixed(2)}` : "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.absences, minAbsences, true) ? "font-bold" : ""}>
                          {emp.absences?.toFixed(0) ?? "N/A"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>
          );
        })()}

        {/* Top Special Projects */}
        {data.top_special_projects && data.top_special_projects.length > 0 && (() => {
          const maxSpecialProjects = getColumnExtreme(data.top_special_projects, 'special_projects_count');
          const maxSalary = getColumnExtreme(data.top_special_projects, 'salary');
          const maxPerformance = getColumnExtreme(data.top_special_projects, 'performance_score');
          const maxEngagement = getColumnExtreme(data.top_special_projects, 'engagement_score');
          const maxSatisfaction = getColumnExtreme(data.top_special_projects, 'satisfaction_score');
          
          return (
            <Card className="p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="h-5 w-5 text-primary" />
                <h3 className="text-xl font-semibold">Top 5 Employees by Special Projects Count</h3>
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Rank</TableHead>
                      <TableHead>Employee Name</TableHead>
                      <TableHead>Special Projects</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Manager</TableHead>
                      <TableHead>Salary</TableHead>
                      <TableHead>Performance</TableHead>
                      <TableHead>Engagement</TableHead>
                      <TableHead>Satisfaction</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.top_special_projects.map((emp) => (
                      <TableRow key={emp.rank}>
                        <TableCell>{emp.rank}</TableCell>
                        <TableCell className="font-medium">{emp.employee_name}</TableCell>
                        <TableCell className={shouldBold(emp.special_projects_count, maxSpecialProjects) ? "font-bold" : ""}>
                          {emp.special_projects_count?.toFixed(0) ?? "N/A"}
                        </TableCell>
                        <TableCell>{emp.department ?? "N/A"}</TableCell>
                        <TableCell>{emp.manager ?? "N/A"}</TableCell>
                        <TableCell className={shouldBold(emp.salary, maxSalary) ? "font-bold" : ""}>
                          {emp.salary ? `$${emp.salary.toFixed(2)}` : "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.performance_score, maxPerformance) ? "font-bold" : ""}>
                          {emp.performance_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.engagement_score, maxEngagement) ? "font-bold" : ""}>
                          {emp.engagement_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.satisfaction_score, maxSatisfaction) ? "font-bold" : ""}>
                          {emp.satisfaction_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>
          );
        })()}

        {/* Top Salary */}
        {data.top_salary && data.top_salary.length > 0 && (() => {
          const maxSalary = getColumnExtreme(data.top_salary, 'salary');
          const maxPerformance = getColumnExtreme(data.top_salary, 'performance_score');
          const maxEngagement = getColumnExtreme(data.top_salary, 'engagement_score');
          const maxSatisfaction = getColumnExtreme(data.top_salary, 'satisfaction_score');
          
          return (
            <Card className="p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="h-5 w-5 text-primary" />
                <h3 className="text-xl font-semibold">Top 5 Employees by Salary</h3>
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Rank</TableHead>
                      <TableHead>Employee Name</TableHead>
                      <TableHead>Salary</TableHead>
                      <TableHead>Performance</TableHead>
                      <TableHead>Engagement</TableHead>
                      <TableHead>Satisfaction</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Manager</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.top_salary.map((emp) => (
                      <TableRow key={emp.rank}>
                        <TableCell>{emp.rank}</TableCell>
                        <TableCell className="font-medium">{emp.employee_name}</TableCell>
                        <TableCell className={shouldBold(emp.salary, maxSalary) ? "font-bold" : ""}>
                          {emp.salary ? `$${emp.salary.toFixed(2)}` : "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.performance_score, maxPerformance) ? "font-bold" : ""}>
                          {emp.performance_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.engagement_score, maxEngagement) ? "font-bold" : ""}>
                          {emp.engagement_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.satisfaction_score, maxSatisfaction) ? "font-bold" : ""}>
                          {emp.satisfaction_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell>{emp.department ?? "N/A"}</TableCell>
                        <TableCell>{emp.manager ?? "N/A"}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>
          );
        })()}

        {/* Top Absences */}
        {data.top_absences && data.top_absences.length > 0 && (() => {
          const maxAbsences = getColumnExtreme(data.top_absences, 'absences');
          const maxPerformance = getColumnExtreme(data.top_absences, 'performance_score');
          const maxEngagement = getColumnExtreme(data.top_absences, 'engagement_score');
          const maxSatisfaction = getColumnExtreme(data.top_absences, 'satisfaction_score');
          const maxDaysLate = getColumnExtreme(data.top_absences, 'days_late_last30');
          const maxSalary = getColumnExtreme(data.top_absences, 'salary');
          
          return (
            <Card className="p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="h-5 w-5 text-primary" />
                <h3 className="text-xl font-semibold">Top 5 Employees by Absences</h3>
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Rank</TableHead>
                      <TableHead>Employee Name</TableHead>
                      <TableHead>Absences</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Position</TableHead>
                      <TableHead>Manager</TableHead>
                      <TableHead>Performance</TableHead>
                      <TableHead>Engagement</TableHead>
                      <TableHead>Satisfaction</TableHead>
                      <TableHead>Days Late (30d)</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.top_absences.map((emp) => (
                      <TableRow key={emp.rank}>
                        <TableCell>{emp.rank}</TableCell>
                        <TableCell className="font-medium">{emp.employee_name}</TableCell>
                        <TableCell className={shouldBold(emp.absences, maxAbsences) ? "font-bold" : ""}>
                          {emp.absences?.toFixed(0) ?? "N/A"}
                        </TableCell>
                        <TableCell>{emp.department ?? "N/A"}</TableCell>
                        <TableCell>{emp.position ?? "N/A"}</TableCell>
                        <TableCell>{emp.manager ?? "N/A"}</TableCell>
                        <TableCell className={shouldBold(emp.performance_score, maxPerformance) ? "font-bold" : ""}>
                          {emp.performance_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.engagement_score, maxEngagement) ? "font-bold" : ""}>
                          {emp.engagement_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.satisfaction_score, maxSatisfaction) ? "font-bold" : ""}>
                          {emp.satisfaction_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.days_late_last30, maxDaysLate) ? "font-bold" : ""}>
                          {emp.days_late_last30?.toFixed(0) ?? "N/A"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>
          );
        })()}

        {/* Bottom Engagement */}
        {data.bottom_engagement && data.bottom_engagement.length > 0 && (() => {
          const minEngagement = getColumnExtreme(data.bottom_engagement, 'engagement_score', true);
          const maxPerformance = getColumnExtreme(data.bottom_engagement, 'performance_score');
          const maxSatisfaction = getColumnExtreme(data.bottom_engagement, 'satisfaction_score');
          const maxSalary = getColumnExtreme(data.bottom_engagement, 'salary');
          const maxAbsences = getColumnExtreme(data.bottom_engagement, 'absences');
          
          return (
            <Card className="p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <AlertCircle className="h-5 w-5 text-primary" />
                <h3 className="text-xl font-semibold">Bottom 5 Employees by Engagement</h3>
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Rank</TableHead>
                      <TableHead>Employee Name</TableHead>
                      <TableHead>Engagement Score</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Manager</TableHead>
                      <TableHead>Satisfaction</TableHead>
                      <TableHead>Performance</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.bottom_engagement.map((emp) => (
                      <TableRow key={emp.rank}>
                        <TableCell>{emp.rank}</TableCell>
                        <TableCell className="font-medium">{emp.employee_name}</TableCell>
                        <TableCell className={shouldBold(emp.engagement_score, minEngagement, true) ? "font-bold" : ""}>
                          {emp.engagement_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell>{emp.department ?? "N/A"}</TableCell>
                        <TableCell>{emp.manager ?? "N/A"}</TableCell>
                        <TableCell className={shouldBold(emp.satisfaction_score, maxSatisfaction) ? "font-bold" : ""}>
                          {emp.satisfaction_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                        <TableCell className={shouldBold(emp.performance_score, maxPerformance) ? "font-bold" : ""}>
                          {emp.performance_score?.toFixed(2) ?? "N/A"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>
          );
        })()}
      </div>

      {/* Additional Insights - By Position */}
      {data.additional?.by_position && data.additional.by_position.length > 0 && (
        <div className="mt-8">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <Briefcase className="h-6 w-6 text-primary" />
            Position Analysis
          </h2>
          <Card className="p-6">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Position</TableHead>
                    <TableHead>Employees</TableHead>
                    <TableHead>Active Employees</TableHead>
                    <TableHead>Avg Performance</TableHead>
                    <TableHead>Avg Salary</TableHead>
                    <TableHead>Avg Satisfaction</TableHead>
                    <TableHead>Avg Engagement</TableHead>
                    <TableHead>Avg Absences</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.additional.by_position.map((pos, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="font-medium">{pos.position}</TableCell>
                      <TableCell>{pos.employee_count}</TableCell>
                      <TableCell>{pos.active_employees ?? "N/A"}</TableCell>
                      <TableCell>
                        {pos.avg_performance_score?.toFixed(2) ?? "N/A"}
                      </TableCell>
                      <TableCell>
                        {pos.avg_salary ? `$${pos.avg_salary.toFixed(2)}` : "N/A"}
                      </TableCell>
                      <TableCell>
                        {pos.avg_satisfaction?.toFixed(2) ?? "N/A"}
                      </TableCell>
                      <TableCell>
                        {pos.avg_engagement?.toFixed(2) ?? "N/A"}
                      </TableCell>
                      <TableCell>
                        {pos.avg_absences?.toFixed(2) ?? "N/A"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Card>
        </div>
      )}

      {/* Additional Insights - By Employment Status */}
      {data.additional?.by_employment_status && data.additional.by_employment_status.length > 0 && (
        <div className="mt-8">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <Users className="h-6 w-6 text-primary" />
            Employment Status Analysis
          </h2>
          <Card className="p-6">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Employment Status</TableHead>
                    <TableHead>Employees</TableHead>
                    <TableHead>Avg Performance</TableHead>
                    <TableHead>Avg Engagement</TableHead>
                    <TableHead>Avg Satisfaction</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.additional.by_employment_status.map((status, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="font-medium">{status.employment_status}</TableCell>
                      <TableCell>{status.employee_count}</TableCell>
                      <TableCell>
                        {status.avg_performance_score?.toFixed(2) ?? "N/A"}
                      </TableCell>
                      <TableCell>
                        {status.avg_engagement?.toFixed(2) ?? "N/A"}
                      </TableCell>
                      <TableCell>
                        {status.avg_satisfaction?.toFixed(2) ?? "N/A"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
