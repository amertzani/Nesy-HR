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
  top_absences?: Array<{
    employee_name: string;
    absences: number;
    rank: number;
    department?: string;
    position?: string;
  }>;
  bottom_engagement?: Array<{
    employee_name: string;
    engagement_score: number;
    rank: number;
    department?: string;
    position?: string;
    manager?: string;
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
        const insights = response.data.insights || response.data.data?.insights;
        
        console.log("Insights data received:", insights);
        console.log("Manager data:", insights?.by_manager);
        console.log("Manager performance data:", insights?.manager_performance);
        
        if (!insights) {
          throw new Error("No insights data in response");
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

  if (!data || Object.keys(data).length === 0) {
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
            <p>No operational insights available. Please upload a CSV file to generate insights.</p>
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
        {data.by_manager && data.by_manager.length > 0 && (
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
                    <TableCell>{mgr.employee_count}</TableCell>
                    <TableCell>
                      {mgr.avg_performance_score?.toFixed(2) ?? "N/A"}
                    </TableCell>
                    <TableCell>
                      {mgr.avg_satisfaction?.toFixed(2) ?? "N/A"}
                    </TableCell>
                    <TableCell>
                      {mgr.avg_engagement?.toFixed(2) ?? "N/A"}
                    </TableCell>
                    <TableCell>
                      {mgr.avg_absences?.toFixed(2) ?? "N/A"}
                    </TableCell>
                    <TableCell>
                      {mgr.avg_salary ? `$${mgr.avg_salary.toFixed(2)}` : "N/A"}
                    </TableCell>
                    <TableCell>
                      {mgr.total_salary ? `$${mgr.total_salary.toFixed(2)}` : "N/A"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Card>
        )}
      </div>

      {/* Department Analysis Section - All Departments */}
      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <Building2 className="h-6 w-6 text-primary" />
          Department Analysis (All Departments)
        </h2>
        {data.by_department && data.by_department.length > 0 && (
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
                      <TableCell>{dept.employee_count}</TableCell>
                      <TableCell>
                        {dept.avg_performance_score?.toFixed(2) ?? "N/A"}
                      </TableCell>
                      <TableCell>
                        {dept.avg_absences?.toFixed(2) ?? "N/A"}
                      </TableCell>
                      <TableCell>
                        {dept.avg_salary ? `$${dept.avg_salary.toFixed(2)}` : "N/A"}
                      </TableCell>
                      <TableCell>
                        {dept.avg_engagement?.toFixed(2) ?? "N/A"}
                      </TableCell>
                      <TableCell>
                        {dept.avg_satisfaction?.toFixed(2) ?? "N/A"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Card>
        )}
      </div>

      {/* Other Insights Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="h-6 w-6 text-primary" />
          Other Insights
        </h2>

        {/* Top Absences */}
      {data.top_absences && data.top_absences.length > 0 && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Top 5 Employees by Absences</h2>
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
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.top_absences.map((emp) => (
                  <TableRow key={emp.rank}>
                    <TableCell>{emp.rank}</TableCell>
                    <TableCell className="font-medium">{emp.employee_name}</TableCell>
                    <TableCell>{emp.absences?.toFixed(0) ?? "N/A"}</TableCell>
                    <TableCell>{emp.department ?? "N/A"}</TableCell>
                    <TableCell>{emp.position ?? "N/A"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
                    </div>
        </Card>
      )}

      {/* Bottom Engagement */}
      {data.bottom_engagement && data.bottom_engagement.length > 0 && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Bottom 5 Employees by Engagement</h2>
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
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.bottom_engagement.map((emp) => (
                  <TableRow key={emp.rank}>
                    <TableCell>{emp.rank}</TableCell>
                    <TableCell className="font-medium">{emp.employee_name}</TableCell>
                    <TableCell>{emp.engagement_score?.toFixed(2) ?? "N/A"}</TableCell>
                    <TableCell>{emp.department ?? "N/A"}</TableCell>
                    <TableCell>{emp.manager ?? "N/A"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
                </div>
        </Card>
      )}

      {/* By Recruitment Source */}
      {data.by_recruitment_source && data.by_recruitment_source.length > 0 && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Briefcase className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Recruitment Source Analysis</h2>
          </div>
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
                    <TableCell>{source.employee_count}</TableCell>
                    <TableCell>
                      {source.active_employees ?? "N/A"}
                    </TableCell>
                    <TableCell>
                      {source.active_percentage !== undefined && source.active_percentage !== null
                        ? `${source.active_percentage.toFixed(2)}%`
                        : "N/A"}
                    </TableCell>
                    <TableCell>
                      {source.avg_performance_score?.toFixed(2) ?? "N/A"}
                    </TableCell>
                    <TableCell>
                      {source.avg_salary ? `$${source.avg_salary.toFixed(2)}` : "N/A"}
                    </TableCell>
                    <TableCell>
                      {source.avg_absences?.toFixed(2) ?? "N/A"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
                  </div>
        </Card>
      )}

      {/* Additional Insights - By Position */}
      {data.additional?.by_position && data.additional.by_position.length > 0 && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Briefcase className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Position Analysis</h2>
          </div>
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
      )}

      {/* Additional Insights - By Employment Status */}
      {data.additional?.by_employment_status && data.additional.by_employment_status.length > 0 && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Users className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Employment Status Analysis</h2>
          </div>
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
      )}
      </div>
    </div>
  );
}
