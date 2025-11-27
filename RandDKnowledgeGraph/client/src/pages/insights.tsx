import { useQuery } from "@tanstack/react-query";
import { hfApi } from "@/lib/api-client";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader2, TrendingUp, AlertTriangle, Target, Users, BarChart3, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useState, useEffect } from "react";
import React from "react";
import ReactMarkdown from "react-markdown";

interface StrategicInsight {
  type: string;
  category: string;
  title: string;
  description: string;
  recommendation?: string;
  priority: "high" | "medium" | "low";
  evidence?: any;
  query: string;
  id: string;
  icon: any;
}

export default function InsightsPage() {
  const { toast } = useToast();
  const [selectedQuery, setSelectedQuery] = useState<string | null>(null);
  const [queryResult, setQueryResult] = useState<Record<string, string>>({});
  const [isLoadingQuery, setIsLoadingQuery] = useState<Record<string, boolean>>({});
  const [hasAutoRun, setHasAutoRun] = useState(false);

  // Strategic Queries (S1, S2)
  const strategicQueries: StrategicInsight[] = [
    {
      id: "s1_1",
      title: "Early-Warning Risk Clusters",
      description: "Identify employee segments with high performance (PerformanceScore/PerfScoreID), low engagement (EngagementSurvey), and elevated termination rates (EmploymentStatus)",
      query: "Identify employee segments that show high performance scores (PerfScoreID or PerformanceScore), low engagement levels (EngagementSurvey), and higher termination rates (EmploymentStatus). Summarise these clusters by Department (Department), Role (Position), and Tenure band (DateofHire).",
      category: "Risk Management",
      icon: AlertTriangle,
      priority: "high",
      type: "strategic"
    },
    {
      id: "s1_2",
      title: "Active High Performers at Risk",
      description: "Detect active employees (EmploymentStatus='Active') with high performance (PerfScoreID/PerformanceScore) but declining engagement (EngagementSurvey)",
      query: "Among active employees (EmploymentStatus = 'Active'), detect groups with high performance scores (PerfScoreID or PerformanceScore) but declining engagement scores over time (EngagementSurvey). Flag these groups as early-warning retention risks and describe their common attributes, including Department (Department), Manager (ManagerName), Role (Position), and Workload (SpecialProjectsCount).",
      category: "Retention",
      icon: Users,
      priority: "high",
      type: "strategic"
    },
    {
      id: "s2_1",
      title: "Strategic Ranking of Recruitment Channels",
      description: "Rank recruitment sources (RecruitmentSource) by performance (PerfScoreID/PerformanceScore) and retention (EmploymentStatus)",
      query: "For each recruitment source (RecruitmentSource), compute the joint distribution of performance scores (PerfScoreID or PerformanceScore) and employment outcomes (EmploymentStatus). Rank recruitment channels by how well they produce high-performing employees who remain active.",
      category: "Recruitment",
      icon: Target,
      priority: "medium",
      type: "strategic"
    },
    {
      id: "s2_2",
      title: "Underperforming Recruitment Sources",
      description: "Identify recruitment sources (RecruitmentSource) with low performance (PerfScoreID/PerformanceScore) and early turnover (EmploymentStatus)",
      query: "Identify recruitment sources (RecruitmentSource) that are associated with low performance (PerfScoreID or PerformanceScore) and early turnover (EmploymentStatus with termination categories). Provide quantitative evidence and recommendations for adjusting sourcing strategy.",
      category: "Recruitment",
      icon: AlertTriangle,
      priority: "medium",
      type: "strategic"
    },
  ];

  // Operational Queries (O1, O2, O3)
  const operationalQueries: StrategicInsight[] = [
    {
      id: "o1_1",
      title: "Departmental Performance Monitoring",
      description: "Monitor performance scores (PerfScoreID/PerformanceScore) by department (Department)",
      query: "For each department (Department), monitor the distribution of performance scores (PerfScoreID or PerformanceScore) over time. Highlight departments whose performance trends decline or fall below organisation-wide averages.",
      category: "Performance",
      icon: BarChart3,
      priority: "high",
      type: "operational"
    },
    {
      id: "o1_2",
      title: "Low-Performance Concentration Tracking",
      description: "Track proportion of employees with low performance (PerformanceScore) by department (Department)",
      query: "Track the proportion of employees rated as 'Needs Improvement' or on PIP (PerformanceScore) within each department (Department). Trigger alerts when these exceed predefined thresholds.",
      category: "Performance",
      icon: AlertTriangle,
      priority: "high",
      type: "operational"
    },
    {
      id: "o2_1",
      title: "Absence Patterns by Employment Status",
      description: "Compare absences (Absences) between active vs terminated employees (EmploymentStatus)",
      query: "Compare average and distributional absence levels (Absences) between active vs. terminated employees (EmploymentStatus), broken down by Department (Department) and Role (Position). Identify absence patterns statistically associated with future termination.",
      category: "Attendance",
      icon: TrendingUp,
      priority: "medium",
      type: "operational"
    },
    {
      id: "o3_1",
      title: "Team-Level Engagement Monitoring",
      description: "Analyze engagement scores (EngagementSurvey) by manager (ManagerName)",
      query: "For each manager (ManagerName), analyse the distribution and trend of engagement scores (EngagementSurvey) within their team. Identify managers whose teams show persistently low or declining engagement and prioritise these cases for intervention.",
      category: "Engagement",
      icon: Users,
      priority: "high",
      type: "operational"
    },
  ];

  // Load operational insights from API first, then auto-run if not available
  useEffect(() => {
    if (hasAutoRun) return;
    
    const loadAndRunOperationalQueries = async () => {
      setHasAutoRun(true);
      
      // First, try to load insights from API (generated during upload)
      try {
        console.log("🔄 Loading operational insights from API...");
        const insightsResponse = await hfApi.getOperationalInsights();
        
        if (insightsResponse.success && insightsResponse.data?.insights) {
          const insights = insightsResponse.data.insights;
          let loadedCount = 0;
          
          // Map API insights to query IDs
          for (const [docName, docInsights] of Object.entries(insights)) {
            if (typeof docInsights === 'object' && docInsights !== null) {
              for (const [queryId, result] of Object.entries(docInsights)) {
                if (operationalQueries.find(q => q.id === queryId)) {
                  const resultText = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
                  setQueryResult(prev => ({ ...prev, [queryId]: resultText }));
                  loadedCount++;
                  console.log(`✅ Loaded ${queryId} from API`);
                }
              }
            }
          }
          
          if (loadedCount > 0) {
            console.log(`✅ Loaded ${loadedCount} operational insights from API`);
            return; // Don't auto-run if we loaded from API
          }
        }
      } catch (error) {
        console.log("⚠️  Could not load insights from API, will auto-run queries:", error);
      }
      
      // If no insights from API, auto-run queries
      console.log("🔄 Auto-running operational queries...");
      for (const query of operationalQueries) {
        try {
          setIsLoadingQuery(prev => ({ ...prev, [query.id]: true }));
          const response = await hfApi.sendChatMessage(query.query, []);
          
          if (response.success && response.data) {
            const resultText = typeof response.data === 'string' ? response.data : JSON.stringify(response.data, null, 2);
            setQueryResult(prev => ({ ...prev, [query.id]: resultText }));
            console.log(`✅ Auto-ran ${query.id}: ${query.title}`);
          }
        } catch (error) {
          console.error(`Error auto-running ${query.id}:`, error);
        } finally {
          setIsLoadingQuery(prev => ({ ...prev, [query.id]: false }));
        }
      }
    };
    
    // Small delay to ensure page is loaded
    const timer = setTimeout(() => {
      loadAndRunOperationalQueries();
    }, 1000);
    
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasAutoRun]);

  const handleRunQuery = async (query: string, title: string, queryId: string) => {
    setIsLoadingQuery(prev => ({ ...prev, [queryId]: true }));
    setSelectedQuery(title);
    setQueryResult(prev => ({ ...prev, [queryId]: null as any }));

    try {
      const response = await hfApi.sendChatMessage(query, []);
      
      if (response.success && response.data) {
        // Handle both string and object responses
        const resultText = typeof response.data === 'string' ? response.data : JSON.stringify(response.data, null, 2);
        setQueryResult(prev => ({ ...prev, [queryId]: resultText }));
        toast({
          title: "Query completed",
          description: "Analysis completed successfully",
        });
      } else {
        const errorMsg = response.error || "Failed to get response";
        throw new Error(errorMsg);
      }
    } catch (error) {
      console.error("Query error:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      
      // Provide helpful error messages
      let userFriendlyMessage = errorMessage;
      if (errorMessage.includes("timeout") || errorMessage.includes("aborted")) {
        userFriendlyMessage = "The query took too long to process. This might happen if:\n- The dataset is very large\n- The backend is processing other requests\n\nPlease try again in a moment.";
      } else if (errorMessage.includes("couldn't find") || errorMessage.includes("CSV")) {
        userFriendlyMessage = "No data found. Please ensure:\n1. A CSV file has been uploaded\n2. The data contains the required columns (Department, PerformanceScore, EngagementSurvey, etc.)\n3. The file was processed successfully";
      }
      
      toast({
        title: "Error",
        description: errorMessage.length > 100 ? errorMessage.substring(0, 100) + "..." : errorMessage,
        variant: "destructive",
      });
      setQueryResult(prev => ({ ...prev, [queryId]: `Error: ${userFriendlyMessage}` }));
    } finally {
      setIsLoadingQuery(prev => ({ ...prev, [queryId]: false }));
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-500/10 text-red-700 dark:text-red-400 border-red-500/20";
      case "medium":
        return "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-500/20";
      case "low":
        return "bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-500/20";
      default:
        return "bg-gray-500/10 text-gray-700 dark:text-gray-400 border-gray-500/20";
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case "Risk Management":
      case "Retention":
        return AlertTriangle;
      case "Recruitment":
        return Target;
      case "Performance":
        return BarChart3;
      case "Attendance":
        return TrendingUp;
      case "Engagement":
        return Users;
      default:
        return Sparkles;
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold mb-2">Insights</h1>
        <p className="text-muted-foreground">
          Run strategic and operational queries to get actionable insights from your HR data
        </p>
      </div>

      {/* Query Results Section - Removed old single result display, now using per-query results */}

      {/* Strategic Queries Section */}
      <div>
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <Target className="h-6 w-6 text-amber-500" />
          Strategic Queries (S1, S2)
        </h2>
        <p className="text-sm text-muted-foreground mb-4">
          Multi-variable strategic analysis for high-level decision making
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {strategicQueries.map((query) => {
            const Icon = query.icon;
            const CategoryIcon = getCategoryIcon(query.category);
            const isRunning = isLoadingQuery[query.id] || false;

            return (
              <Card
                key={query.id}
                className="p-6 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Icon className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-lg">{query.title}</h3>
                      <div className="flex items-center gap-2 mt-1">
                        <CategoryIcon className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">{query.category}</span>
                      </div>
                    </div>
                  </div>
                  <Badge className={getPriorityColor(query.priority)}>
                    {query.priority}
                  </Badge>
                </div>

                <p className="text-sm text-muted-foreground mb-4">{query.description}</p>

                <Button
                  onClick={() => handleRunQuery(query.query, query.title, query.id)}
                  disabled={isLoadingQuery[query.id]}
                  className="w-full"
                  variant="outline"
                >
                  {isLoadingQuery[query.id] ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4 mr-2" />
                      {queryResult[query.id] ? "Re-run Analysis" : "Run Analysis"}
                    </>
                  )}
                </Button>
                
                {/* Display results if available */}
                {queryResult[query.id] && (
                  <div className="mt-4 p-4 bg-muted rounded-lg prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown>
                      {queryResult[query.id]}
                    </ReactMarkdown>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      </div>

      {/* Operational Queries Section */}
      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-blue-500" />
          Operational Queries (O1, O2, O3)
        </h2>
        <p className="text-sm text-muted-foreground mb-4">
          Day-to-day operational monitoring and tracking queries
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {operationalQueries.map((query) => {
            const Icon = query.icon;
            const CategoryIcon = getCategoryIcon(query.category);
            const isRunning = isLoadingQuery[query.id] || false;

            return (
              <Card
                key={query.id}
                className="p-6 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Icon className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-lg">{query.title}</h3>
                      <div className="flex items-center gap-2 mt-1">
                        <CategoryIcon className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">{query.category}</span>
                      </div>
                    </div>
                  </div>
                  <Badge className={getPriorityColor(query.priority)}>
                    {query.priority}
                  </Badge>
                </div>

                <p className="text-sm text-muted-foreground mb-4">{query.description}</p>

                <Button
                  onClick={() => handleRunQuery(query.query, query.title, query.id)}
                  disabled={isLoadingQuery[query.id]}
                  className="w-full"
                  variant="outline"
                >
                  {isLoadingQuery[query.id] ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4 mr-2" />
                      {queryResult[query.id] ? "Re-run Analysis" : "Run Analysis"}
                    </>
                  )}
                </Button>
                
                {/* Display results if available */}
                {queryResult[query.id] && (
                  <div className="mt-4 p-4 bg-muted rounded-lg prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown>
                      {queryResult[query.id]}
                    </ReactMarkdown>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}
