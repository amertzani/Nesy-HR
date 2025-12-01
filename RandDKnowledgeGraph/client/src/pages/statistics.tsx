import { useQuery } from "@tanstack/react-query";
import { hfApi } from "@/lib/api-client";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader2, BarChart3, TrendingUp, AlertCircle, FileText, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import type { Document } from "@shared/schema";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  ComposedChart,
} from "recharts";

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff7300'];

export default function StatisticsPage() {
  const { data: documentsData, isLoading: docsLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: async () => {
      const result = await hfApi.getDocuments();
      if (result.success && result.data?.documents) {
        return result.data.documents as Document[];
      }
      return [];
    },
  });

  const documents = documentsData || [];

  if (docsLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <FileText className="h-12 w-12 text-muted-foreground" />
        <p className="text-muted-foreground">No documents uploaded yet</p>
        <p className="text-sm text-muted-foreground">Upload a CSV file to see statistics</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Statistics & Visualizations</h1>
        <p className="text-muted-foreground mt-2">
          Statistical analysis and visualizations from your uploaded documents
        </p>
      </div>

      <div className="grid gap-6">
        {documents.map((doc) => (
          <DocumentStatisticsCard key={doc.id} document={doc} />
        ))}
      </div>
    </div>
  );
}

function DocumentStatisticsCard({ document }: { document: any }) {
  const documentId = document.name || document.id;
  const { toast } = useToast();

  const { data: statsData, isLoading: statsLoading } = useQuery({
    queryKey: ["document-statistics", documentId],
    queryFn: async () => {
      const response = await hfApi.getDocumentStatistics(documentId);
      if (response.success) {
        return response.data?.statistics;
      }
      return null;
    },
    enabled: !!documentId,
  });

  const { data: vizData, isLoading: vizLoading } = useQuery({
    queryKey: ["document-visualizations", documentId],
    queryFn: async () => {
      const response = await hfApi.getDocumentVisualizations(documentId);
      if (response.success) {
        return response.data?.visualizations;
      }
      return null;
    },
    enabled: !!documentId,
  });

  const { data: summaryData, isLoading: summaryLoading } = useQuery({
    queryKey: ["document-summary", documentId],
    queryFn: async () => {
      const response = await hfApi.getDocumentSummary(documentId);
      if (response.success) {
        return response.data?.summary;
      }
      return null;
    },
    enabled: !!documentId,
  });

  const isLoading = statsLoading || vizLoading;
  const hasStats = statsData !== null && statsData !== undefined;
  const hasViz = vizData !== null && vizData !== undefined;

  const handleExportCorrelationMatrix = () => {
    if (!statsData?.correlations) {
      toast({
        title: "No correlation data",
        description: "Correlation matrix is not available for this document.",
        variant: "destructive",
      });
      return;
    }

    try {
      // Get all column names
      const columns = Object.keys(statsData.correlations);
      
      // Build CSV content
      let csvContent = "Column," + columns.join(",") + "\n";
      
      // Add each row (symmetric matrix - check both directions)
      columns.forEach((col1) => {
        const row = [col1];
        columns.forEach((col2) => {
          if (col1 === col2) {
            row.push("1.000"); // Self-correlation (diagonal)
          } else {
            // Check both directions since matrix is symmetric
            let corrValue = statsData.correlations[col1]?.[col2] ?? 
                            statsData.correlations[col2]?.[col1] ?? 
                            null;
            
            // Normalize correlation value if it appears to be scaled (e.g., 1000 instead of 1.0)
            // Correlation values should be in range [-1, 1]
            if (corrValue !== null && typeof corrValue === 'number') {
              // If value is outside [-1, 1] range, it might be scaled
              // corrValue = corrValue / 1000;
              if (Math.abs(corrValue) > 1) {
                // Normalize by dividing by 1000 (e.g., 1000 -> 1.0, -33 -> -0.033, -202 -> -0.202)
                corrValue = corrValue / 1000;
                // Clamp to [-1, 1] range
                corrValue = Math.max(-1, Math.min(1, corrValue));
              }
            }
            
            row.push(corrValue !== null ? corrValue.toFixed(3) : "0.000");
          }
        });
        csvContent += row.join(",") + "\n";
      });

      // Create blob and download
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      
      // Check if we're in a browser environment
      if (typeof window === 'undefined' || !window.document) {
        throw new Error("Export is only available in browser environment");
      }
      
      const link = window.document.createElement("a");
      const url = URL.createObjectURL(blob);
      link.setAttribute("href", url);
      link.setAttribute("download", `${document.name || documentId}_correlation_matrix.csv`);
      link.style.visibility = "hidden";
      window.document.body.appendChild(link);
      link.click();
      window.document.body.removeChild(link);
      URL.revokeObjectURL(url);

      toast({
        title: "Export successful",
        description: `Correlation matrix exported for ${columns.length} columns.`,
      });
    } catch (error) {
      toast({
        title: "Export failed",
        description: error instanceof Error ? error.message : "Failed to export correlation matrix",
        variant: "destructive",
      });
    }
  };

  const handleExportStatistics = async () => {
    try {
      toast({
        title: "Exporting statistics...",
        description: "Preparing your statistics export file.",
      });

      const result = await hfApi.exportDocumentStatistics(documentId);
      
      if (result.success && result.data) {
        const exportData = result.data;
        
        // Check if we're in a browser environment
        if (typeof window === 'undefined' || !window.document) {
          throw new Error("Export is only available in browser environment");
        }
        
        // Create blob with the statistics data
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        
        // Use window.document to ensure we're accessing the browser document
        const a = window.document.createElement("a");
        a.href = url;
        const safeFileName = documentId.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        a.download = `statistics-${safeFileName}-${Date.now()}.json`;
        window.document.body.appendChild(a);
        a.click();
        window.document.body.removeChild(a);
        URL.revokeObjectURL(url);

        toast({
          title: "Export successful",
          description: `Statistics exported for ${document.name}`,
        });
      } else {
        throw new Error(result.error || "Failed to export statistics");
      }
    } catch (error) {
      console.error("Export error:", error);
      toast({
        title: "Export failed",
        description: error instanceof Error ? error.message : "Failed to export statistics",
        variant: "destructive",
      });
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-xl font-semibold">{document.name}</h2>
          <p className="text-sm text-muted-foreground mt-1">
            {document.file_type?.toUpperCase()} • {document.size ? `${(document.size / 1024).toFixed(1)} KB` : 'Unknown size'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {hasStats && (
            <>
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                <BarChart3 className="h-3 w-3 mr-1" />
                Statistics Available
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={handleExportStatistics}
                disabled={isLoading || !hasStats}
              >
                <Download className="h-4 w-4 mr-2" />
                Export Statistics
              </Button>
            </>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}

      {!isLoading && !hasStats && (
        <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
          <AlertCircle className="h-8 w-8 mb-2" />
          <p>No statistics available for this document</p>
          <p className="text-sm mt-1">Statistics are generated for CSV files</p>
        </div>
      )}

      {/* Document Summary */}
      {summaryLoading ? (
        <div className="mb-6 p-4 border rounded-lg bg-muted/50">
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Generating summary...</span>
          </div>
        </div>
      ) : summaryData ? (
        <div className="mb-6 p-4 border rounded-lg bg-blue-50/50 dark:bg-blue-950/20">
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Document Summary
          </h3>
          <p className="text-sm text-muted-foreground whitespace-pre-wrap">{summaryData}</p>
        </div>
      ) : null}

      {!isLoading && hasStats && (
        <div className="space-y-6">
          {/* Overview Cards */}
          <div className="grid grid-cols-3 gap-4">
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Total Rows</p>
              <p className="text-2xl font-bold">{statsData.total_rows || 0}</p>
            </div>
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Total Columns</p>
              <p className="text-2xl font-bold">{statsData.total_columns || 0}</p>
            </div>
            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Column Types</p>
              <p className="text-2xl font-bold">
                {statsData.column_types ? Object.keys(statsData.column_types).length : 0}
              </p>
            </div>
          </div>

          {/* Correlation Analysis */}
          {statsData.correlations && Object.keys(statsData.correlations).length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold mb-3">Correlation Analysis</h3>
              
              {/* Strong Correlations List */}
              {(() => {
                const strongCorrs = getStrongCorrelations(statsData.correlations);
                if (strongCorrs.length > 0) {
                  return (
                    <div className="mb-6">
                      <h4 className="text-md font-medium mb-3">Strong Correlations (|r| &gt; 0.5)</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {strongCorrs.slice(0, 10).map((corr, idx) => (
                          <div
                            key={idx}
                            className="p-3 border rounded-lg bg-muted/30"
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm font-medium">
                                {corr.col1} ↔ {corr.col2}
                              </span>
                              <Badge
                                variant={Math.abs(corr.value) > 0.7 ? "default" : "secondary"}
                                className="ml-2"
                              >
                                {corr.value > 0 ? "+" : ""}{corr.value.toFixed(3)}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-2 mt-2">
                              <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
                                <div
                                  className={`h-full ${
                                    corr.value > 0 ? "bg-green-500" : "bg-red-500"
                                  }`}
                                  style={{
                                    width: `${Math.abs(corr.value) * 100}%`,
                                    marginLeft: corr.value < 0 ? "auto" : "0",
                                  }}
                                />
                              </div>
                              <span className="text-xs text-muted-foreground">
                                {Math.abs(corr.value) > 0.7 ? "Strong" : "Moderate"}
                                {corr.value > 0 ? " Positive" : " Negative"}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                }
                return null;
              })()}

              {/* Correlation Heatmap Table */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-md font-medium">
                    Correlation Matrix ({Object.keys(statsData.correlations).length} columns)
                  </h4>
                  <Button
                    onClick={handleExportCorrelationMatrix}
                    variant="outline"
                    size="sm"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Export to CSV
                  </Button>
                </div>
                <div className="overflow-x-auto border rounded-lg max-h-[600px] overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-background z-20">
                      <tr className="bg-muted border-b">
                        <th className="p-2 text-left font-medium sticky left-0 bg-muted z-30 border-r">
                          Variable
                        </th>
                        {Object.keys(statsData.correlations).map((col) => (
                          <th
                            key={col}
                            className="p-2 text-center font-medium min-w-[80px]"
                            title={col}
                          >
                            {col.length > 12 ? col.substring(0, 12) + "..." : col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(statsData.correlations).map((col1, i) => (
                        <tr key={col1} className="border-t hover:bg-muted/30">
                          <td
                            className="p-2 font-medium sticky left-0 bg-background z-10 border-r"
                            title={col1}
                          >
                            {col1.length > 20 ? col1.substring(0, 20) + "..." : col1}
                          </td>
                      {Object.keys(statsData.correlations).map((col2, j) => {
                            // Get correlation value (check both directions for symmetric matrix)
                            let corrValue = statsData.correlations[col1]?.[col2] ?? 
                                            statsData.correlations[col2]?.[col1] ?? 
                                            (i === j ? 1.0 : null);
                            
                            // Normalize correlation value if it appears to be scaled (e.g., 1000 instead of 1.0)
                            if (corrValue !== null && typeof corrValue === 'number' && i !== j) {
                              // If value is outside [-1, 1] range, it might be scaled
                              if (Math.abs(corrValue) > 1) {
                                // Normalize by dividing by 1000 (e.g., 1000 -> 1.0, -33 -> -0.033, -202 -> -0.202)
                                corrValue = corrValue / 1000;
                                // Clamp to [-1, 1] range
                                corrValue = Math.max(-1, Math.min(1, corrValue));
                              }
                            }
                            
                            if (i === j) {
                              return (
                                <td key={col2} className="p-2 text-center bg-muted/50">
                                  <span className="text-muted-foreground font-bold">1.00</span>
                                </td>
                              );
                            }
                            if (corrValue === null || corrValue === undefined) {
                              return (
                                <td key={col2} className="p-2 text-center">
                                  <span className="text-muted-foreground">—</span>
                                </td>
                              );
                            }
                            const absValue = Math.abs(corrValue);
                            const intensity = Math.min(absValue * 1.2, 1); // Scale for better visibility
                            const bgColor = corrValue > 0 
                              ? `rgba(34, 197, 94, ${intensity * 0.3})` // Green for positive
                              : `rgba(239, 68, 68, ${intensity * 0.3})`; // Red for negative
                            
                            return (
                              <td
                                key={col2}
                                className="p-2 text-center relative"
                                style={{ backgroundColor: bgColor }}
                                title={`${col1} ↔ ${col2}: ${corrValue.toFixed(3)}`}
                              >
                                <span
                                  className={`font-medium ${
                                    absValue > 0.7
                                      ? "text-foreground font-bold"
                                      : absValue > 0.5
                                      ? "text-foreground"
                                      : "text-muted-foreground"
                                  }`}
                                >
                                  {corrValue.toFixed(2)}
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-green-500/30 border border-green-500/50"></div>
                    <span>Positive correlation</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-500/30 border border-red-500/50"></div>
                    <span>Negative correlation</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-muted"></div>
                    <span>No correlation (diagonal)</span>
                  </div>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Values range from -1 (perfect negative) to +1 (perfect positive). 
                  Strong correlations (|r| &gt; 0.5) are highlighted.
                </p>
                <div className="mt-3">
                  <Button
                    onClick={handleExportCorrelationMatrix}
                    variant="outline"
                    size="sm"
                    className="w-full"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Export Full Correlation Matrix to CSV
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Individual Column Statistics - One graph per column */}
          {statsData.descriptive_stats && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Column Statistics</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {Object.entries(statsData.descriptive_stats).map(([col, stats]: [string, any]) => {
                  // Categorical columns
                  if (stats.value_counts && statsData.column_types?.[col] === "categorical") {
                    const chartData = Object.entries(stats.value_counts)
                      .slice(0, 10)
                      .map(([value, count]) => ({ name: value, value: count }));
                    
                    return (
                      <div key={col} className="p-4 border rounded-lg">
                        <h4 className="font-semibold mb-3">{col} Distribution</h4>
                        <ResponsiveContainer width="100%" height={250}>
                          <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="value" fill="#8884d8" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    );
                  }
                  
                  // Numeric columns
                  if (statsData.column_types?.[col] === "numeric" && stats.mean !== null) {
                    const statData = [
                      { name: 'Mean', value: stats.mean },
                      { name: 'Median', value: stats.median },
                      { name: 'Min', value: stats.min },
                      { name: 'Max', value: stats.max },
                      { name: 'Std Dev', value: stats.std },
                    ].filter(item => item.value !== null && item.value !== undefined);

                    return (
                      <div key={col} className="p-4 border rounded-lg">
                        <h4 className="font-semibold mb-3">{col}</h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Mean:</span>{" "}
                            <span className="font-medium">{stats.mean?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Median:</span>{" "}
                            <span className="font-medium">{stats.median?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Min:</span>{" "}
                            <span className="font-medium">{stats.min?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Max:</span>{" "}
                            <span className="font-medium">{stats.max?.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Std Dev:</span>{" "}
                            <span className="font-medium">{stats.std?.toFixed(2)}</span>
                          </div>
                        </div>
                        {statData.length > 0 && (
                          <ResponsiveContainer width="100%" height={200}>
                            <BarChart data={statData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <Tooltip />
                              <Bar dataKey="value" fill="#0088FE" />
                            </BarChart>
                          </ResponsiveContainer>
                        )}
                      </div>
                    );
                  }
                  return null;
                })}
              </div>
            </div>
          )}

          {/* Column Type Distribution Chart */}
          {statsData.column_types && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Column Type Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getColumnTypeData(statsData.column_types)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#0088FE" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Missing Values Chart */}
          {statsData.missing_values && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Missing Values by Column</h3>
              {(() => {
                const missingData = getMissingValuesData(statsData.missing_values, statsData.total_rows);
                if (missingData.length === 0) {
                  return (
                    <div className="p-8 text-center border rounded-lg bg-muted/50">
                      <p className="text-muted-foreground">No missing values found in the dataset</p>
                      <p className="text-sm text-muted-foreground mt-2">All columns are complete</p>
                    </div>
                  );
                }
                return (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={missingData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="column" angle={-45} textAnchor="end" height={100} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="missing" fill="#FF8042" name="Missing Values" />
                      <Bar dataKey="complete" fill="#00C49F" name="Complete Values" />
                    </BarChart>
                  </ResponsiveContainer>
                );
              })()}
            </div>
          )}

          {/* Data Quality Summary */}
          {statsData.data_quality && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Data Quality Overview</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(statsData.data_quality).slice(0, 9).map(([col, quality]: [string, any]) => (
                  <div key={col} className="p-4 border rounded-lg">
                    <h4 className="font-medium text-sm mb-2">{col}</h4>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Completeness:</span>
                        <span className="font-medium">{(quality.completeness * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Unique Values:</span>
                        <span className="font-medium">{quality.unique_values}</span>
                      </div>
                      <div className="mt-2">
                        <div className="w-full bg-muted rounded-full h-2">
                          <div
                            className="bg-primary h-2 rounded-full"
                            style={{ width: `${quality.completeness * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

// Helper functions to transform data for charts
function getColumnTypeData(columnTypes: Record<string, string>) {
  const typeCounts: Record<string, number> = {};
  Object.values(columnTypes).forEach((type) => {
    typeCounts[type] = (typeCounts[type] || 0) + 1;
  });
  return Object.entries(typeCounts).map(([type, count]) => ({ type, count }));
}

function getMissingValuesData(missingValues: Record<string, number>, totalRows: number) {
  return Object.entries(missingValues)
    .filter(([_, missing]) => missing > 0)
    .slice(0, 15)
    .map(([column, missing]) => ({
      column: column.length > 20 ? column.substring(0, 20) + '...' : column,
      missing,
      complete: totalRows - missing,
    }))
    .sort((a, b) => b.missing - a.missing);
}

function getStrongCorrelations(correlations: Record<string, Record<string, number>>) {
  const strongCorrs: Array<{ col1: string; col2: string; value: number }> = [];
  const cols = Object.keys(correlations);
  
  cols.forEach((col1, i) => {
    cols.forEach((col2, j) => {
      if (i < j && correlations[col1]?.[col2] !== undefined) {
        const corrValue = correlations[col1][col2];
        if (Math.abs(corrValue) > 0.5) {
          strongCorrs.push({
            col1,
            col2,
            value: corrValue,
          });
        }
      }
    });
  });
  
  // Sort by absolute value descending
  strongCorrs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  
  return strongCorrs;
}
