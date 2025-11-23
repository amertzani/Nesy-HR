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
  ScatterChart,
  Scatter,
  ZAxis,
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

          {/* Correlation Matrix */}
          {statsData.correlations && Object.keys(statsData.correlations).length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Correlation Matrix</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={getCorrelationData(statsData.correlations)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" type="number" domain={[-1, 1]} />
                  <YAxis dataKey="y" type="number" domain={[-1, 1]} />
                  <ZAxis dataKey="value" range={[0, 1]} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter name="Correlations" data={getCorrelationData(statsData.correlations)} fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
              <div className="mt-4 text-sm text-muted-foreground">
                <p>Strong correlations (|r| &gt; 0.5) indicate relationships between variables</p>
              </div>
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

function getCorrelationData(correlations: Record<string, Record<string, number>>) {
  const data: Array<{ x: number; y: number; value: number; col1: string; col2: string }> = [];
  const cols = Object.keys(correlations);
  
  cols.forEach((col1, i) => {
    cols.forEach((col2, j) => {
      if (i < j && correlations[col1][col2] !== undefined) {
        data.push({
          x: i,
          y: j,
          value: Math.abs(correlations[col1][col2]),
          col1,
          col2,
        });
      }
    });
  });
  
  return data;
}
