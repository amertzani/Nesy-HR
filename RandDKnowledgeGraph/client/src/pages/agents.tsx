import { useQuery } from "@tanstack/react-query";
import { hfApi } from "@/lib/api-client";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Network, Bot, Cpu, FileText, CheckCircle2, Clock, AlertCircle, BarChart3, TrendingUp, Target } from "lucide-react";
import { AgentNetwork } from "@/components/AgentNetwork";

export default function AgentsPage() {
  const { data: architectureData, isLoading, error } = useQuery({
    queryKey: ["agent-architecture"],
    queryFn: async () => {
      const response = await hfApi.getAgentArchitecture();
      if (response.success) {
        return response.data?.architecture;
      }
      throw new Error(response.error || "Failed to load agent architecture");
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <AlertCircle className="h-8 w-8 text-destructive" />
        <p className="text-muted-foreground">Failed to load agent architecture</p>
      </div>
    );
  }

  const orchestratorAgents = architectureData?.orchestrator_agents || [];
  const statisticsAgents = architectureData?.statistics_agents || [];
  const visualizationAgents = architectureData?.visualization_agents || [];
  const kgAgents = architectureData?.kg_agents || [];
  const llmAgents = architectureData?.llm_agents || [];
  const operationalQueryAgents = architectureData?.operational_query_agents || [];
  const documentAgents = architectureData?.document_agents || [];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-green-500";
      case "processing":
        return "bg-blue-500";
      case "completed":
        return "bg-green-500";
      case "idle":
        return "bg-gray-500";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active":
      case "completed":
        return <CheckCircle2 className="h-4 w-4" />;
      case "processing":
        return <Clock className="h-4 w-4 animate-spin" />;
      default:
        return <Clock className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Agent Architecture</h1>
        <p className="text-muted-foreground">
          Multi-agent system for HR decision support
        </p>
      </div>

      {/* Network Visualization */}
      <AgentNetwork
        orchestratorAgents={orchestratorAgents}
        statisticsAgents={statisticsAgents}
        visualizationAgents={visualizationAgents}
        kgAgents={kgAgents}
        llmAgents={llmAgents}
        operationalQueryAgents={operationalQueryAgents}
        documentAgents={documentAgents}
      />

      {/* Core Agents */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Statistics Agent */}
        {statisticsAgents.map((agent: any) => (
          <Card key={agent.id} className="p-4">
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-blue-500" />
                <h3 className="font-semibold">{agent.name}</h3>
              </div>
              <Badge
                variant="outline"
                className={`${getStatusColor(agent.status)} text-white border-0`}
              >
                <span className="flex items-center gap-1">
                  {getStatusIcon(agent.status)}
                  {agent.status}
                </span>
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              {agent.metadata?.description || "No description"}
            </p>
            <div className="text-xs text-muted-foreground">
              Created: {new Date(agent.created_at).toLocaleString()}
            </div>
          </Card>
        ))}

        {/* Visualization Agent */}
        {visualizationAgents.map((agent: any) => (
          <Card key={agent.id} className="p-4">
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-green-500" />
                <h3 className="font-semibold">{agent.name}</h3>
              </div>
              <Badge
                variant="outline"
                className={`${getStatusColor(agent.status)} text-white border-0`}
              >
                <span className="flex items-center gap-1">
                  {getStatusIcon(agent.status)}
                  {agent.status}
                </span>
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              {agent.metadata?.description || "No description"}
            </p>
            <div className="text-xs text-muted-foreground">
              Created: {new Date(agent.created_at).toLocaleString()}
            </div>
          </Card>
        ))}

        {/* KG Agent */}
        {kgAgents.map((agent: any) => (
          <Card key={agent.id} className="p-4">
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <Network className="h-5 w-5 text-purple-500" />
                <h3 className="font-semibold">{agent.name}</h3>
              </div>
              <Badge
                variant="outline"
                className={`${getStatusColor(agent.status)} text-white border-0`}
              >
                <span className="flex items-center gap-1">
                  {getStatusIcon(agent.status)}
                  {agent.status}
                </span>
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              {agent.metadata?.description || "No description"}
            </p>
            <div className="text-xs text-muted-foreground">
              Created: {new Date(agent.created_at).toLocaleString()}
            </div>
          </Card>
        ))}
      </div>


      {/* Operational Query Agents */}
      {operationalQueryAgents.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Operational Query Agents ({operationalQueryAgents.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {operationalQueryAgents.map((agent: any) => (
              <Card key={agent.id} className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-blue-500" />
                    <h3 className="font-semibold">{agent.name}</h3>
                  </div>
                  <Badge
                    variant="outline"
                    className={`${getStatusColor(agent.status)} text-white border-0`}
                  >
                    <span className="flex items-center gap-1">
                      {getStatusIcon(agent.status)}
                      {agent.status}
                    </span>
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground mb-2">
                  {agent.metadata?.description || "Processes operational-level multi-variable queries"}
                </p>
                <div className="flex flex-wrap gap-2 mb-2">
                  {(agent.metadata?.capabilities || []).map((cap: string) => (
                    <Badge key={cap} variant="secondary" className="text-xs">
                      {cap}
                    </Badge>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground">
                  Created: {new Date(agent.created_at).toLocaleString()}
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* LLM Agents */}
      <div>
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Bot className="h-5 w-5" />
          LLM Agents ({llmAgents.length})
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {llmAgents.map((agent: any) => (
            <Card key={agent.id} className="p-4">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Bot className="h-5 w-5 text-purple-500" />
                  <h3 className="font-semibold">{agent.name}</h3>
                </div>
                <Badge
                  variant="outline"
                  className={`${getStatusColor(agent.status)} text-white border-0`}
                >
                  <span className="flex items-center gap-1">
                    {getStatusIcon(agent.status)}
                    {agent.status}
                  </span>
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mb-2">
                {agent.metadata?.description || "No description"}
              </p>
              <div className="text-xs text-muted-foreground mt-2">
                Created: {new Date(agent.created_at).toLocaleString()}
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Document Agents */}
      <div>
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <FileText className="h-5 w-5" />
          Document Agents ({documentAgents.filter((a: any) => a.type !== "document_worker").length})
          {documentAgents.filter((a: any) => a.type === "document_worker").length > 0 && (
            <Badge variant="outline" className="ml-2">
              {documentAgents.filter((a: any) => a.type === "document_worker").length} Workers
            </Badge>
          )}
        </h2>
        {documentAgents.filter((a: any) => a.type !== "document_worker").length === 0 ? (
          <Card className="p-8 text-center">
            <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">
              No document agents yet. Upload a document to create a document agent.
            </p>
          </Card>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {documentAgents.filter((a: any) => a.type !== "document_worker").map((agent: any) => (
                <Card key={agent.id} className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <FileText className="h-5 w-5 text-blue-500" />
                      <h3 className="font-semibold text-sm truncate">
                        {agent.document_name || agent.name}
                      </h3>
                    </div>
                    <Badge
                      variant="outline"
                      className={`${getStatusColor(agent.status)} text-white border-0`}
                    >
                      <span className="flex items-center gap-1">
                        {getStatusIcon(agent.status)}
                        {agent.status}
                      </span>
                    </Badge>
                  </div>
                <div className="space-y-1 text-xs text-muted-foreground">
                  <div>Document: {agent.document_name || "N/A"}</div>
                  <div>Facts extracted: {agent.facts_extracted || 0}</div>
                  {agent.employee_names && agent.employee_names.length > 0 && (
                    <div>Employees: {agent.employee_names.length}</div>
                  )}
                  {agent.data_range && (
                    <div>
                      {agent.data_range.chunks ? (
                        <>
                          Rows: {agent.data_range.start || 0}-{agent.data_range.end || 0} 
                          ({agent.data_range.rows || 0} total)
                          <div className="text-[10px] mt-0.5">
                            Split into {agent.data_range.chunks} chunks ({agent.data_range.chunk_size || 'N/A'} rows/chunk)
                          </div>
                        </>
                      ) : (
                        <>
                          Rows: {agent.data_range.start || 0}-{agent.data_range.end || 0} 
                          ({agent.data_range.rows || 0} total)
                        </>
                      )}
                    </div>
                  )}
                  {agent.columns_processed && agent.columns_processed.length > 0 && (
                    <div>
                      Columns: {agent.columns_processed.length}
                      {agent.data_range?.total_cols && agent.data_range.total_cols !== agent.columns_processed.length && (
                        <span className="text-[10px] text-muted-foreground ml-1">
                          (expected: {agent.data_range.total_cols})
                        </span>
                      )}
                    </div>
                  )}
                  {agent.metadata?.processed_at && (
                    <div>
                      Processed: {new Date(agent.metadata.processed_at).toLocaleString()}
                    </div>
                  )}
                </div>
                  <div className="text-xs text-muted-foreground mt-2">
                    Created: {new Date(agent.created_at).toLocaleString()}
                  </div>
                </Card>
              ))}
            </div>
            
            {/* Worker Agents Section */}
            {documentAgents.filter((a: any) => a.type === "document_worker").length > 0 && (
              <div className="mt-6 pt-4 border-t">
                <div className="flex items-center gap-2 mb-3">
                  <Cpu className="h-4 w-4" />
                  <h3 className="text-sm font-semibold">Worker Agents</h3>
                  <Badge variant="outline" className="text-xs">
                    {documentAgents.filter((a: any) => a.type === "document_worker").length}
                  </Badge>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
                  {documentAgents.filter((a: any) => a.type === "document_worker").map((agent: any) => (
                    <Card key={agent.id} className="p-2">
                      <div className="flex items-center gap-1 mb-1">
                        <div className={`w-2 h-2 rounded-full ${getStatusColor(agent.status)}`} />
                        <span className="font-medium text-xs">{agent.metadata?.chunk_range || agent.id.split('_').pop()}</span>
                      </div>
                      <div className="text-xs text-muted-foreground space-y-0.5">
                        <div>{agent.facts_extracted || 0} facts</div>
                        {agent.employee_names && agent.employee_names.length > 0 && (
                          <div>{agent.employee_names.length} employees</div>
                        )}
                        {agent.data_range && (
                          <div className="text-[10px]">{agent.data_range.rows || 0} rows</div>
                        )}
                        {agent.columns_processed && agent.columns_processed.length > 0 && (
                          <div className="text-[10px]">{agent.columns_processed.length} cols</div>
                        )}
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Summary */}
      <Card className="p-4 bg-muted">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold">{orchestratorAgents.length}</div>
            <div className="text-xs text-muted-foreground">Orchestrator</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{statisticsAgents.length}</div>
            <div className="text-xs text-muted-foreground">Statistics</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{visualizationAgents.length}</div>
            <div className="text-xs text-muted-foreground">Visualization</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{kgAgents.length}</div>
            <div className="text-xs text-muted-foreground">KG Agents</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{llmAgents.length}</div>
            <div className="text-xs text-muted-foreground">LLM Agents</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{documentAgents.filter((a: any) => a.type !== "document_worker").length}</div>
            <div className="text-xs text-muted-foreground">Documents</div>
            {documentAgents.filter((a: any) => a.type === "document_worker").length > 0 && (
              <div className="text-xs text-muted-foreground mt-1">
                ({documentAgents.filter((a: any) => a.type === "document_worker").length} workers)
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}

