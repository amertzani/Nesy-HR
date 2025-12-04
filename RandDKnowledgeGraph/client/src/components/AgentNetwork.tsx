import { useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BarChart3, TrendingUp, Network, Bot, FileText, Cpu, Target } from "lucide-react";

interface AgentNode {
  id: string;
  name: string;
  type: string;
  x: number;
  y: number;
  icon: any;
  color: string;
}

interface Connection {
  from: string;
  to: string;
  bidirectional?: boolean;
}

interface AgentNetworkProps {
  orchestratorAgents: any[];
  statisticsAgents: any[];
  visualizationAgents: any[];
  kgAgents: any[];
  llmAgents: any[];
  operationalQueryAgents: any[];
  documentAgents: any[];
}

export function AgentNetwork({
  orchestratorAgents,
  statisticsAgents,
  visualizationAgents,
  kgAgents,
  llmAgents,
  operationalQueryAgents,
  documentAgents,
}: AgentNetworkProps) {
  // Define node positions and connections
  const { nodes, connections } = useMemo(() => {
    const nodeMap: Map<string, AgentNode> = new Map();
    const conns: Connection[] = [];
    
    // Core agent positions (reorganized to avoid overlaps and make all visible)
    // Spread out more to show all agents clearly with names visible
    const corePositions = {
      orchestrator: { x: 400, y: 40 },        // Top center
      operational_query: { x: 400, y: 140 },  // Center, below orchestrator
      statistics: { x: 80, y: 280 },          // Far left (more space from strategic)
      visualization: { x: 720, y: 280 },      // Far right (more space from operational)
      kg: { x: 80, y: 420 },                  // Far left bottom (more space)
      llm: { x: 720, y: 420 },                // Far right bottom (more space)
    };
    
    // Add Orchestrator Agent (top center - coordinates everything)
    if (orchestratorAgents.length > 0) {
      const agent = orchestratorAgents[0];
      nodeMap.set(agent.id, {
        id: agent.id,
        name: agent.name,
        type: "orchestrator",
        ...corePositions.orchestrator,
        icon: Network,
        color: "#8b5cf6", // purple - central coordinator
      });
    }
    
    // Add Statistics Agent
    if (statisticsAgents.length > 0) {
      const agent = statisticsAgents[0];
      nodeMap.set(agent.id, {
        id: agent.id,
        name: agent.name,
        type: "statistics",
        ...corePositions.statistics,
        icon: BarChart3,
        color: "#3b82f6", // blue
      });
    }
    
    // Add Visualization Agent
    if (visualizationAgents.length > 0) {
      const agent = visualizationAgents[0];
      nodeMap.set(agent.id, {
        id: agent.id,
        name: agent.name,
        type: "visualization",
        ...corePositions.visualization,
        icon: TrendingUp,
        color: "#10b981", // green
      });
    }
    
    // Add KG Agent
    if (kgAgents.length > 0) {
      const agent = kgAgents[0];
      nodeMap.set(agent.id, {
        id: agent.id,
        name: agent.name,
        type: "kg",
        ...corePositions.kg,
        icon: Network,
        color: "#a855f7", // purple
      });
    }
    
    // Add LLM Agent
    if (llmAgents.length > 0) {
      const agent = llmAgents[0];
      nodeMap.set(agent.id, {
        id: agent.id,
        name: agent.name,
        type: "llm",
        ...corePositions.llm,
        icon: Bot,
        color: "#ec4899", // pink
      });
    }
    
    
    // Add Operational Query Agent (right of center, below orchestrator)
    if (operationalQueryAgents && operationalQueryAgents.length > 0) {
      const agent = operationalQueryAgents[0];
      nodeMap.set(agent.id, {
        id: agent.id,
        name: agent.name,
        type: "operational_query",
        ...corePositions.operational_query,
        icon: BarChart3,
        color: "#3b82f6", // blue
      });
    }
    
    // Add Document Agents (arranged in a row below)
    // Separate main document agents from worker agents
    const mainDocAgents = documentAgents.filter((a: any) => a.type !== "document_worker");
    const workerAgents = documentAgents.filter((a: any) => a.type === "document_worker");
    
    // Main document agents (arranged at bottom center, avoiding KG agent area)
    // Position them in the center-bottom area, away from left side where KG is
    mainDocAgents.forEach((agent, index) => {
      const cols = Math.min(5, Math.ceil(Math.sqrt(mainDocAgents.length)));
      const startX = 400 - (cols - 1) * 60; // Center horizontally
      const x = startX + (index % cols) * 120;
      const y = 500 + Math.floor(index / cols) * 80;
      nodeMap.set(agent.id, {
        id: agent.id,
        name: agent.document_name || agent.name,
        type: "document",
        x,
        y,
        icon: FileText,
        color: "#6366f1", // indigo
      });
    });
    
    // Worker agents (arranged below main document agents, offset to avoid KG)
    // Position workers further right to avoid overlapping with KG connections
    workerAgents.forEach((agent, index) => {
      const cols = Math.min(8, Math.ceil(Math.sqrt(workerAgents.length * 1.5)));
      const startX = 350; // Start more to the right to avoid KG area
      const x = startX + (index % cols) * 90;
      const y = 500 + Math.ceil(mainDocAgents.length / 5) * 80 + 50 + Math.floor(index / cols) * 55;
      nodeMap.set(agent.id, {
        id: agent.id,
        name: agent.metadata?.chunk_range || `Worker ${index + 1}`,
        type: "document_worker",
        x,
        y,
        icon: Cpu,
        color: "#8b5cf6", // purple for workers
      });
    });
    
    // Define connections based on architecture
    const orchId = orchestratorAgents[0]?.id;
    const statsId = statisticsAgents[0]?.id;
    const vizId = visualizationAgents[0]?.id;
    const kgId = kgAgents[0]?.id;
    const llmId = llmAgents[0]?.id;
    
    // Orchestrator → All Core Agents (coordinates everything)
    if (orchId) {
      if (statsId) conns.push({ from: orchId, to: statsId });
      if (vizId) conns.push({ from: orchId, to: vizId });
      if (kgId) conns.push({ from: orchId, to: kgId });
      if (llmId) conns.push({ from: orchId, to: llmId });
    }
    
    // LLM → Orchestrator (queries go through orchestrator)
    if (llmId && orchId) {
      conns.push({ from: llmId, to: orchId });
    }
    
    // Operational Query Agent connections
    const operationalId = operationalQueryAgents[0]?.id;
    if (operationalId) {
      // Orchestrator → Operational Query Agent (orchestrator routes operational queries)
      if (orchId) conns.push({ from: orchId, to: operationalId });
      // Operational Query Agent → Statistics Agent (uses statistics)
      if (statsId) conns.push({ from: operationalId, to: statsId });
      // Operational Query Agent → KG Agent (uses knowledge graph)
      if (kgId) conns.push({ from: operationalId, to: kgId });
    }
    
    // Statistics ↔ Visualization (direct collaboration)
    if (statsId && vizId) {
      conns.push({ from: statsId, to: vizId, bidirectional: true });
    }
    
    // Statistics ↔ KG (direct collaboration)
    if (statsId && kgId) {
      conns.push({ from: statsId, to: kgId, bidirectional: true });
    }
    
    // LLM ↔ KG (direct access for context)
    if (llmId && kgId) {
      conns.push({ from: llmId, to: kgId, bidirectional: true });
    }
    
    // Document Agents → Orchestrator (orchestrator routes to appropriate agents)
    documentAgents.filter((a: any) => a.type !== "document_worker").forEach((docAgent) => {
      if (orchId) conns.push({ from: docAgent.id, to: orchId });
    });
    
    // Worker Agents → Parent Document Agent (aggregation)
    documentAgents.filter((a: any) => a.type === "document_worker").forEach((workerAgent: any) => {
      const parentId = workerAgent.metadata?.parent_document;
      if (parentId) {
        const parentAgentId = `doc_${parentId}`;
        const parentExists = documentAgents.some((a: any) => a.id === parentAgentId);
        if (parentExists) {
          conns.push({ from: workerAgent.id, to: parentAgentId });
        }
      }
    });
    
    // Worker Agents → Knowledge Graph (direct triple construction - KEY ARCHITECTURE!)
    // Workers construct triples directly, bypassing KG Agent for CSV files
    if (kgId) {
      documentAgents.filter((a: any) => a.type === "document_worker").forEach((workerAgent: any) => {
        conns.push({ from: workerAgent.id, to: kgId });
      });
    }
    
    return { nodes: Array.from(nodeMap.values()), connections: conns };
  }, [orchestratorAgents, statisticsAgents, visualizationAgents, kgAgents, llmAgents, documentAgents]);
  
  const width = 800;
  const mainDocCount = documentAgents.filter((a: any) => a.type !== "document_worker").length;
  const workerCount = documentAgents.filter((a: any) => a.type === "document_worker").length;
  const docRows = Math.ceil(mainDocCount / 5);
  const workerRows = Math.ceil(workerCount / 8);
  const height = Math.max(700, 500 + docRows * 80 + workerRows * 55 + 100);
  
  return (
    <Card className="p-6 overflow-auto">
      <h3 className="text-lg font-semibold mb-4">Agent Network Architecture</h3>
      <div className="relative border rounded-lg bg-muted/20" style={{ width, height, minHeight: 600 }}>
        <svg width={width} height={height} className="absolute inset-0">
          {/* Draw connections with curved paths to avoid overlaps */}
          {connections.map((conn, idx) => {
            const fromNode = nodes.find(n => n.id === conn.from);
            const toNode = nodes.find(n => n.id === conn.to);
            if (!fromNode || !toNode) return null;
            
            // Check if this is a document/worker to orchestrator connection
            const isDocToOrch = (fromNode.type === "document" || fromNode.type === "document_worker") && 
                              toNode.type === "orchestrator";
            
            // Use curved path for document connections to avoid overlapping with KG agent
            if (isDocToOrch) {
              // Create a curved path that goes around the KG agent area
              const midX = (fromNode.x + toNode.x) / 2;
              const midY = (fromNode.y + toNode.y) / 2;
              // Offset curve to the right to avoid KG agent (x: 150)
              const controlX = Math.max(midX, 250);
              const controlY = midY - 50; // Curve upward
              
              return (
                <g key={`conn-${idx}`}>
                  <path
                    d={`M ${fromNode.x} ${fromNode.y} Q ${controlX} ${controlY} ${toNode.x} ${toNode.y}`}
                    stroke="#94a3b8"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                    fill="none"
                    opacity="0.6"
                    markerEnd="url(#arrowhead)"
                  />
                </g>
              );
            }
            
            // Straight lines for other connections
            return (
              <g key={`conn-${idx}`}>
                <line
                  x1={fromNode.x}
                  y1={fromNode.y}
                  x2={toNode.x}
                  y2={toNode.y}
                  stroke="#94a3b8"
                  strokeWidth="2"
                  strokeDasharray={conn.bidirectional ? "0" : "5,5"}
                  opacity="0.6"
                  markerEnd={!conn.bidirectional ? "url(#arrowhead)" : undefined}
                />
                {conn.bidirectional && (
                  <line
                    x1={toNode.x}
                    y1={toNode.y}
                    x2={fromNode.x}
                    y2={fromNode.y}
                    stroke="#94a3b8"
                    strokeWidth="2"
                    opacity="0.6"
                    markerEnd="url(#arrowhead)"
                  />
                )}
              </g>
            );
          })}
          
          {/* Arrow marker definition */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 10 3, 0 6" fill="#94a3b8" />
            </marker>
          </defs>
          
          {/* Draw nodes */}
          {nodes.map((node) => {
            const Icon = node.icon;
            const isDocument = node.type === "document";
            const nodeSize = isDocument ? 50 : 70;
            
            return (
              <g key={node.id}>
                {/* Node circle */}
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={nodeSize / 2}
                  fill={node.color}
                  stroke="white"
                  strokeWidth="3"
                  className="drop-shadow-lg"
                />
                
                {/* Icon */}
                <foreignObject
                  x={node.x - nodeSize / 4}
                  y={node.y - nodeSize / 4}
                  width={nodeSize / 2}
                  height={nodeSize / 2}
                >
                  <div className="flex items-center justify-center h-full text-white">
                    <Icon className={`${isDocument ? 'h-4 w-4' : 'h-6 w-6'}`} />
                  </div>
                </foreignObject>
                
                {/* Label with description for core agents - positioned to avoid overlap */}
                <text
                  x={node.x}
                  y={node.y + nodeSize / 2 + 18}
                  textAnchor="middle"
                  className="text-xs font-medium fill-foreground"
                  style={{ fontSize: '11px', fontWeight: '600' }}
                >
                  {isDocument
                    ? node.name.length > 15
                      ? node.name.substring(0, 15) + "..."
                      : node.name
                    : node.name}
                </text>
                {/* Description for core agents (smaller text below name) - more spacing */}
                {!isDocument && node.type !== "document_worker" && (
                  <text
                    x={node.x}
                    y={node.y + nodeSize / 2 + 35}
                    textAnchor="middle"
                    className="text-[9px] fill-muted-foreground"
                    style={{ fontSize: '9px' }}
                  >
                    {node.type === "orchestrator" ? "Coordinates queries" :
                     node.type === "operational_query" ? "Operational analysis" :
                     node.type === "statistics" ? "Statistical analysis" :
                     node.type === "visualization" ? "Data visualization" :
                     node.type === "kg" ? "Knowledge extraction" :
                     node.type === "llm" ? "HR Assistant" : ""}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      
      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-blue-500"></div>
          <span>Statistics Agent</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-green-500"></div>
          <span>Visualization Agent</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-purple-500"></div>
          <span>KG Agent</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-pink-500"></div>
          <span>LLM Agent</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-blue-500"></div>
          <span>Operational Query Agent</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-indigo-500"></div>
          <span>Document Agents</span>
        </div>
        <div className="flex items-center gap-2 ml-4">
          <div className="w-8 h-0.5 bg-slate-400"></div>
          <span>Bidirectional</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-0.5 bg-slate-400 border-dashed border-t-2"></div>
          <span>Unidirectional</span>
        </div>
      </div>
    </Card>
  );
}

