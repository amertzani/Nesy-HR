import { useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BarChart3, TrendingUp, Network, Bot, FileText, Cpu } from "lucide-react";

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
  documentAgents: any[];
}

export function AgentNetwork({
  orchestratorAgents,
  statisticsAgents,
  visualizationAgents,
  kgAgents,
  llmAgents,
  documentAgents,
}: AgentNetworkProps) {
  // Define node positions and connections
  const { nodes, connections } = useMemo(() => {
    const nodeMap: Map<string, AgentNode> = new Map();
    const conns: Connection[] = [];
    
    // Core agent positions (arranged in a network layout)
    const corePositions = {
      orchestrator: { x: 400, y: 50 },
      statistics: { x: 200, y: 150 },
      visualization: { x: 400, y: 150 },
      kg: { x: 200, y: 300 },
      llm: { x: 400, y: 300 },
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
    
    // Add Document Agents (arranged in a row below)
    // Separate main document agents from worker agents
    const mainDocAgents = documentAgents.filter((a: any) => a.type !== "document_worker");
    const workerAgents = documentAgents.filter((a: any) => a.type === "document_worker");
    
    // Main document agents
    mainDocAgents.forEach((agent, index) => {
      const x = 100 + (index % 5) * 120;
      const y = 450 + Math.floor(index / 5) * 80;
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
    
    // Worker agents (arranged below main document agents, smaller)
    workerAgents.forEach((agent, index) => {
      const x = 100 + (index % 8) * 100;
      const y = 450 + Math.ceil(mainDocAgents.length / 5) * 80 + 60 + Math.floor(index / 8) * 60;
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
    
    // Worker Agents → Parent Document Agent
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
    
    return { nodes: Array.from(nodeMap.values()), connections: conns };
  }, [orchestratorAgents, statisticsAgents, visualizationAgents, kgAgents, llmAgents, documentAgents]);
  
  const width = 800;
  const height = Math.max(600, 450 + Math.ceil(documentAgents.length / 5) * 80);
  
  return (
    <Card className="p-6 overflow-auto">
      <h3 className="text-lg font-semibold mb-4">Agent Network Architecture</h3>
      <div className="relative border rounded-lg bg-muted/20" style={{ width, height, minHeight: 600 }}>
        <svg width={width} height={height} className="absolute inset-0">
          {/* Draw connections */}
          {connections.map((conn, idx) => {
            const fromNode = nodes.find(n => n.id === conn.from);
            const toNode = nodes.find(n => n.id === conn.to);
            if (!fromNode || !toNode) return null;
            
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
                
                {/* Label */}
                <text
                  x={node.x}
                  y={node.y + nodeSize / 2 + 15}
                  textAnchor="middle"
                  className="text-xs font-medium fill-foreground"
                >
                  {isDocument
                    ? node.name.length > 15
                      ? node.name.substring(0, 15) + "..."
                      : node.name
                    : node.name}
                </text>
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

