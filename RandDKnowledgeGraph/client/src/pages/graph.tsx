import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { KnowledgeGraphVisualization } from "@/components/KnowledgeGraphVisualization";
import { NodeEditDialog } from "@/components/NodeEditDialog";
import { EdgeEditDialog } from "@/components/EdgeEditDialog";
import { ConnectionCreateDialog } from "@/components/ConnectionCreateDialog";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { useKnowledgeStore } from "@/lib/knowledge-store";
import type { GraphNode, GraphEdge } from "@shared/schema";

export default function GraphPage() {
  const { toast } = useToast();
  const [location] = useLocation();
  const { 
    nodes, 
    edges, 
    updateNode, 
    updateEdge, 
    deleteEdge, 
    addEdge,
    updateNodePosition,
    refreshFacts
  } = useKnowledgeStore();
  
  // Reload facts from backend when navigating to this page
  useEffect(() => {
    if (location === '/graph' && refreshFacts) {
      console.log('GraphPage: Route detected, refreshing facts...');
      refreshFacts().catch(err => {
        console.error('GraphPage: Error refreshing facts:', err);
      });
    }
  }, [location, refreshFacts]);
  
  const [editingNode, setEditingNode] = useState<GraphNode | null>(null);
  const [editingEdge, setEditingEdge] = useState<GraphEdge | null>(null);
  const [isNodeDialogOpen, setIsNodeDialogOpen] = useState(false);
  const [isEdgeDialogOpen, setIsEdgeDialogOpen] = useState(false);
  const [isConnectionDialogOpen, setIsConnectionDialogOpen] = useState(false);
  const [pendingConnection, setPendingConnection] = useState<{ sourceId: string; targetId: string; sourceLabel: string; targetLabel: string } | null>(null);

  const handleNodeClick = (node: GraphNode) => {
    toast({
      title: "Node selected",
      description: `${node.label} (${node.type}, ${node.connections} connections)`,
    });
  };

  const handleNodeEdit = (node: GraphNode) => {
    setEditingNode(node);
    setIsNodeDialogOpen(true);
  };

  const handleNodeMove = (nodeId: string, position: { x: number; y: number; z: number }) => {
    updateNodePosition(nodeId, position);
  };

  const handleSaveNode = async (updatedNode: GraphNode) => {
    try {
      await updateNode(updatedNode.id, updatedNode);
      toast({
        title: "Node updated",
        description: "Changes synced to knowledge base",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update node';
      toast({
        title: "Node not updated",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  const handleEdgeEdit = (edge: GraphEdge) => {
    setEditingEdge(edge);
    setIsEdgeDialogOpen(true);
  };

  const handleSaveEdge = async (updatedEdge: GraphEdge) => {
    try {
      await updateEdge(updatedEdge.id, updatedEdge);
      // Refresh facts after update to ensure sync
      await refreshFacts();
      toast({
        title: "Connection updated",
        description: "Edge label synced to knowledge base",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update connection';
      toast({
        title: "Connection not updated",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  const handleDeleteEdge = async (edgeId: string) => {
    try {
      await deleteEdge(edgeId);
      // Refresh facts after deletion to ensure sync
      await refreshFacts();
      toast({
        title: "Connection deleted",
        description: "Edge removed from graph and knowledge base",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete connection';
      toast({
        title: "Connection not deleted",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  const handleConnectionCreate = (sourceId: string, targetId: string) => {
    const sourceNode = nodes.find(n => n.id === sourceId);
    const targetNode = nodes.find(n => n.id === targetId);
    
    if (!sourceNode || !targetNode) {
      toast({
        title: "Connection not created",
        description: "Source or target node not found",
        variant: "destructive",
      });
      return;
    }
    
    // Show dialog to get relation label
    setPendingConnection({
      sourceId,
      targetId,
      sourceLabel: sourceNode.label,
      targetLabel: targetNode.label
    });
    setIsConnectionDialogOpen(true);
  };

  const handleConnectionDialogSave = async (label: string) => {
    if (!pendingConnection) return;
    
    try {
      await addEdge(pendingConnection.sourceId, pendingConnection.targetId, label);
      // Refresh facts after creation to ensure sync
      await refreshFacts();
      toast({
        title: "Connection created",
        description: "New relationship added to graph and knowledge base",
      });
    } catch (error) {
      // Handle duplicate or other errors
      const errorMessage = error instanceof Error ? error.message : 'Failed to create connection';
      toast({
        title: "Connection not created",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setPendingConnection(null);
      setIsConnectionDialogOpen(false);
    }
  };

  const handleConnectionDialogCancel = () => {
    setPendingConnection(null);
    setIsConnectionDialogOpen(false);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold mb-2">Knowledge Graph</h1>
        <p className="text-muted-foreground">
          Explore, edit, and create connections in your research knowledge network
        </p>
      </div>

      <KnowledgeGraphVisualization
        nodes={nodes}
        edges={edges}
        onNodeClick={handleNodeClick}
        onNodeEdit={handleNodeEdit}
        onNodeMove={handleNodeMove}
        onEdgeEdit={handleEdgeEdit}
        onEdgeDelete={handleDeleteEdge}
        onConnectionCreate={handleConnectionCreate}
      />

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-2">Interactive Graph Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-2">Node Operations</h4>
            <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
              <li>Drag nodes to reposition them in 3D space</li>
              <li>Hover and click edit button to modify labels and types</li>
              <li>Node size indicates connectivity (more connections = larger)</li>
              <li>Click nodes to view details</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">Connection Operations</h4>
            <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
              <li>Click "Connect" button, then click two nodes to link them</li>
              <li>Click existing edges to edit relationship labels</li>
              <li>Delete unwanted connections via edge dialog</li>
              <li>Edge thickness reflects node connectivity</li>
            </ul>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 pt-4 border-t">
          <div>
            <p className="text-2xl font-semibold">{nodes.length}</p>
            <p className="text-sm text-muted-foreground">Total Nodes</p>
          </div>
          <div>
            <p className="text-2xl font-semibold">{edges.length}</p>
            <p className="text-sm text-muted-foreground">Total Edges</p>
          </div>
          <div>
            <p className="text-2xl font-semibold">
              {Math.max(...nodes.map((n) => n.connections), 0)}
            </p>
            <p className="text-sm text-muted-foreground">Max Connections</p>
          </div>
        </div>
      </Card>

      <NodeEditDialog
        node={editingNode}
        open={isNodeDialogOpen}
        onOpenChange={setIsNodeDialogOpen}
        onSave={handleSaveNode}
      />

      <EdgeEditDialog
        edge={editingEdge}
        open={isEdgeDialogOpen}
        onOpenChange={setIsEdgeDialogOpen}
        onSave={handleSaveEdge}
        onDelete={handleDeleteEdge}
      />

      {pendingConnection && (
        <ConnectionCreateDialog
          open={isConnectionDialogOpen}
          onOpenChange={setIsConnectionDialogOpen}
          sourceLabel={pendingConnection.sourceLabel}
          targetLabel={pendingConnection.targetLabel}
          onSave={handleConnectionDialogSave}
          onCancel={handleConnectionDialogCancel}
        />
      )}
    </div>
  );
}
