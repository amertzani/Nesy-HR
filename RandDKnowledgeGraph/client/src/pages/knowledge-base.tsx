import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { KnowledgeBaseTable } from "@/components/KnowledgeBaseTable";
import { FactEditDialog } from "@/components/FactEditDialog";
import { AddFactDialog } from "@/components/AddFactDialog";
import { StatsCards } from "@/components/StatsCards";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useKnowledgeStore } from "@/lib/knowledge-store";
import { hfApi } from "@/lib/api-client";
import type { Fact } from "@shared/schema";

export default function KnowledgeBasePage() {
  const { toast } = useToast();
  const [location] = useLocation();
  const { facts, nodes, edges, addFact, updateFact, deleteFact, refreshFacts } = useKnowledgeStore();
  const [documentsCount, setDocumentsCount] = useState(0);
  
  // Reload facts and documents count from backend when navigating to this page
  useEffect(() => {
    console.log('KnowledgeBasePage: useEffect triggered, location:', location);
    // Always refresh when on this route
    if (location === '/knowledge-base') {
      const refresh = async () => {
        try {
          // Refresh facts
          if (refreshFacts) {
            console.log('KnowledgeBasePage: Calling refreshFacts...');
            await refreshFacts();
            console.log('KnowledgeBasePage: refreshFacts completed successfully');
          }
          
          // Fetch documents count (only documents that contributed facts)
          console.log('KnowledgeBasePage: Fetching documents count...');
          const docsResult = await hfApi.getDocuments();
          if (docsResult.success && docsResult.data) {
            // Handle different response formats
            let documents = [];
            if (Array.isArray(docsResult.data)) {
              documents = docsResult.data;
            } else if (docsResult.data.documents && Array.isArray(docsResult.data.documents)) {
              documents = docsResult.data.documents;
            } else if (docsResult.data.docs && Array.isArray(docsResult.data.docs)) {
              documents = docsResult.data.docs;
            }
            
            // Filter: only count documents that have contributed facts
            const documentsWithFacts = documents.filter((doc: any) => (doc.facts_extracted || 0) > 0);
            setDocumentsCount(documentsWithFacts.length);
            console.log(`KnowledgeBasePage: Found ${documentsWithFacts.length} documents with facts (out of ${documents.length} total)`);
          } else {
            console.warn('KnowledgeBasePage: Failed to fetch documents:', docsResult);
            setDocumentsCount(0);
          }
        } catch (err) {
          console.error('KnowledgeBasePage: Error refreshing:', err);
        }
      };
      refresh();
    } else {
      console.log('KnowledgeBasePage: Skipping refresh - location:', location);
    }
  }, [location, refreshFacts]); // Run when location changes
  const [editingFact, setEditingFact] = useState<Fact | null>(null);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);

  const handleEdit = (fact: Fact) => {
    setEditingFact(fact);
    setIsEditDialogOpen(true);
  };

  const handleSave = async (updatedFact: Fact) => {
    try {
      await updateFact(updatedFact.id, updatedFact);
      // Refresh facts after update to ensure sync
      if (refreshFacts) {
        await refreshFacts();
      }
      toast({
        title: "Fact updated",
        description: "Changes synced to knowledge graph",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update fact';
      toast({
        title: "Fact not updated",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteFact(id);
      // Refresh facts after deletion to ensure sync
      if (refreshFacts) {
        await refreshFacts();
      }
      toast({
        title: "Fact deleted",
        description: "Removed from knowledge base and graph",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete fact';
      toast({
        title: "Fact not deleted",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  const handleAddFact = async (newFactData: any) => {
    try {
      await addFact(newFactData);
      toast({
        title: "Fact added",
        description: "New fact created in both knowledge base and graph",
      });
    } catch (error) {
      // Handle duplicate fact error
      const errorMessage = error instanceof Error ? error.message : 'Failed to add fact';
      toast({
        title: "Fact not added",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold mb-2">Knowledge Base</h1>
          <p className="text-muted-foreground">
            View and manage extracted knowledge facts from your research documents
          </p>
        </div>
        <Button onClick={() => setIsAddDialogOpen(true)} data-testid="button-add-fact">
          <Plus className="h-4 w-4 mr-2" />
          Add New Fact
        </Button>
      </div>

      <StatsCards
        documentsCount={documentsCount}
        factsCount={facts.length}
        nodesCount={nodes.length}
        connectionsCount={edges.length}
      />

      <KnowledgeBaseTable
        facts={facts}
        onEdit={handleEdit}
        onDelete={handleDelete}
      />

      <FactEditDialog
        fact={editingFact}
        open={isEditDialogOpen}
        onOpenChange={setIsEditDialogOpen}
        onSave={handleSave}
      />

      <AddFactDialog
        open={isAddDialogOpen}
        onOpenChange={setIsAddDialogOpen}
        onSave={handleAddFact}
      />
    </div>
  );
}
