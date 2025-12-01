import { useState, useEffect } from "react";
import { FileUploadZone } from "@/components/FileUploadZone";
import { DocumentList } from "@/components/DocumentList";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { hfApi } from "@/lib/api-client";
import { useKnowledgeStore } from "@/lib/knowledge-store";
import type { Document } from "@shared/schema";

// Store File objects alongside document metadata
interface DocumentWithFile extends Document {
  file?: File; // Store the actual File object
}

export default function UploadPage() {
  const [documents, setDocuments] = useState<DocumentWithFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const { toast } = useToast();
  const { refreshFacts } = useKnowledgeStore();

  // Fetch existing documents from backend on mount to show agent_id
  useEffect(() => {
    async function loadDocuments() {
      try {
        const result = await hfApi.getDocuments();
        if (result.success && result.data?.documents) {
          const backendDocs = result.data.documents as Document[];
          console.log('✅ Upload: Loaded documents from backend:', backendDocs);
          setDocuments(backendDocs.map((doc) => ({
            ...doc,
            status: doc.status || "completed",
          })));
        }
      } catch (error) {
        console.error('Error loading documents:', error);
      }
    }
    loadDocuments();
  }, []);

  const handleFilesSelected = (files: File[]) => {
    const newDocuments: DocumentWithFile[] = files.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      type: file.name.split('.').pop() as any,
      size: file.size,
      uploadedAt: new Date().toISOString(),
      status: "pending" as const,
      file: file, // Store the actual File object
    }));

    setDocuments((prev) => [...prev, ...newDocuments]);
    toast({
      title: "Files added",
      description: `${files.length} file(s) ready for processing`,
    });
  };

  const handleRemove = (id: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== id));
  };

  const handleProcess = async () => {
    const pendingDocs = documents.filter((doc) => doc.status === "pending");
    if (pendingDocs.length === 0) {
      return;
    }

    setIsProcessing(true);
    setDocuments((prev) =>
      prev.map((doc) =>
        doc.status === "pending" ? { ...doc, status: "processing" as const } : doc
      )
    );
    
    toast({
      title: "Processing started",
      description: `Uploading and processing ${pendingDocs.length} document(s)...`,
    });

    try {
      // Get the actual File objects from pending documents
      const fileObjects: File[] = [];
      for (const doc of pendingDocs) {
        if (doc.file) {
          fileObjects.push(doc.file);
        } else {
          console.warn(`Document ${doc.name} has no file object, skipping`);
        }
      }

      if (fileObjects.length === 0) {
        throw new Error("No files to upload");
      }

      console.log(`Uploading ${fileObjects.length} file(s) to backend...`);
      const result = await hfApi.uploadDocuments(fileObjects);
      
      if (result.success) {
        // Fetch documents from backend to get agent_id and other metadata
        console.log('🔄 Upload: Fetching documents from backend to get agent_id...');
        const docsResult = await hfApi.getDocuments();
        if (docsResult.success && docsResult.data?.documents) {
          const backendDocs = docsResult.data.documents as Document[];
          console.log('✅ Upload: Fetched documents from backend:', backendDocs);
          
          // Update local documents with backend data (including agent_id)
          setDocuments((prev) => {
            const updated = prev.map((localDoc) => {
              const backendDoc = backendDocs.find((bd) => bd.name === localDoc.name);
              if (backendDoc) {
                // Merge backend data (includes agent_id) with local file reference
                return {
                  ...backendDoc,
                  file: localDoc.file, // Keep file reference if needed
                  status: "completed" as const,
                };
              }
              // If not found in backend, mark as completed anyway
              return { ...localDoc, status: "completed" as const };
            });
            return updated;
          });
        } else {
          // Fallback: just mark as completed
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.status === "processing" ? { ...doc, status: "completed" as const } : doc
          )
        );
        }
        
        // Refresh facts to show newly extracted facts
        if (refreshFacts) {
          console.log('🔄 Upload: Refreshing facts after document processing...');
          await refreshFacts();
          console.log('✅ Upload: Facts refreshed');
        } else {
          console.warn('⚠️ Upload: refreshFacts not available');
        }
        
        toast({
          title: "Processing complete",
          description: result.data?.message || "Knowledge extraction finished successfully",
        });
      } else {
        // Mark as failed
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.status === "processing" ? { ...doc, status: "pending" as const } : doc
          )
        );
        toast({
          title: "Processing failed",
          description: result.error || "Failed to process documents",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error processing documents:", error);
      setDocuments((prev) =>
        prev.map((doc) =>
          doc.status === "processing" ? { ...doc, status: "pending" as const } : doc
        )
      );
      toast({
        title: "Processing failed",
        description: error instanceof Error ? error.message : "Failed to process documents",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const hasPendingDocs = documents.some((doc) => doc.status === "pending");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold mb-2">Upload Documents</h1>
        <p className="text-muted-foreground">
          Upload research documents to extract knowledge and build your knowledge graph
        </p>
      </div>

      <FileUploadZone onFilesSelected={handleFilesSelected} />

      {documents.length > 0 && (
        <>
          <DocumentList documents={documents} onRemove={handleRemove} />
          <div className="flex gap-2">
            <Button
              onClick={handleProcess}
              disabled={!hasPendingDocs || isProcessing}
              data-testid="button-process-documents"
            >
              {isProcessing ? "Processing..." : "Process Documents"}
            </Button>
            <Button
              variant="outline"
              onClick={() => setDocuments([])}
              data-testid="button-clear-all"
            >
              Clear All
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
