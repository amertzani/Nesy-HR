import { Upload } from "lucide-react";
import { useState, useCallback } from "react";
import { Card } from "@/components/ui/card";

interface FileUploadZoneProps {
  onFilesSelected: (files: File[]) => void;
  acceptedTypes?: string;
}

export function FileUploadZone({ onFilesSelected, acceptedTypes = ".pdf,.docx,.txt,.csv,.pptx" }: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    onFilesSelected(files);
  }, [onFilesSelected]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      onFilesSelected(files);
    }
  }, [onFilesSelected]);

  return (
    <Card
      className={`border-2 border-dashed transition-colors ${
        isDragging ? "border-primary bg-accent" : "border-border"
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      data-testid="zone-file-upload"
    >
      <label className="flex min-h-64 cursor-pointer flex-col items-center justify-center p-8">
        <Upload className="h-12 w-12 text-muted-foreground mb-4" />
        <p className="text-base font-medium mb-2">
          Drop research documents here
        </p>
        <p className="text-sm text-muted-foreground mb-4">
          or click to browse
        </p>
        <p className="text-xs text-muted-foreground font-mono">
          Supported: PDF, DOCX, TXT, CSV, PPTX
        </p>
        <input
          type="file"
          multiple
          accept={acceptedTypes}
          onChange={handleFileInput}
          className="hidden"
          data-testid="input-file"
        />
      </label>
    </Card>
  );
}
