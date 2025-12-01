import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useForm } from "react-hook-form";

interface ConnectionCreateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceLabel: string;
  targetLabel: string;
  onSave: (label: string) => void;
  onCancel: () => void;
}

interface ConnectionFormData {
  label: string;
}

export function ConnectionCreateDialog({ 
  open, 
  onOpenChange, 
  sourceLabel, 
  targetLabel, 
  onSave, 
  onCancel 
}: ConnectionCreateDialogProps) {
  const { register, handleSubmit, reset } = useForm<ConnectionFormData>({
    defaultValues: { label: "" },
  });

  useEffect(() => {
    if (open) {
      reset({ label: "" });
    }
  }, [open, reset]);

  const onSubmit = (data: ConnectionFormData) => {
    const label = data.label.trim() || "related_to";
    onSave(label);
    onOpenChange(false);
  };

  const handleCancel = () => {
    onCancel();
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent data-testid="dialog-create-connection">
        <DialogHeader>
          <DialogTitle>Create Connection</DialogTitle>
          <DialogDescription>
            Specify the relationship between {sourceLabel} and {targetLabel}
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="label">Relationship Label</Label>
              <Input
                id="label"
                {...register("label")}
                placeholder="e.g., uses, requires, is_part_of, related_to"
                data-testid="input-connection-label"
                autoFocus
              />
              <p className="text-xs text-muted-foreground">
                Leave empty to use "related_to" as default
              </p>
            </div>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>Connection: <span className="font-mono">{sourceLabel} → {targetLabel}</span></p>
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={handleCancel}
              data-testid="button-cancel-connection"
            >
              Cancel
            </Button>
            <Button type="submit" data-testid="button-create-connection">
              Create Connection
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

