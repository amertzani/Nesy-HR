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
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { insertFactSchema } from "@shared/schema";
import type { InsertFact } from "@shared/schema";

interface AddFactDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (fact: InsertFact) => void;
}

export function AddFactDialog({ open, onOpenChange, onSave }: AddFactDialogProps) {
  const { register, handleSubmit, setValue, watch, reset, formState: { errors } } = useForm<InsertFact>({
    resolver: zodResolver(insertFactSchema),
    defaultValues: {
      subject: "",
      predicate: "",
      object: "",
      source: "manual",
      details: "",
    },
  });

  const onSubmit = (data: InsertFact) => {
    onSave(data);
    reset();
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent data-testid="dialog-add-fact">
        <DialogHeader>
          <DialogTitle>Add New Fact</DialogTitle>
          <DialogDescription>
            Create a new knowledge fact. It will appear in both the knowledge base and graph.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="subject">Subject</Label>
              <Input
                id="subject"
                {...register("subject")}
                placeholder="e.g., Machine Learning"
                data-testid="input-fact-subject"
              />
              {errors.subject && (
                <p className="text-sm text-destructive">{errors.subject.message}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="predicate">Predicate (Relationship)</Label>
              <Input
                id="predicate"
                {...register("predicate")}
                placeholder="e.g., uses, requires, is_part_of"
                data-testid="input-fact-predicate"
              />
              {errors.predicate && (
                <p className="text-sm text-destructive">{errors.predicate.message}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="object">Object</Label>
              <Input
                id="object"
                {...register("object")}
                placeholder="e.g., Neural Networks"
                data-testid="input-fact-object"
              />
              {errors.object && (
                <p className="text-sm text-destructive">{errors.object.message}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="source">Source</Label>
              <Input
                id="source"
                {...register("source")}
                placeholder="e.g., research_paper.pdf or manual"
                data-testid="input-fact-source"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="details">Details / Comment (Optional)</Label>
              <Textarea
                id="details"
                {...register("details")}
                placeholder="Add additional context, details, or comments about this fact..."
                rows={4}
                className="resize-none"
                data-testid="input-fact-details"
              />
              {errors.details && (
                <p className="text-sm text-destructive">{errors.details.message}</p>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              data-testid="button-cancel"
            >
              Cancel
            </Button>
            <Button type="submit" data-testid="button-save-fact">
              Add Fact
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
